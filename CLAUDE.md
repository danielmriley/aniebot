# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build          # compile
cargo run            # run (requires .env, LM Studio running, Telegram bot token)
cargo test           # run tests (unit tests in tools.rs cover cron utilities)
cargo test -- cron   # run a subset of tests by name filter
```

Logging is controlled via `RUST_LOG` (e.g. `RUST_LOG=debug cargo run`). The default level is `info`.

Configuration is loaded from `.env` at startup via `dotenvy`. Copy `.env.example` to `.env` and fill in the required values (`TELEGRAM_TOKEN`, `LM_STUDIO_URL`, `MODEL_NAME`, `ALLOWED_USER_ID`).

## Architecture

AnieBot is a Telegram bot that uses a local LLM (via LM Studio, OpenAI-compatible API) as an orchestrator. The LLM is never given direct internet or filesystem access — it calls **tools**, which the Rust process dispatches.

### Request flow

```
Telegram message
  → bot.rs (teloxide dispatcher)
  → orchestrator::process_message()
  → try_process(): load history + memory, build system prompt
  → run_agentic_loop(): iterative LLM ↔ tool dispatch until terminal call
      ├── single-call path (one tool per LLM response)
      └── multi-call path (batched tools, with intra-batch dedup)
  → persist new messages to conversation history
  → return reply text → bot.rs sends to Telegram
```

### Background execution

Three background entry points all go through the same `run_agentic_loop()`:

| Entry point | Trigger | Purpose |
|---|---|---|
| `run_heartbeat()` | Cron (`HEARTBEAT_CRON`, default hourly) | Check active interests, work curiosity queue, deliver proactive messages |
| `run_interest_check()` | Per-interest cron (set when registering) | Focused check for a single interest |
| `run_consolidation()` | After N episodic entries (`CONSOLIDATION_THRESHOLD`) | Reflect on recent events, update beliefs, decay stale interests |

All three acquire `BACKGROUND_GUARD` (`OnceLock<tokio::sync::Mutex<()>>`) via `try_lock()` at startup — concurrent runs are skipped with a log message rather than queued.

### Memory layers

| Layer | File | Description |
|---|---|---|
| Core memory | `data/core_memory.json` | Identity, beliefs, user profile, active interests, curiosity queue. Written atomically via tmp→rename. |
| Episodic memory | `data/episodic.jsonl` | Append-only log of events, importance-tagged. Entries above importance 3 are promoted to `data/archival.jsonl`. |
| Conversation history | `data/conversations/{chat_id}.json` | Per-chat rolling window (up to `HISTORY_MAX_STORED` messages). |
| Agenda | `data/agenda.json` | Pending/in-progress tasks the bot has committed to. |
| Schedules | `data/schedules.json` | User-created cron jobs persisted across restarts. |

### Key modules

- **`orchestrator.rs`** — everything: system prompt assembly, `run_agentic_loop()`, heartbeat/interest/consolidation logic, memory eval, session summary, synthesis. The largest file; read it first.
- **`tools.rs`** — tool descriptors (JSON schema sent to LLM) + `dispatch_tool_call()` handler. Also contains cron helpers (`cron_local_to_utc_with_offset`, `advance_dow`, `validate_cron_fields`).
- **`scheduler.rs`** — `tokio-cron-scheduler` wrapper. All mutation goes through an `mpsc` channel to a single worker task; `SchedulerHandle` is just the channel sender and is freely cloneable.
- **`cli_wrapper.rs`** — spawns `copilot --allow-all` in `workspace_dir`. A static `CLI_LOCK` (`OnceLock<Mutex<()>>`) serialises all CLI invocations because Copilot corrupts its session file under concurrency.
- **`core_memory.rs`** — `CoreMemory` struct + load/save. `Interest` has `health` (0–100, decayed during consolidation) and `last_seen_date` (used to sort oldest-first for heartbeat capping).
- **`config.rs`** — all env vars with defaults. Add new tunables here.

### Agentic loop internals (`run_agentic_loop`)

The loop runs up to `max_iters` iterations. Each iteration:
1. Optionally compresses context (fires once when `messages.len() > context_compress_threshold`)
2. Calls LM Studio with the current message list and tool definitions
3. Dispatches tool calls — **single-call path** or **multi-call path** depending on response
4. Terminates on `reply_to_user`, `nothing`, or `reflect(done=true)`

**Anti-stall mechanisms** (all live in `run_agentic_loop`):
- Cross-iter fingerprint dedup: repeated `(tool_name, args)` across iterations injects a warning and blocks re-dispatch
- Per-tool call counter: warns at 8 calls of the same tool per turn, hard-stops at 15
- Intra-batch dedup: identical `(tool_name, args)` pairs within a single multi-call response are suppressed with a synthetic result
- Budget warning: injected at 80% iteration consumption

**Tool dispatch:**
- `PARALLEL_SAFE_TOOLS` (`fetch_url`, `recall`, `list_*`) are dispatched concurrently via `JoinSet` when the entire batch is safe
- All write tools dispatch sequentially to avoid read-modify-write races on JSON files
- Terminal tools (`reply_to_user`, `nothing`, `reflect`) always run last

### Cron expressions

Cron is 6-field Quartz format (`SEC MIN HRS DOM MON DOW`). 5-field standard cron is auto-promoted by prepending `"0 "`. After promotion, `validate_cron_fields()` checks hours (0–23) and minutes (0–59). `cron_local_to_utc_with_offset()` converts local time to UTC; range DOW expressions (`Mon-Fri`) cannot be auto-adjusted across day boundaries and emit a `tracing::warn!`.

### Tool sets

Three distinct tool sets are passed to `run_agentic_loop()` depending on context:
- `tool_definitions()` — full set for user conversations
- `heartbeat_tool_definitions()` — subset for background heartbeat (no `schedule_*`, no `remember`)
- `memory_eval_tools()` — only `update_core_memory` + `nothing` (used in `run_memory_eval`)

### Personality / identity

`personality.md` seeds `core_memory.json` on first run. After that, the LLM updates core memory directly via the `update_core_memory` tool — `personality.md` is not re-read.
