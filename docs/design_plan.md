# Plan: AnieBot MVP Implementation

## TL;DR

Build a single-binary Rust Telegram bot (`aniebot`) that uses LM Studio as a local orchestrator to decide when to reply directly vs. delegate to `copilot -p` (or `gemini -p`). Conversation history is stored per chat_id indefinitely; the LM Studio prompt receives a windowed slice (last 20 exchanges + last 5 memory entries) to stay within local model context limits. Auth is enforced by a single `ALLOWED_USER_ID` env var.

---

## Decisions & Scope

- **Primary delegate**: `copilot -p "<task>" --allow-all --silent --no-ask-user` in `WORKSPACE_DIR`
- **Fallback delegate**: `gemini -p "<task>" --yolo -s` (future, not MVP)
- **Authorization**: Single owner via `ALLOWED_USER_ID` env var — drop all other messages silently
- **Workspace dir**: `WORKSPACE_DIR` env var — what copilot/gemini run inside
- **Crate name**: `aniebot`
- **Context window strategy**: Store full history on disk; inject last 20 exchanges + last 5 memory entries into LM Studio prompt
- **Memory**: Write-only log for MVP (raw interactions); extracted-facts layer deferred to iteration 2
- **Summarization**: Deferred to iteration 2 (small model like Qwen3.5 4B)
- **Permissions**: MVP uses `--allow-all` (YOLO mode); safe mode with Telegram approval deferred to roadmap item 4
- **Async I/O**: All file I/O via `tokio::fs` — no blocking calls in async context
- **CLI timeout**: 120s configurable via `CLI_TIMEOUT_SECS` env var

---

## Project Structure

```
aniebot/
├── Cargo.toml
├── .env                          # secrets + config (gitignored)
├── .env.example                  # committed template
├── personality.md                # loaded at runtime for each prompt
├── data/
│   ├── conversations/            # one JSON file per chat_id
│   └── memory.json               # interaction log
└── src/
    ├── main.rs                   # startup, env, bot init
    ├── config.rs                 # typed Config struct from env
    ├── bot.rs                    # teloxide dispatcher + auth gate
    ├── orchestrator.rs           # LM Studio prompt builder + call
    ├── cli_wrapper.rs            # copilot/gemini subprocess runner
    └── memory.rs                 # conversation store + interaction log
```

---

## Phases

### Phase 1: Scaffold

1. Create `Cargo.toml` with all dependencies
2. Create `src/config.rs` — typed `Config` struct loaded from env vars (`TELEGRAM_TOKEN`, `LM_STUDIO_URL`, `MODEL_NAME`, `ALLOWED_USER_ID`, `WORKSPACE_DIR`, `CLI_TIMEOUT_SECS`)
3. Create `src/main.rs` — load dotenvy, init tracing, load `Config`, create `Bot`, call `bot::run`
4. Create `src/bot.rs` — teloxide `Dispatcher`, auth check (compare `msg.from().map(|u| u.id)` against `Config::allowed_user_id`), wire to orchestrator
5. Create `.env.example` and `personality.md` seed file

### Phase 2: Memory Layer

6. Create `src/memory.rs`:
   - `ConversationMessage { role, content, timestamp }` — serializable struct
   - `load_history(chat_id) -> Vec<ConversationMessage>` — reads `data/conversations/<chat_id>.json` with `tokio::fs`, returns empty vec if missing
   - `append_messages(chat_id, messages)` — appends to same file (read-modify-write with `tokio::fs`)
   - `store_interaction(chat_id, user_msg, assistant_reply)` — appends to `data/memory.json` with timestamp
   - `load_recent_memory(n) -> Vec<MemoryEntry>` — returns last N entries from memory.json

### Phase 3: Orchestrator

7. Create `src/orchestrator.rs`:
   - `process_message(config, chat_id, user_input) -> String`
   - Reads `personality.md` with `tokio::fs::read_to_string`
   - Calls `memory::load_history(chat_id)` → slices last 20 exchanges
   - Calls `memory::load_recent_memory(5)` → formats as bullet points
   - Builds LM Studio messages array: system prompt (personality + memory bullets + JSON schema instructions with few-shot example) + conversation history as user/assistant turns + current user message
   - POSTs to `{LM_STUDIO_URL}/chat/completions` at temperature 0.0
   - Parses JSON response: `{ "action": "direct_reply"|"delegate_cli", "reply": "...", "task": "..." }`
   - On JSON parse failure: retries once with explicit correction message; on second failure, treats raw content as a direct reply (graceful degradation)
   - If `delegate_cli`: calls `cli_wrapper::run`, formats result
   - Calls `memory::append_messages` and `memory::store_interaction`
   - Returns final reply string

### Phase 4: CLI Wrapper

8. Create `src/cli_wrapper.rs`:
   - `run(config, task: &str) -> Result<String>`
   - Constructs: `copilot -p "<task>" --allow-all --silent --no-ask-user`
   - Sets working directory to `config.workspace_dir`
   - Wraps `cmd.output()` in `tokio::time::timeout(Duration::from_secs(config.cli_timeout_secs))`
   - Returns stdout as String; on timeout returns descriptive error
   - Sanitizes: task passed as a single argument (no shell interpolation — use `.arg()` not `.args(["-c", ...])`)

### Phase 5: Wire & Verify

9. `bot.rs` calls `orchestrator::process_message` and sends the reply
2. Run `cargo build` and fix any compile errors
3. Manual end-to-end test via Telegram

---

## Relevant Files (all new)

- `Cargo.toml` — deps: teloxide 0.13, tokio full, reqwest 0.12 + json, serde + derive, serde_json, dotenvy, anyhow, tracing, tracing-subscriber + env-filter, chrono
- `src/config.rs` — `Config::from_env()` using `std::env::var`
- `src/memory.rs` — `ConversationMessage`, `MemoryEntry`, async file I/O via `tokio::fs`
- `src/orchestrator.rs` — LM Studio client, prompt construction, JSON parse with retry
- `src/cli_wrapper.rs` — `tokio::process::Command` + `tokio::time::timeout`
- `src/bot.rs` — teloxide dispatcher, auth gate
- `src/main.rs` — entrypoint

---

## LM Studio Prompt Schema

System message instructs the model to output ONLY:

```json
{"action":"direct_reply","reply":"..."}
// or
{"action":"delegate_cli","task":"..."}
```

Includes one few-shot example each. Temperature 0.0. `max_tokens: 400`.

---

## .env Variables

```
TELEGRAM_TOKEN=
LM_STUDIO_URL=http://localhost:1234/v1
MODEL_NAME=qwen3.5-9b-claude-4.6-opus-reasoning-distilled
ALLOWED_USER_ID=
WORKSPACE_DIR=
CLI_TIMEOUT_SECS=120
```

---

## Verification

1. `cargo build --release` — must compile clean, zero warnings
2. Set `.env` with real values, run `cargo run`
3. Send a message from a non-authorized Telegram account → bot must not respond
4. Send "hello" from authorized account → bot replies directly (no CLI invocation)
5. Send "use copilot to list files in the workspace" → bot delegates to `copilot -p`, reply includes output
6. Check `data/conversations/<chat_id>.json` — should contain full history
7. Check `data/memory.json` — should contain interaction entries
8. Send a follow-up referencing prior message → bot should have context (confirming history injection works)
9. Kill and restart bot → send follow-up again → context should survive restart (disk persistence)
