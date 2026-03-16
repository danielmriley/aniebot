# AnieBot: Post-MVP Roadmap

## Iteration Order & Rationale

| # | Name | Why this order |
|---|---|---|
| Quick win | GitHub MCP tools | One line in `cli_wrapper.rs`, unlocks full GitHub API for copilot today |
| 1 | Proactive Scheduling | User's #1 priority. Fully independent — only needs `tokio-cron-scheduler` |
| 1b | Dynamic Scheduling | Natural extension of 1 — lets the bot schedule its own jobs via chat; needs persistence before CLI makes sense |
| 2 | Local Control CLI | Enables local testing/control without Telegram; needed before permissions UX makes sense |
| 3 | Permissions System | Required before bot can run autonomously without oversight |
| 4 | Self-Evolving Personality | Fun, impactful, builds on a stable base |
| 5 | Letta Hierarchical Memory | Most complex, biggest dep — best saved for last |

---

## Quick Win: GitHub MCP Tools

**Goal:** Give copilot access to the full GitHub MCP tool suite (create PRs, open issues, list repos, etc.) without any Rust changes except one argument.

**Change:** In `src/cli_wrapper.rs`, add `--enable-all-github-mcp-tools` to the copilot invocation:

```rust
cmd.arg("-p")
    .arg(task)
    .arg("--allow-all")
    .arg("--silent")
    .arg("--no-ask-user")
    .arg("--enable-all-github-mcp-tools")
    .current_dir(&config.workspace_dir);
```

**Impact:** Copilot can now create PRs, open issues, clone repos, comment on issues — all from a Telegram message.

---

## Iteration 1: Proactive Scheduling

**Goal:** Let AnieBot initiate messages on a schedule without user input — morning briefings, health checks, reminders.

### Key Design Decisions
- Use `tokio-cron-scheduler` (pure async, no system cron dependency)
- `ALLOWED_USER_ID` doubles as the owner's Telegram `chat_id` — use it directly for proactive sends
- Cron expressions configurable via env vars (with sensible defaults)
- Scheduler runs as a `tokio::spawn`-ed sibling task alongside the bot dispatcher

### New Files
- `src/scheduler.rs` — `pub async fn start(bot: Bot, config: Arc<Config>)`

### Config Changes (`src/config.rs`)
Add fields:
```rust
pub morning_summary_cron: String,  // default: "0 8 * * *" (08:00 daily)
pub health_check_interval_mins: u64, // default: 30
```

### `Cargo.toml` Addition
```toml
tokio-cron-scheduler = { version = "0.13", features = ["signal"] }
```

### `src/scheduler.rs` Outline
```rust
pub async fn start(bot: Bot, config: Arc<Config>) {
    let sched = JobScheduler::new().await.unwrap();

    // Job 1: Morning summary
    let bot1 = bot.clone(); let cfg1 = config.clone();
    sched.add(Job::new_async(&config.morning_summary_cron, move |_, _| {
        let bot = bot1.clone(); let cfg = cfg1.clone();
        Box::pin(async move {
            let reply = orchestrator::process_message(
                cfg.clone(), cfg.allowed_user_id as i64,
                "Give me a proactive morning summary."
            ).await;
            let _ = bot.send_message(ChatId(cfg.allowed_user_id as i64), reply).await;
        })
    }).unwrap()).await.unwrap();

    // Job 2: LM Studio health check
    // Pings /v1/models every N minutes; sends one warning per outage, resets on recovery
    // ...

    sched.start().await.unwrap();
}
```

### `src/main.rs` Change
```rust
tokio::spawn(scheduler::start(bot.clone(), config.clone()));
bot::run(bot, config).await;
```

### Verification
1. Set `MORNING_SUMMARY_CRON="* * * * *"` temporarily → bot sends a message every minute
2. Kill LM Studio → health check fires within 30min and sends a warning
3. Restore LM Studio → next check sends recovery notice
4. Set cron back to `"0 8 * * *"`, redeploy

---

## Iteration 1b: Dynamic Scheduling

**Goal:** Let the bot (and the user via chat) create, list, and delete scheduled jobs at runtime — without restarting the process or editing env vars.

### Key Design Decisions
- Expose `Arc<JobScheduler>` from `scheduler::start()` so the orchestrator can add/remove jobs
- Persist job specs to `data/schedules.json` so jobs survive a restart (reloaded during `scheduler::start()`)
- Add a new orchestrator action `"schedule_task"` to the LM response schema
- Jobs created via chat are one-shot or recurring and carry a human-readable label
- The `JobScheduler` handle is passed into the bot handler via dptree alongside `Config`

### New Orchestrator Action
Extend `OrchestratorResponse` in `src/orchestrator.rs`:
```rust
struct OrchestratorResponse {
    action: String, // "direct_reply" | "delegate_cli" | "schedule_task" | "list_schedules" | "delete_schedule"
    reply: Option<String>,
    task: Option<String>,
    // --- schedule_task fields ---
    cron: Option<String>,   // 6-field quartz cron, e.g. "0 30 9 * * Mon-Fri"
    label: Option<String>,  // human-readable name, e.g. "standup reminder"
    schedule_id: Option<String>, // for delete_schedule
}
```

Dispatch logic additions:
- `"schedule_task"` → validate cron, create `Job::new_async(...)`, add to scheduler, append to `data/schedules.json`, reply with confirmation + assigned UUID
- `"list_schedules"` → read `data/schedules.json`, format as human-readable list, return as `direct_reply`
- `"delete_schedule"` → remove job from scheduler by UUID, remove from `data/schedules.json`, confirm

### Persistence Schema (`data/schedules.json`)
```json
[
  {
    "id": "<uuid>",
    "label": "standup reminder",
    "cron": "0 30 9 * * Mon-Fri",
    "task": "Remind me about standup.",
    "created_at": "2026-03-15T08:00:00Z"
  }
]
```

### New / Changed Files
- `src/scheduler.rs` — `start()` now returns `Arc<JobScheduler>`; add `add_job()`, `remove_job()`, `load_persisted_jobs()` helpers
- `src/orchestrator.rs` — extend `OrchestratorResponse`, add dispatch arms for the three new actions
- `src/bot.rs` — inject `Arc<JobScheduler>` via dptree so the handler can pass it to the orchestrator
- `src/main.rs` — capture returned `Arc<JobScheduler>` from `tokio::spawn` result (or restructure to get handle before spawn)
- `data/schedules.json` — created on first dynamic job; absent on fresh install is fine

### Prompt Additions
Add examples to the system prompt in `orchestrator.rs`:
```
User: "remind me every weekday at 9:30 to do standup"
{"action":"schedule_task","cron":"0 30 9 * * Mon-Fri","label":"standup reminder","task":"Remind me to do standup."}

User: "show my scheduled jobs"
{"action":"list_schedules"}

User: "delete the standup reminder"
{"action":"delete_schedule","schedule_id":"<uuid>"}
```

### Verification
1. Send "remind me every minute to drink water" → bot confirms with a UUID, fires after ~1 min
2. Send "show my scheduled jobs" → bot lists the job with its label and cron
3. Restart the bot → job reloads from `data/schedules.json` and still fires
4. Send "delete the water reminder" → bot confirms, job no longer fires
5. Send a bad cron like `"99 99 99 * * *"` → bot replies with a validation error, nothing is added

---

## Iteration 2: Local Control CLI

**Goal:** Add a `clap`-based CLI so the bot can be controlled from the terminal — useful for testing, scripting, and running one-off tasks without Telegram.

### `Cargo.toml` Addition
```toml
clap = { version = "4", features = ["derive"] }
```

### CLI Subcommands
```
aniebot run                          # starts the Telegram bot (current default)
aniebot ask "<message>"              # send a message through the orchestrator, print reply
aniebot copilot "<task>"             # invoke cli_wrapper::run directly, print output
aniebot history <chat_id>            # dump conversation history as JSON
aniebot memory [--last N]            # dump recent memory entries
aniebot clear-history <chat_id>      # wipe conversation for a chat
```

### Design
- `src/main.rs` parses `Args` with clap before any async setup
- Non-`run` subcommands build a minimal `Config`, then call the relevant module function directly
- All output goes to stdout; errors to stderr with non-zero exit code

---

## Iteration 3: Permissions System + Audit Log

**Goal:** Replace the `--allow-all` YOLO flag with a per-action approval flow so the bot can run semi-autonomously without silent, unchecked shell execution.

### Design
- New enum `Permission { AllowAll, AskPerAction, Deny }`
- Stored in `data/permissions.json` — per-operation-type rules (e.g. `file_write: AskPerAction`, `shell_exec: AskPerAction`, `web_fetch: AllowAll`)
- When copilot wants to do a restricted action, bot sends an inline keyboard to Telegram with ✅ Allow / ❌ Deny
- Copilot invocation waits on a `tokio::sync::oneshot` channel for the approval signal
- Timeout: if no response in N seconds, auto-deny and notify

### Audit Log
- `data/audit.json` — append-only log of every copilot invocation: timestamp, task, permissions granted, outcome (success/failure/denied)
- Queryable via `aniebot audit [--last N]` (Iteration 2 CLI)

### New Files
- `src/permissions.rs` — load/save rules, `check_permission(action_type) -> Permission`
- `src/audit.rs` — `log_action(...)`, `load_recent(n)`

---

## Iteration 4: Self-Evolving Personality

**Goal:** Let the bot update its own `personality.md` based on interactions — evolving tone, adding learned preferences, and reflecting user feedback.

### Design
- After every N interactions (configurable), delegate to LM Studio with a meta-prompt: "Given the last N interactions, suggest 1–3 specific updates to the personality file that would make you more useful or aligned with the user's style."
- LM Studio returns a diff-like suggestion: `{ "action": "update_personality", "changes": [...] }`
- Bot sends the proposed changes to Telegram for approval (inline keyboard)
- On approval, atomically writes the new `personality.md`
- Version history kept in `data/personality_history/` (one timestamped copy per accepted update)

### Config Addition
```
PERSONALITY_EVOLVE_EVERY=50   # interactions between evolution attempts
```

### Safeguard
- Maximum personality file size enforced (e.g. 4KB) to prevent runaway growth
- User can always send `/reset-personality` to restore from `personality_history/`

---

## Iteration 5: Letta Hierarchical Memory

**Goal:** Replace the flat `memory.json` interaction log with a Letta-powered hierarchical memory system that extracts facts, maintains a user model, and enables semantic recall.

### Why Letta
- Provides structured memory tiers: in-context, archival, recall storage
- Supports external embeddings + vector search out of the box
- Has a REST API — no Rust SDK needed, just `reqwest` calls

### Design
- Run Letta as a sidecar service (Docker or native binary)
- Replace `memory::store_interaction` with a Letta memory write
- Replace `memory::load_recent_memory` with a Letta semantic recall query keyed on the current user message
- Extracted facts (e.g. "user prefers Rust over Python") automatically promoted to archival memory by Letta
- `LETTA_URL` env var (default: `http://localhost:8283`)

### Migration
- Existing `data/memory.json` can be bulk-imported into Letta at startup via a one-time migration flag: `--import-legacy-memory`
- Fall back to flat file if Letta is unreachable (graceful degradation)

### Config Addition
```
LETTA_URL=http://localhost:8283
LETTA_AGENT_ID=                # created once, persisted in .env
```
