# Parallel LLM Calls in the LM Studio Harness — Technical Plan

**Date:** 2026-03-19  
**Scope:** `src/orchestrator.rs`, `src/config.rs`, `src/tools.rs`, `src/cli_wrapper.rs`  
**Baseline:** AnieBot v0.1.0 — single-binary Rust Telegram bot, `tokio` async runtime, `reqwest` HTTP client

---

## 1. Current Architecture

### 1.1 The LM Studio Harness

The harness is centred on a single function in `orchestrator.rs`:

```rust
async fn post_to_lm_studio(
    config: &Config,
    messages: &[LmMessage],
    tools: Option<&serde_json::Value>,
    max_tokens: u32,
) -> anyhow::Result<LmCompletionResponse>
```

Every LM call goes through this function. It:

- Creates a **new `reqwest::Client` per call** (wasteful — no connection reuse)
- POSTs to `{LM_STUDIO_URL}/chat/completions` (OpenAI-compatible API)
- Uses `temperature: 0.0`, a 900-second timeout, and `tool_choice: "required"` when tools are supplied
- Returns a deserialized `LmCompletionResponse` containing `choices[0].message`

### 1.2 Call Sites

| Call Site | Location | Concurrency Today |
|-----------|----------|-------------------|
| `try_process` — main user turn | orchestrator.rs:~130 | One call, then sequential tool dispatch |
| `run_agentic_loop` — heartbeat/consolidation | orchestrator.rs:~720 | Sequential loop (up to 12 iterations) |
| `run_memory_eval` — post-response reflection | orchestrator.rs:~820 | Fire-and-forget via `tokio::spawn` |
| `try_execute_scheduled_task` — cron jobs | orchestrator.rs:~90 | Sequential |

### 1.3 Existing Concurrency

The codebase already uses several async patterns worth preserving:

- `tokio::try_join!` for parallel context loading (history, core memory, episodic, recent memory)
- `tokio::spawn` for fire-and-forget background tasks (memory eval, consolidation trigger)
- `OnceLock<Mutex<()>>` in `cli_wrapper.rs` — **global serialization gate** for all `copilot` CLI subprocess calls (prevents corruption of shared CLI session files)
- `Arc<Config>` / `Arc<SchedulerHandle>` for cheap cloning across spawned tasks

---

## 2. The Core Constraint: LM Studio's Inference Concurrency

Before planning any parallelism, we must understand the bottleneck.

LM Studio runs a local model (typically on a single GPU or CPU). By default:

- **One inference slot** — requests beyond the first queue at the HTTP server level
- The server does **not** reject overload; it serializes internally
- With `n_parallel > 1` configured in LM Studio, multiple sequences can be batched, but:
  - GPU memory per sequence goes down (shorter effective context)
  - Per-request latency increases due to attention over larger combined batch
  - Throughput may improve for small requests, but not for 1024–4096 token generations

**Implication:** naively firing many simultaneous requests yields the same wall-clock time as sequencing them, with added overhead. The goal of parallelism is therefore **not** maximum concurrent inference — it is **independent work that does not need to block each other**.

---

## 3. Parallelization Opportunities

Ranked by impact and implementation complexity.

### 3.1 Shared `reqwest::Client` (Quick Win, High Value)

**Current problem:** A new `reqwest::Client` is constructed on every `post_to_lm_studio` call. `reqwest::Client` manages a connection pool internally — creating it fresh every time throws away that pool, forcing a new TCP handshake to LM Studio on each call.

**Fix:** Store a single `reqwest::Client` on `Config` (or as a `OnceLock<reqwest::Client>`).

```rust
// config.rs — add field
pub struct Config {
    // ... existing fields ...
    pub http_client: reqwest::Client,
}

// In Config::from_env():
let http_client = reqwest::Client::builder()
    .timeout(Duration::from_secs(900))
    .tcp_keepalive(Duration::from_secs(60))
    .pool_max_idle_per_host(4)
    .build()
    .expect("failed to build HTTP client");
```

`reqwest::Client` is `Clone` and internally `Arc`-wrapped, so passing it around via `Arc<Config>` is zero-cost.

**Impact:** Eliminates TCP connection setup latency on every LM call. For a 900s timeout and a local server this is small in absolute terms, but it is free correctness and aligns with best practice.

**Risk:** None. The change is purely additive.

---

### 3.2 LM Studio Concurrency Semaphore (Required Foundation)

Before enabling any fan-out, we need a global semaphore that caps simultaneous in-flight LM requests. This prevents queue pile-up at LM Studio during periods when multiple async tasks (heartbeat, memory eval, consolidation) fire close together.

```rust
// In Config or as a module-level OnceLock:
use tokio::sync::Semaphore;

static LM_SEMAPHORE: OnceLock<Semaphore> = OnceLock::new();

fn lm_semaphore() -> &'static Semaphore {
    LM_SEMAPHORE.get_or_init(|| Semaphore::new(LM_MAX_CONCURRENCY))
}
```

Where `LM_MAX_CONCURRENCY` defaults to `1` and is configurable via an env var (`LM_MAX_CONCURRENCY`). Setting it to `1` preserves today's behaviour exactly. Setting it to `2` or `3` only makes sense if LM Studio is configured with `n_parallel` to match.

```rust
async fn post_to_lm_studio(config: &Config, ...) -> anyhow::Result<LmCompletionResponse> {
    let _permit = lm_semaphore().acquire().await?;
    // ... existing HTTP call ...
}
```

The permit is held for the duration of the HTTP request and dropped on return (or on error).

**Trade-off:** With `LM_MAX_CONCURRENCY=1`, `run_memory_eval` (currently fire-and-forget) will wait behind the main turn's LM call before it can execute its own call — this is the correct behaviour. Without a semaphore, memory eval could interrupt an agentic loop iteration.

**Trade-off against raising to 2+:** LM Studio must have `n_parallel ≥ LM_MAX_CONCURRENCY`, otherwise the second request gets queued inside LM Studio anyway (zero benefit, doubled memory cost). Document this coupling clearly in `.env.example`.

---

### 3.3 Parallel Interest Checks in Heartbeat (Medium Complexity, High Value)

**Current flow:**

```
run_heartbeat()
  └─ run_agentic_loop(messages, tools, max_iters=12)
       → LM decides which interests to check
       → sequential delegate_cli calls for each
```

The LM drives all interest research sequentially. For N active interests, each `delegate_cli` call takes 5–60 seconds (copilot subprocess). With 4 interests, heartbeat can take 4–8 minutes of wall-clock time where most of that is the CLI tool, not LM.

**Proposed flow (fan-out per interest):**

Instead of a single monolithic agentic loop, run a **dedicated per-interest check** for each interest concurrently, then gather and synthesize:

```
run_heartbeat()
  ├─ Gather: interests = core_memory.interests (e.g., 4 interests)
  │
  ├─ Fan-out (parallel, bounded by semaphore):
  │   ├─ check_single_interest(config, interest_0) → Option<String>
  │   ├─ check_single_interest(config, interest_1) → Option<String>
  │   ├─ check_single_interest(config, interest_2) → Option<String>
  │   └─ check_single_interest(config, interest_3) → Option<String>
  │
  └─ Synthesize findings into single message (or stay silent)
```

`check_single_interest` is a focused 2–3 iteration agentic loop that:
1. Calls `delegate_cli` to research the interest
2. Calls `reflect` to assess whether it's worth sending
3. Returns `Some(finding)` or `None`

The fan-out uses `futures::future::join_all` or `tokio::task::JoinSet`:

```rust
use tokio::task::JoinSet;

let mut set = JoinSet::new();
for interest in &core.interests {
    let cfg = config.clone();
    let bot = bot.clone();
    let sched = scheduler.clone();
    let interest = interest.clone();
    set.spawn(async move {
        check_single_interest(cfg, bot, sched, interest).await
    });
}

let mut findings: Vec<String> = Vec::new();
while let Some(res) = set.join_next().await {
    if let Ok(Ok(Some(text))) = res {
        findings.push(text);
    }
}
```

**LM Studio concurrency:** Each `check_single_interest` call acquires the semaphore before calling `post_to_lm_studio`. With `LM_MAX_CONCURRENCY=1` (default), they serialize at the semaphore. The parallelism benefit comes from the `delegate_cli` work (subprocess calls) which runs **outside** the semaphore — those are CPU/network-bound, not GPU-bound, and can run fully concurrently even when LM_MAX_CONCURRENCY=1.

**Architecture for `check_single_interest`:**

```rust
async fn check_single_interest(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    interest: Interest,
) -> anyhow::Result<Option<String>> {
    // 1. Build a tight, interest-specific system prompt
    let system_prompt = /* focused prompt for single interest */;
    let messages = vec![/* system + task prompt */];

    // 2. Run a minimal 2-iteration loop: research → reflect → (maybe reply)
    run_agentic_loop(config, bot, scheduler, messages, tools::interest_check_tool_defs(), 3).await
}
```

**Trade-off:** Synthesis step adds one LM call to combine findings. This is acceptable because `run_heartbeat` is fire-and-forget from the user's perspective.

**Trade-off:** If LM_MAX_CONCURRENCY=1, wall-clock time for LM inference steps is unchanged (they still serialize). The gain is that `delegate_cli` calls run in parallel. Measured gain depends on the interest check mix — delegate_cli often takes 10–30s, while LM calls for a short interest check take 2–10s, so the ratio typically favours parallelism.

---

### 3.4 Parallel Context Hydration for `try_process` (Already Done)

`try_process` already uses `tokio::try_join!` to load core memory, history, recent memory, and episodic entries in parallel. No change needed.

However, there is one additional parallelisation that could be added: **pre-building the system prompt concurrently with the history window**. Currently both operations are sequential but both are pure computation (no I/O). Given that `build_system_prompt` is microseconds, this is not worth the added complexity.

---

### 3.5 Memory Eval + Main Turn Fully Independent (Already Done, Needs Semaphore)

`run_memory_eval` is already `tokio::spawn`'d after the main reply is sent. The user does not wait for it. It makes one `post_to_lm_studio` call.

**Current gap:** Without the semaphore from §3.2, memory eval could fire an LM call while the next user message is already in-flight through the main turn, causing two simultaneous requests to LM Studio.

**Fix:** The semaphore from §3.2 fully addresses this. Memory eval queues behind whatever LM call is currently running, and the user never sees the wait.

---

### 3.6 Parallel Scheduled Task Execution (Future Consideration)

Scheduled tasks via `execute_scheduled_task` are independent of each other and of heartbeats. The scheduler already fires them as independent async tasks. With the semaphore in place they automatically serialize at the LM level, which is the correct behaviour.

If in the future there are many scheduled tasks firing simultaneously (e.g., 5 interests on independent cron schedules), the semaphore prevents LM Studio overload while allowing scheduler concurrency at the Rust level. No additional changes are needed.

---

## 4. Architecture Decisions

### Decision 1: Where Does the Semaphore Live?

**Option A:** Inside `post_to_lm_studio` (transparent to callers)  
**Option B:** At call sites (explicit acquire before calling)  
**Option C:** As a field on `Config`

**Chosen: Option A (transparent, inside `post_to_lm_studio`)**

Rationale: Every LM call goes through this function. Placing the acquire there gives centralised, unforgeable back-pressure. Callers don't need to know about it. The semaphore itself lives in a `OnceLock<Semaphore>` (or on a shared state struct) at module level in `orchestrator.rs`.

### Decision 2: Semaphore Capacity Default

Default `LM_MAX_CONCURRENCY=1`. This is conservative and correct for single-slot LM Studio. Users who configure `n_parallel` in LM Studio can raise it.

Rationale for `1` as default:
- Prevents silent queue pile-up on underpowered hardware
- Preserves exact current behaviour
- Upgrade path is a single env var change

### Decision 3: Fan-Out Strategy for Heartbeat

**Option A:** Keep single agentic loop, let LM drive all interest checks sequentially  
**Option B:** Fan out per-interest in Rust, synthesize afterwards  
**Option C:** Fan out per-interest, skip synthesis (send N independent messages if any)

**Chosen: Option B (fan-out + synthesis)**

Rationale:
- Option A is the current approach — works but is slow for N > 2 interests
- Option C produces N separate Telegram messages from one heartbeat, which is poor UX
- Option B keeps user experience clean while exploiting I/O parallelism on CLI calls

The synthesis step costs one extra LM call but is well worth it for cohesion.

### Decision 4: Error Handling in Fan-Out

Each `check_single_interest` task is independently error-contained. A single task failure logs a warning and produces `None` — it does not cancel the other tasks or fail the heartbeat. Partial results are still synthesized and sent.

```rust
while let Some(res) = set.join_next().await {
    match res {
        Ok(Ok(Some(text))) => findings.push(text),
        Ok(Ok(None)) => {},
        Ok(Err(e)) => tracing::warn!("Interest check failed: {e}"),
        Err(e) => tracing::warn!("Interest check task panicked: {e}"),
    }
}
```

### Decision 5: `reqwest::Client` Lifetime

Stored as a field on `Config`. `Config` is already `Arc<Config>`-wrapped everywhere. Client is built once in `Config::from_env()`. `reqwest::Client` is already `Clone + Send + Sync` and reference-counted internally.

---

## 5. Implementation Plan

The changes are ordered by dependency and risk.

### Phase 1 — Foundations (Zero Behaviour Change)

#### 1.1 Shared `reqwest::Client` on `Config`

**Files:** `src/config.rs`, `src/orchestrator.rs`

```rust
// config.rs
pub struct Config {
    // ... existing fields ...
    pub http_client: reqwest::Client,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        // ... existing code ...
        let http_client = reqwest::Client::builder()
            .tcp_keepalive(Duration::from_secs(60))
            .pool_max_idle_per_host(4)
            .build()?;
        Ok(Self { ..., http_client })
    }
}
```

```rust
// orchestrator.rs — post_to_lm_studio
async fn post_to_lm_studio(config: &Config, ...) -> anyhow::Result<LmCompletionResponse> {
    let url = format!("{}/chat/completions", config.lm_studio_url);
    let response = config.http_client          // ← use shared client
        .post(&url)
        .json(&payload)
        .timeout(Duration::from_secs(900))     // per-request timeout still needed
        .send()
        .await
        ...
}
```

Note: The `.timeout(Duration::from_secs(900))` moves from the `Client::builder()` (which sets a global connect+read timeout) to the individual request `.timeout()` call, which applies per-request. This is the correct place for a long inference timeout.

#### 1.2 LM Studio Semaphore

**Files:** `src/orchestrator.rs`, `src/config.rs`, `.env.example`

```rust
// orchestrator.rs — add near top
use std::sync::OnceLock;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use std::sync::Arc as StdArc;

static LM_SEMAPHORE: OnceLock<StdArc<Semaphore>> = OnceLock::new();

fn lm_semaphore(max_concurrent: usize) -> &'static StdArc<Semaphore> {
    LM_SEMAPHORE.get_or_init(|| StdArc::new(Semaphore::new(max_concurrent)))
}

// post_to_lm_studio — wrap with permit acquisition
async fn post_to_lm_studio(config: &Config, ...) -> anyhow::Result<LmCompletionResponse> {
    let _permit = lm_semaphore(config.lm_max_concurrency)
        .clone()
        .acquire_owned()
        .await
        .context("LM semaphore closed (shutting down?)")?;
    // ... existing HTTP call ...
    // permit drops here, releasing slot
}
```

```rust
// config.rs — new field
pub lm_max_concurrency: usize,

// from_env():
let lm_max_concurrency = std::env::var("LM_MAX_CONCURRENCY")
    .unwrap_or_else(|_| "1".into())
    .parse::<usize>()
    .context("LM_MAX_CONCURRENCY must be a positive integer")?;
let lm_max_concurrency = lm_max_concurrency.max(1); // clamp to at least 1
```

```dotenv
# .env.example — add
# Max concurrent LM Studio inference requests.
# Raise only if LM Studio is configured with n_parallel to match.
LM_MAX_CONCURRENCY=1
```

**Verification:** After this phase, the bot should behave identically. Run integration smoke test: send a message, verify response, check that memory eval fires correctly in logs.

---

### Phase 2 — Heartbeat Fan-Out

#### 2.1 Extract `check_single_interest`

**Files:** `src/orchestrator.rs`

Refactor `try_run_interest_check` into a function that:
1. Accepts a single `Interest` struct rather than an ID (avoid extra `core_memory::load()` per task)
2. Returns `Option<String>` — the finding text if worth sending, else `None`
3. Uses a 3-iteration agentic loop (research → reflect → decision)

```rust
async fn check_single_interest(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    interest: crate::core_memory::Interest,
    recent_sends: &[crate::episodic::EpisodicEntry],
    recent_checks: &[crate::episodic::EpisodicEntry],
) -> anyhow::Result<Option<String>> {
    let system_prompt = format!(
        "{core_block}\n\n## Focused Interest Check\n\
         Topic: {topic}\nDescription: {description}\n\n\
         Recent sends (skip if topic was already covered): ...\n\n\
         Research whether there are noteworthy recent developments. \
         Only return a finding if genuinely new. \
         Use delegate_cli for research, then reflect(done=true) to conclude.",
        core_block = core.to_prompt_block(),
        topic = interest.topic,
        description = interest.description,
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), .. },
        LmMessage { role: "user".into(), content: Some(format!("Check for updates: {}", interest.topic)), .. },
    ];

    // 3 iterations: research → reflect → (optional) reply
    run_agentic_loop(config, bot, scheduler, messages, tools::interest_check_tool_definitions(), 3).await
}
```

**New tool set for interest checks** (`interest_check_tool_definitions`): A subset of heartbeat tools containing only `delegate_cli`, `reflect`, and `nothing`. No `reply_to_user` — the finding is returned as a string, not sent directly. This keeps the fan-out tasks from each independently sending messages.

#### 2.2 Fan-Out in `try_run_heartbeat`

**Files:** `src/orchestrator.rs`

```rust
async fn try_run_heartbeat(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
) -> anyhow::Result<()> {
    let (core, all_recent) = tokio::join!(
        crate::core_memory::load(),
        crate::episodic::load_recent(50),
    );
    let core = core?;
    let all_recent = all_recent?;

    // ... existing cutoff/recent_sends/recent_checks filtering ...

    // Fan-out: check each interest in parallel
    let mut set = tokio::task::JoinSet::new();
    for interest in core.interests.clone() {
        let cfg = config.clone();
        let bot = bot.clone();
        let sched = scheduler.clone();
        let sends = recent_sends.clone();
        let checks = recent_checks.clone();
        set.spawn(async move {
            check_single_interest(cfg, bot, sched, interest, &sends, &checks).await
        });
    }

    let mut findings: Vec<String> = Vec::new();
    while let Some(res) = set.join_next().await {
        match res {
            Ok(Ok(Some(text))) => findings.push(text),
            Ok(Ok(None)) => {},
            Ok(Err(e)) => tracing::warn!("Interest check task error: {e}"),
            Err(e) => tracing::warn!("Interest check task panicked: {e}"),
        }
    }

    // Synthesize findings → send one message if anything is worth sharing
    if !findings.is_empty() {
        let combined = findings.join("\n\n---\n\n");
        let synthesis = synthesize_heartbeat_findings(&config, &core, &combined).await?;
        if let Some(msg) = synthesis {
            bot.send_message(ChatId(config.allowed_user_id as i64), msg.clone()).await?;
            // ... log heartbeat-sent episodic entry as before ...
        }
    }

    // ... log heartbeat-checked episodic entry as before ...
    Ok(())
}
```

#### 2.3 `synthesize_heartbeat_findings`

A simple one-shot LM call (no tools) that combines the raw findings into a single coherent message:

```rust
async fn synthesize_heartbeat_findings(
    config: &Config,
    core: &CoreMemory,
    raw_findings: &str,
) -> anyhow::Result<Option<String>> {
    let system_prompt = format!(
        "{core_block}\n\nYou have gathered findings from multiple interest checks. \
         Write a single cohesive update to share with the user. \
         If no finding is genuinely worth sending, output only the word SILENT.",
        core_block = core.to_prompt_block(),
    );
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), .. },
        LmMessage { role: "user".into(), content: Some(raw_findings.to_string()), .. },
    ];
    let response = post_to_lm_studio(config, &messages, None, 1024).await?;
    let text = response.choices.into_iter().next()
        .and_then(|c| c.message.content)
        .map(|t| strip_think_blocks(&t).trim().to_string())
        .unwrap_or_default();
    if text.is_empty() || text.eq_ignore_ascii_case("SILENT") {
        return Ok(None);
    }
    Ok(Some(text))
}
```

---

### Phase 3 — Metrics & Observability (Recommended)

Add structured timing logs to `post_to_lm_studio` so parallelism impact is measurable:

```rust
async fn post_to_lm_studio(config: &Config, ...) -> anyhow::Result<LmCompletionResponse> {
    let wait_start = std::time::Instant::now();
    let _permit = lm_semaphore(config.lm_max_concurrency).clone().acquire_owned().await?;
    let wait_elapsed = wait_start.elapsed();

    if wait_elapsed > Duration::from_millis(100) {
        tracing::debug!("LM semaphore wait: {:.1}s", wait_elapsed.as_secs_f32());
    }

    let request_start = std::time::Instant::now();
    // ... HTTP call ...
    let request_elapsed = request_start.elapsed();

    tracing::info!(
        lm_wait_ms = wait_elapsed.as_millis(),
        lm_request_ms = request_elapsed.as_millis(),
        "LM Studio call completed"
    );
    Ok(response)
}
```

This surfaces queueing pressure at the semaphore, making it easy to decide when to raise `LM_MAX_CONCURRENCY`.

---

## 6. Trade-Off Summary

| Change | Gain | Cost | Risk |
|--------|------|------|------|
| Shared `reqwest::Client` | Eliminates TCP handshake per LM call | Minor Config change | None |
| LM Semaphore | Back-pressure, prevents LM Studio overload | Tiny overhead per call | Memory eval queues behind main turn (correct behaviour) |
| Heartbeat fan-out | CLI delegate calls run in parallel — heartbeat latency drops proportionally to N interests | Synthesis LM call added; more complex code | If synthesis fails, findings are dropped (add fallback: send raw) |
| Raising LM_MAX_CONCURRENCY | True concurrent inference | Higher VRAM usage, longer per-call latency at LM Studio | Only safe if LM Studio n_parallel matches |

### When NOT to Parallelize

- **Inside `run_agentic_loop`**: Each iteration depends on the prior tool result. Parallelism is structurally impossible here. Do not change this.
- **CLI tool calls across turns**: The `OnceLock<Mutex<()>>` in `cli_wrapper.rs` must remain. Removing it would corrupt Copilot's shared session file, causing garbled tool results or authentication failures.
- **`try_process` main turn**: The user is waiting. Adding parallelism here adds complexity with no perceivable latency gain (the LM call dominates).

---

## 7. Configuration Reference

After all phases, add to `.env.example`:

```dotenv
# ── LM Studio parallelisation ─────────────────────────────────────────────
# Maximum number of simultaneous in-flight requests to LM Studio.
# 1 = default (safe for any hardware, exact current behaviour).
# 2+ = only raise if LM Studio is configured with n_parallel=N in its
#       model settings. Mismatches waste memory without improving latency.
LM_MAX_CONCURRENCY=1
```

---

## 8. File Change Summary

| File | Change |
|------|--------|
| `src/config.rs` | Add `http_client: reqwest::Client` field and `lm_max_concurrency: usize` field |
| `src/orchestrator.rs` | Use `config.http_client` in `post_to_lm_studio`; add semaphore acquire; add `check_single_interest`; add `synthesize_heartbeat_findings`; refactor `try_run_heartbeat` to use `JoinSet` fan-out |
| `.env.example` | Document `LM_MAX_CONCURRENCY` |
| `Cargo.toml` | No new dependencies — `tokio::task::JoinSet` is in `tokio` 1.x with `full` feature; `OnceLock` is in `std` since Rust 1.70 |

---

## 9. Testing Strategy

1. **Unit:** `check_single_interest` can be tested with a mock `post_to_lm_studio` (extract to trait or pass as closure). Verify that `None` is returned when LM returns `nothing`, and `Some(text)` when `reply_to_user` fires.

2. **Integration smoke test (manual):** Configure 2–3 test interests. Trigger heartbeat manually (via a scheduled task or a direct call). Verify:
   - Logs show multiple concurrent interest check tasks
   - Semaphore wait is logged if `LM_MAX_CONCURRENCY=1`
   - One synthesized message arrives (not N separate messages)
   - `heartbeat-sent` and `heartbeat-checked` episodic entries are written correctly

3. **Regression:** Verify existing `try_process` flow is unchanged — main turn responses, tool calls, memory eval, and consolidation all continue to work.

4. **Load test (optional):** Send 10 messages rapidly. With `LM_MAX_CONCURRENCY=1`, verify logs show semaphore wait times rather than LM Studio errors or timeouts.

---

*End of plan.*
