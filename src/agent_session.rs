use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use teloxide::prelude::*;

use crate::config::Config;
use crate::lm::{LmAssistantMessage, LmMessage, post_to_lm_studio, strip_think_blocks};
use crate::memory::ConversationMessage;
use crate::scheduler::SchedulerHandle;
use crate::tools;
use crate::tools::LmToolCall;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PARALLEL_SAFE_TOOLS: &[&str] = &[
    "fetch_url",
    "recall",
    "list_interests",
    "list_agenda_items",
    "list_schedules",
];

const TOOL_CALL_WARN_THRESHOLD: usize = 8;
const TOOL_CALL_HARD_LIMIT: usize = 15;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Observable state of a running session — updated each iteration.
#[derive(Debug, Clone)]
pub enum SessionState {
    /// About to call the LLM.
    Thinking,
    /// Dispatching one or more tool calls.
    ToolExecuting,
    /// Just completed a `reflect(done=false)` observation.
    Reflecting,
    /// Session finished — carries the final reply if any was produced.
    Done(Option<String>),
}

#[allow(dead_code)]
impl SessionState {
    /// Returns the final reply text when the session is `Done`, or `None` otherwise.
    pub fn final_reply(&self) -> Option<&str> {
        match self {
            SessionState::Done(reply) => reply.as_deref(),
            _ => None,
        }
    }
}

pub struct PolicySet {
    pub seen_fingerprints: HashSet<(String, String)>,
    pub seen_reflections: HashSet<String>,
    pub tool_call_counts: HashMap<String, usize>,
    /// Warning message to inject after the current iteration (if any).
    pub tool_call_warning: Option<String>,
    pub anti_stall_injected: bool,
    pub budget_warning_injected: bool,
    pub context_compressed: bool,
}

impl PolicySet {
    fn new() -> Self {
        Self {
            seen_fingerprints: HashSet::new(),
            seen_reflections: HashSet::new(),
            tool_call_counts: HashMap::new(),
            tool_call_warning: None,
            anti_stall_injected: false,
            budget_warning_injected: false,
            context_compressed: false,
        }
    }
}

/// Signal returned by each step of the session.
pub struct StepAction {
    pub should_exit: bool,
    pub final_reply: Option<String>,
}

/// Owns all mutable state for one agentic run (user turn, heartbeat, etc.).
pub struct AgentSession {
    pub config: Arc<Config>,
    /// Swapped to `background_copilot_model` when configured, for CLI calls.
    pub call_config: Arc<Config>,
    pub bot: Bot,
    pub scheduler: Arc<SchedulerHandle>,
    /// Full LM context window (grows each iteration).
    pub messages: Vec<LmMessage>,
    /// New messages produced during this run (returned to caller for persistence).
    pub new_messages: Vec<ConversationMessage>,
    /// When true, the session sends Telegram messages directly (background/heartbeat paths).
    pub send_reply: bool,
    /// Current execution state — updated each iteration for observability.
    pub state: SessionState,
    pub policies: PolicySet,
    /// Stored for budget-warning math inside `apply_pre_step_policies`.
    max_iters: usize,
}

// ---------------------------------------------------------------------------
// AgentSession implementation
// ---------------------------------------------------------------------------

impl AgentSession {
    pub fn new(
        config: Arc<Config>,
        bot: Bot,
        scheduler: Arc<SchedulerHandle>,
        initial_messages: Vec<LmMessage>,
        send_reply: bool,
        max_iters: usize,
    ) -> Self {
        let call_config = if config.background_copilot_model.is_some() {
            Arc::new(Config {
                copilot_model: config.background_copilot_model.clone(),
                ..(*config).clone()
            })
        } else {
            config.clone()
        };

        Self {
            config,
            call_config,
            bot,
            scheduler,
            messages: initial_messages,
            new_messages: Vec::new(),
            send_reply,
            state: SessionState::Thinking,
            policies: PolicySet::new(),
            max_iters,
        }
    }

    /// Run the agentic loop, returning `(reply_text, new_messages)`.
    pub async fn run(
        config: Arc<Config>,
        bot: Bot,
        scheduler: Arc<SchedulerHandle>,
        initial_messages: Vec<LmMessage>,
        tool_defs: serde_json::Value,
        max_iters: usize,
        send_reply: bool,
    ) -> anyhow::Result<(Option<String>, Vec<ConversationMessage>)> {
        let mut session = Self::new(config, bot, scheduler, initial_messages, send_reply, max_iters);

        for iter in 0..max_iters {
            let action = session.step(&tool_defs, iter).await?;

            // Inject the per-tool warning (if set this iteration) before continuing.
            // Skipped on exit so the model doesn't see it after the session ends.
            if !action.should_exit {
                if let Some(warn) = session.policies.tool_call_warning.take() {
                    session.push_user_message(&warn);
                }
                session.state = SessionState::Thinking;
            }

            if action.should_exit {
                session.state = SessionState::Done(action.final_reply.clone());
                return Ok((action.final_reply, session.new_messages));
            }
        }

        session.state = SessionState::Done(None);
        Ok((None, session.new_messages))
    }

    /// Execute one full iteration: pre-step policies → LLM call → response dispatch.
    ///
    /// Separating this from `run` makes individual iterations unit-testable and
    /// keeps the outer loop a clean 10-line driver.
    pub async fn step(
        &mut self,
        tool_defs: &serde_json::Value,
        iter: usize,
    ) -> anyhow::Result<StepAction> {
        self.apply_pre_step_policies(iter).await?;

        self.state = SessionState::Thinking;
        tracing::info!("Agentic loop iter {}: awaiting LM...", iter + 1);
        let lm_start = std::time::Instant::now();
        let response =
            post_to_lm_studio(&self.config, &self.messages, Some(tool_defs), 4096).await?;
        let lm_elapsed = lm_start.elapsed();
        let choice = response
            .choices
            .into_iter()
            .next()
            .context("LM Studio returned no choices")?;

        if choice.finish_reason == "length" {
            anyhow::bail!("Agentic loop response truncated (finish_reason=length)");
        }

        if !choice.message.tool_calls.is_empty() {
            self.state = SessionState::ToolExecuting;
        }

        self.handle_model_response(choice.message, iter, lm_elapsed).await
    }

    // ── Pre-step policies ────────────────────────────────────────────────────

    async fn apply_pre_step_policies(&mut self, iter: usize) -> anyhow::Result<()> {
        let max_iters = self.max_iters;
        // 2.5 — Compress the middle of the context window once, when it grows large.
        if !self.policies.context_compressed
            && self.messages.len() > self.config.context_compress_threshold
        {
            // Need at least: 1 system + 2 middle + 4 tail = 7 messages to be worth splitting.
            if self.messages.len() > 7 {
                let tail = self.messages.split_off(self.messages.len() - 4);
                let mid = self.messages.split_off(1); // messages = [system_prompt]

                let summary_lines: String = mid
                    .iter()
                    .filter_map(|m| {
                        m.content.as_deref().filter(|c| !c.is_empty()).map(|c| {
                            let snippet = if c.len() > 400 { &c[..400] } else { c };
                            format!("[{}] {}", m.role, snippet)
                        })
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");

                let summary_prompt = vec![
                    LmMessage {
                        role: "system".into(),
                        content: Some("You are a transcript summarizer. Be concise.".into()),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    LmMessage {
                        role: "user".into(),
                        content: Some(format!(
                            "Summarize the following exchange in 3 concise bullet points. \
                             Capture: what was attempted, what was found, and what decisions were made.\n\n{}",
                            summary_lines
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ];

                // Use background_lm_model when configured; fall back to the main model.
                let summary_model = self
                    .config
                    .background_lm_model
                    .as_deref()
                    .unwrap_or(&self.config.model_name)
                    .to_string();
                let summary_config = Arc::new(Config {
                    model_name: summary_model,
                    ..(*self.config).clone()
                });

                let n_compressed = mid.len();
                match post_to_lm_studio(&summary_config, &summary_prompt, None, 512).await {
                    Ok(resp) => {
                        let raw = resp
                            .choices
                            .into_iter()
                            .next()
                            .and_then(|c| c.message.content)
                            .unwrap_or_default();
                        let bullets = strip_think_blocks(&raw).trim().to_string();
                        if !bullets.is_empty() {
                            self.messages.push(LmMessage {
                                role: "user".into(),
                                content: Some(format!(
                                    "[Context Summary — {} messages compressed]\n{}",
                                    n_compressed, bullets
                                )),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                            self.policies.context_compressed = true;
                            tracing::info!(
                                "Agentic loop iter {}: context compressed ({} messages → 1 summary)",
                                iter + 1,
                                n_compressed
                            );
                        } else {
                            tracing::warn!(
                                "Context compression produced empty summary — restoring middle"
                            );
                            self.messages.extend(mid);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Context compression LM call failed: {} — restoring middle",
                            e
                        );
                        self.messages.extend(mid);
                    }
                }
                // Always restore the tail.
                self.messages.extend(tail);
            }
        }

        // 1.5C — Inject a budget warning once when 80% of iterations are consumed.
        if !self.policies.budget_warning_injected && max_iters > 4 && iter == (max_iters * 4) / 5 {
            let remaining = max_iters - iter;
            self.messages.push(LmMessage {
                role: "user".into(),
                content: Some(format!(
                    "[System: {} iteration{} remaining in this run. \
                     Wrap up your current work. Queue any unfinished tasks with \
                     add_agenda_item if available, then conclude with reflect(done=true).]",
                    remaining,
                    if remaining == 1 { "" } else { "s" },
                )),
                tool_calls: None,
                tool_call_id: None,
            });
            self.policies.budget_warning_injected = true;
            tracing::info!(
                "Agentic loop iter {}: budget warning injected ({} iters remain)",
                iter + 1,
                remaining
            );
        }

        Ok(())
    }

    // ── Model response router ────────────────────────────────────────────────

    async fn handle_model_response(
        &mut self,
        msg: LmAssistantMessage,
        iter: usize,
        lm_elapsed: std::time::Duration,
    ) -> anyhow::Result<StepAction> {
        if msg.tool_calls.is_empty() {
            // Plain text reply — send if non-empty.
            let raw = msg.content.unwrap_or_default();
            let text = strip_think_blocks(&raw).trim().to_string();
            if !text.is_empty() {
                if self.send_reply {
                    self.bot
                        .send_message(ChatId(self.config.allowed_user_id as i64), text.clone())
                        .await?;
                }
                self.new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: Utc::now(),
                });
                return Ok(StepAction { should_exit: true, final_reply: Some(text) });
            }
            return Ok(StepAction { should_exit: true, final_reply: None });
        }

        let all_tool_calls = msg.tool_calls;

        if all_tool_calls.len() == 1 {
            self.handle_single_tool_call(
                all_tool_calls.into_iter().next().expect("checked non-empty"),
                iter,
                lm_elapsed,
            )
            .await
        } else {
            self.handle_multi_tool_calls(all_tool_calls, iter, lm_elapsed).await
        }
    }

    // ── Single tool call path ────────────────────────────────────────────────
    //
    // Anti-stall and reflect dedup live here exclusively; they don't apply
    // to multi-call responses.

    async fn handle_single_tool_call(
        &mut self,
        call: LmToolCall,
        iter: usize,
        lm_elapsed: std::time::Duration,
    ) -> anyhow::Result<StepAction> {
        let single_call_json = serde_json::to_value(std::slice::from_ref(&call)).ok();
        let tool_name = call.function.name.clone();
        let call_id = call.id.clone();
        let raw_args: serde_json::Value =
            serde_json::from_str(&call.function.arguments).unwrap_or_default();

        // 1.5B — Anti-stall: any repeated (tool, args) fingerprint within the same turn
        // triggers the warning once. Uses a HashSet so alternating A-B-A-B loops are caught
        // just as reliably as consecutive A-A repeats.
        let current_fingerprint = (tool_name.clone(), call.function.arguments.clone());
        if !self.policies.anti_stall_injected
            && !self.policies.seen_fingerprints.insert(current_fingerprint)
        {
            tracing::warn!(
                "Agentic loop iter {}: anti-stall triggered (repeated {})",
                iter + 1,
                tool_name
            );
            let result = "[Already completed — results are already in your context above. \
                           Do not repeat this call. Review the results, then call \
                           reply_to_user with your answer or reflect(done=true) to wrap up.]"
                .to_string();
            let now = Utc::now();
            self.messages.push(LmMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: single_call_json.clone(),
                tool_call_id: None,
            });
            self.new_messages.push(ConversationMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: single_call_json,
                tool_call_id: None,
                timestamp: now,
            });
            self.messages.push(LmMessage {
                role: "tool".into(),
                content: Some(result.clone()),
                tool_calls: None,
                tool_call_id: Some(call_id.clone()),
            });
            self.new_messages.push(ConversationMessage {
                role: "tool".into(),
                content: Some(result),
                tool_calls: None,
                tool_call_id: Some(call_id),
                timestamp: now,
            });
            self.messages.push(LmMessage {
                role: "user".into(),
                content: Some(format!(
                    "[System: You already ran `{}` with these exact arguments earlier this turn. \
                     Stop repeating tool calls. Review results in your context above, \
                     then call reply_to_user with your answer or reflect(done=true) to wrap up.]",
                    tool_name
                )),
                tool_calls: None,
                tool_call_id: None,
            });
            self.policies.anti_stall_injected = true;
            return Ok(StepAction { should_exit: false, final_reply: None });
        }

        // Exact-duplicate backstop: if the model makes the identical reflect call twice, exit.
        if tool_name == "reflect" {
            if !self.policies.seen_reflections.insert(call.function.arguments.clone()) {
                tracing::warn!(
                    "Agentic loop early exit: duplicate reflect call at iter {}",
                    iter + 1
                );
                return Ok(StepAction { should_exit: true, final_reply: None });
            }
        }

        // 1.5D — Per-tool call cap: warn at WARN threshold, hard-stop at HARD limit.
        // Terminal tools are excluded — the loop must be able to conclude.
        let is_terminal = matches!(
            tool_name.as_str(),
            "reply_to_user" | "nothing" | "reflect" | "send_update" | "final_reply"
        );
        if !is_terminal {
            let count = {
                let c = self.policies.tool_call_counts.entry(tool_name.clone()).or_insert(0);
                *c += 1;
                *c
            };
            if count == TOOL_CALL_WARN_THRESHOLD {
                self.policies.tool_call_warning = Some(format!(
                    "[System: You have called `{}` {} times this turn. \
                     This is unusually many — stop calling this tool unless \
                     absolutely necessary and wrap up your work.]",
                    tool_name, count
                ));
            }
            if count >= TOOL_CALL_HARD_LIMIT {
                tracing::warn!(
                    "Agentic loop iter {}: hard-stopped `{}` after {} calls this turn",
                    iter + 1,
                    tool_name,
                    count
                );
                let stop_result = format!(
                    "❌ Tool call suppressed: `{}` has been called {} times this turn, \
                     exceeding the per-turn limit of {}. \
                     Stop calling this tool and wrap up.",
                    tool_name, count, TOOL_CALL_HARD_LIMIT
                );
                let now = Utc::now();
                self.messages.push(LmMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json.clone(),
                    tool_call_id: None,
                });
                self.new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json,
                    tool_call_id: None,
                    timestamp: now,
                });
                self.messages.push(LmMessage {
                    role: "tool".into(),
                    content: Some(stop_result.clone()),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                });
                self.new_messages.push(ConversationMessage {
                    role: "tool".into(),
                    content: Some(stop_result),
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    timestamp: now,
                });
                return Ok(StepAction { should_exit: false, final_reply: None });
            }
        }

        tracing::info!(
            "Agentic loop iter {}: tool={} [LM: {:.1}s]",
            iter + 1,
            tool_name,
            lm_elapsed.as_secs_f32()
        );
        let dispatch_start = std::time::Instant::now();
        let result = tools::dispatch_tool_call(
            self.call_config.clone(),
            self.bot.clone(),
            self.scheduler.clone(),
            call,
        )
        .await?;
        let dispatch_elapsed = dispatch_start.elapsed();
        tracing::info!(
            "Agentic loop iter {}: tool={} done [tool: {:.1}s]",
            iter + 1,
            tool_name,
            dispatch_elapsed.as_secs_f32()
        );

        match tool_name.as_str() {
            "send_update" => {
                self.bot
                    .send_message(ChatId(self.config.allowed_user_id as i64), result.clone())
                    .await?;
                let now = Utc::now();
                self.messages.push(LmMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json.clone(),
                    tool_call_id: None,
                });
                self.new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json,
                    tool_call_id: None,
                    timestamp: now,
                });
                self.messages.push(LmMessage {
                    role: "tool".into(),
                    content: Some(result.clone()),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                });
                self.new_messages.push(ConversationMessage {
                    role: "tool".into(),
                    content: Some(result),
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    timestamp: now,
                });
                self.push_user_message("[System: Update sent. Continue working.]");
                tracing::info!(
                    "Agentic loop iter {}: send_update — update sent, continuing",
                    iter + 1
                );
                Ok(StepAction { should_exit: false, final_reply: None })
            }

            "final_reply" => {
                if self.send_reply {
                    self.bot
                        .send_message(ChatId(self.config.allowed_user_id as i64), result.clone())
                        .await?;
                }
                self.new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(result.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: Utc::now(),
                });
                Ok(StepAction { should_exit: true, final_reply: Some(result) })
            }

            "reply_to_user" => {
                let background = raw_args["background"].as_bool().unwrap_or(false);
                if self.send_reply || background {
                    self.bot
                        .send_message(ChatId(self.config.allowed_user_id as i64), result.clone())
                        .await?;
                }
                // Always persist the reply as an assistant message.
                self.new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(result.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: Utc::now(),
                });
                if !background {
                    return Ok(StepAction { should_exit: true, final_reply: Some(result) });
                }
                // background=true: push the tool-call pair into LM context so the
                // transcript stays well-formed, then inject a continuation hint.
                self.messages.push(LmMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json,
                    tool_call_id: None,
                });
                self.messages.push(LmMessage {
                    role: "tool".into(),
                    content: Some(result),
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                });
                self.push_user_message("[System: Reply sent. Continue working.]");
                tracing::info!(
                    "Agentic loop iter {}: reply_to_user(background=true) — reply sent, continuing",
                    iter + 1
                );
                Ok(StepAction { should_exit: false, final_reply: None })
            }

            "nothing" => Ok(StepAction { should_exit: true, final_reply: None }),

            "reflect" => {
                if raw_args["done"].as_bool().unwrap_or(false) {
                    tracing::info!("Agentic loop: reflect concluded at iter {}", iter + 1);
                    return Ok(StepAction { should_exit: true, final_reply: None });
                }
                // done=false: observation goes into context, loop continues.
                self.state = SessionState::Reflecting;
                tracing::debug!("Agentic loop iter {} result: {:.500}", iter + 1, result);
                self.push_assistant_call(single_call_json);
                self.push_tool_result(call_id, result);
                Ok(StepAction { should_exit: false, final_reply: None })
            }

            _ => {
                tracing::debug!("Agentic loop iter {} result: {:.500}", iter + 1, result);
                self.push_assistant_call(single_call_json);
                self.push_tool_result(call_id, result);
                Ok(StepAction { should_exit: false, final_reply: None })
            }
        }
    }

    // ── Multi tool call path ─────────────────────────────────────────────────
    //
    // All PARALLEL_SAFE_TOOLS are dispatched concurrently via JoinSet.
    // Any batch that contains a write tool, a terminal tool, or a mix
    // of safe and unsafe tools is dispatched sequentially instead.
    // Terminal tools (reply_to_user, nothing, reflect/done, final_reply) are
    // always executed last — other tools are drained first.

    async fn handle_multi_tool_calls(
        &mut self,
        all_tool_calls: Vec<LmToolCall>,
        iter: usize,
        lm_elapsed: std::time::Duration,
    ) -> anyhow::Result<StepAction> {
        let all_calls_json = serde_json::to_value(&all_tool_calls).ok();

        // Extract send_update calls first so they are dispatched and sent to the
        // user before parallel work begins — progress is visible without extra wait.
        let (immediate_calls, all_tool_calls): (Vec<_>, Vec<_>) =
            all_tool_calls.into_iter().partition(|c| c.function.name == "send_update");

        let all_parallel_safe = all_tool_calls
            .iter()
            .all(|c| PARALLEL_SAFE_TOOLS.contains(&c.function.name.as_str()));

        // Separate terminal calls from non-terminal to ensure non-terminal
        // tools run first (they may produce data that informs the reply).
        let (terminal_calls, mut other_calls): (Vec<_>, Vec<_>) =
            all_tool_calls.into_iter().partition(|c| {
                matches!(
                    c.function.name.as_str(),
                    "reply_to_user" | "nothing" | "reflect" | "final_reply"
                )
            });

        // Suppressed items: (call_id, tool_name, reason_msg) — get synthetic results below.
        let mut pre_suppressed: Vec<(String, String, String)> = Vec::new();

        // --- Intra-batch dedup + per-tool call cap for non-terminal tools ---
        {
            let mut batch_seen: HashSet<(String, String)> = HashSet::new();
            let mut filtered: Vec<_> = Vec::with_capacity(other_calls.len());
            for call in other_calls {
                let key = (call.function.name.clone(), call.function.arguments.clone());
                if !batch_seen.insert(key) {
                    pre_suppressed.push((
                        call.id,
                        call.function.name,
                        "⚠️ Duplicate call suppressed — this exact call already appeared \
                         in this batch."
                            .to_string(),
                    ));
                    continue;
                }
                let count = {
                    let c =
                        self.policies.tool_call_counts.entry(call.function.name.clone()).or_insert(0);
                    *c += 1;
                    *c
                };
                if count == TOOL_CALL_WARN_THRESHOLD {
                    self.policies.tool_call_warning = Some(format!(
                        "[System: You have called `{}` {} times this turn. \
                         This is unusually many — stop calling this tool unless \
                         absolutely necessary and wrap up your work.]",
                        call.function.name, count
                    ));
                }
                if count >= TOOL_CALL_HARD_LIMIT {
                    tracing::warn!(
                        "Agentic loop iter {}: hard-stopped `{}` after {} calls (multi-call)",
                        iter + 1,
                        call.function.name,
                        count
                    );
                    pre_suppressed.push((
                        call.id,
                        call.function.name.clone(),
                        format!(
                            "❌ Tool call suppressed: `{}` has been called {} times this \
                             turn, exceeding the per-turn limit of {}. \
                             Stop calling this tool and wrap up.",
                            call.function.name, count, TOOL_CALL_HARD_LIMIT
                        ),
                    ));
                    continue;
                }
                filtered.push(call);
            }
            other_calls = filtered;
        }

        // Dispatch type alias: (original_index, call_id, tool_name, raw_args, result)
        type DispatchItem = (usize, String, String, serde_json::Value, anyhow::Result<String>);

        // --- Dispatch send_update calls immediately (before parallel work) ---
        let mut results: Vec<DispatchItem> = {
            let mut v: Vec<DispatchItem> = Vec::new();
            for (i, call) in immediate_calls.into_iter().enumerate() {
                let raw_args: serde_json::Value =
                    serde_json::from_str(&call.function.arguments).unwrap_or_default();
                let call_id = call.id.clone();
                let tool_name = call.function.name.clone();
                let r = tools::dispatch_tool_call(
                    self.call_config.clone(),
                    self.bot.clone(),
                    self.scheduler.clone(),
                    call,
                )
                .await;
                if let Ok(ref msg) = r {
                    let _ = self
                        .bot
                        .send_message(ChatId(self.config.allowed_user_id as i64), msg.clone())
                        .await;
                    tracing::info!(
                        "Agentic loop iter {}: send_update (multi) — update sent",
                        iter + 1
                    );
                }
                v.push((i, call_id, tool_name, raw_args, r));
            }
            v
        };
        let base_immediate = results.len();

        // --- Dispatch non-terminal tools ---
        let other_results: Vec<DispatchItem> =
            if all_parallel_safe && terminal_calls.is_empty() {
                // All calls are safe to run in parallel.
                let mut join_set: tokio::task::JoinSet<DispatchItem> =
                    tokio::task::JoinSet::new();
                for (idx, call) in other_calls.into_iter().enumerate() {
                    let raw_args: serde_json::Value =
                        serde_json::from_str(&call.function.arguments).unwrap_or_default();
                    let call_id = call.id.clone();
                    let tool_name = call.function.name.clone();
                    let (cc, b, s) =
                        (self.call_config.clone(), self.bot.clone(), self.scheduler.clone());
                    join_set.spawn(async move {
                        let r = tools::dispatch_tool_call(cc, b, s, call).await;
                        (idx, call_id, tool_name, raw_args, r)
                    });
                }
                let mut raw: Vec<DispatchItem> = Vec::with_capacity(join_set.len());
                while let Some(joined) = join_set.join_next().await {
                    raw.push(joined.context("parallel tool task panicked")?);
                }
                raw.sort_by_key(|(idx, ..)| *idx);
                raw
            } else {
                // Sequential dispatch for safety (write tools, mixed batches).
                let mut raw: Vec<DispatchItem> = Vec::new();
                for (idx, call) in other_calls.into_iter().enumerate() {
                    let raw_args: serde_json::Value =
                        serde_json::from_str(&call.function.arguments).unwrap_or_default();
                    let call_id = call.id.clone();
                    let tool_name = call.function.name.clone();
                    let r = tools::dispatch_tool_call(
                        self.call_config.clone(),
                        self.bot.clone(),
                        self.scheduler.clone(),
                        call,
                    )
                    .await;
                    raw.push((idx, call_id, tool_name, raw_args, r));
                }
                raw
            };
        for (i, item) in other_results.into_iter().enumerate() {
            let (_, call_id, tool_name, raw_args, r) = item;
            results.push((base_immediate + i, call_id, tool_name, raw_args, r));
        }

        // --- Append terminal tools (sequential, after all others) ---
        let base_idx = results.len();
        for (i, call) in terminal_calls.into_iter().enumerate() {
            let raw_args: serde_json::Value =
                serde_json::from_str(&call.function.arguments).unwrap_or_default();
            let call_id = call.id.clone();
            let tool_name = call.function.name.clone();
            let r = tools::dispatch_tool_call(
                self.call_config.clone(),
                self.bot.clone(),
                self.scheduler.clone(),
                call,
            )
            .await;
            results.push((base_idx + i, call_id, tool_name, raw_args, r));
        }

        // Append synthetic results for deduplicated/hard-stopped calls.
        // All original call IDs must appear in the results to keep the
        // assistant→tool message pairs well-formed.
        let base_suppressed = results.len();
        for (i, (call_id, tool_name, reason)) in pre_suppressed.into_iter().enumerate() {
            results.push((
                base_suppressed + i,
                call_id,
                tool_name,
                serde_json::Value::Null,
                Ok(reason),
            ));
        }

        tracing::info!(
            "Agentic loop iter {}: multi-call ({} tools dispatched) [LM: {:.1}s]",
            iter + 1,
            results.len(),
            lm_elapsed.as_secs_f32()
        );

        // Push one assistant message containing the full tool_calls array.
        let now = Utc::now();
        self.messages.push(LmMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: all_calls_json.clone(),
            tool_call_id: None,
        });
        self.new_messages.push(ConversationMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: all_calls_json,
            tool_call_id: None,
            timestamp: now,
        });

        // Process results; track whether a terminal action was triggered.
        // reply_result carries (message_text, is_background).
        let mut reply_result: Option<(String, bool)> = None;
        let mut return_none = false;

        for (_idx, call_id, tool_name, raw_args, result) in results {
            let result_str = result.unwrap_or_else(|e| format!("❌ {}", e));
            match tool_name.as_str() {
                "reply_to_user" => {
                    let bg = raw_args["background"].as_bool().unwrap_or(false);
                    reply_result = Some((result_str.clone(), bg));
                    // Always push the tool result so every tool call_id in the
                    // assistant message has a matching tool result in history.
                    self.messages.push(LmMessage {
                        role: "tool".into(),
                        content: Some(result_str.clone()),
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                    });
                    self.new_messages.push(ConversationMessage {
                        role: "tool".into(),
                        content: Some(result_str),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                        timestamp: now,
                    });
                }
                "final_reply" => {
                    // final_reply in multi-call: always non-background
                    reply_result = Some((result_str.clone(), false));
                    self.messages.push(LmMessage {
                        role: "tool".into(),
                        content: Some(result_str.clone()),
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                    });
                    self.new_messages.push(ConversationMessage {
                        role: "tool".into(),
                        content: Some(result_str),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                        timestamp: now,
                    });
                }
                "nothing" => {
                    return_none = true;
                    // Push empty tool result to keep the assistant → tool pair
                    // well-formed in persisted history.
                    self.messages.push(LmMessage {
                        role: "tool".into(),
                        content: Some(String::new()),
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                    });
                    self.new_messages.push(ConversationMessage {
                        role: "tool".into(),
                        content: Some(String::new()),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                        timestamp: now,
                    });
                }
                "reflect" => {
                    if raw_args["done"].as_bool().unwrap_or(false) {
                        return_none = true;
                    }
                    self.messages.push(LmMessage {
                        role: "tool".into(),
                        content: Some(result_str.clone()),
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                    });
                    self.new_messages.push(ConversationMessage {
                        role: "tool".into(),
                        content: Some(result_str),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                        timestamp: now,
                    });
                }
                _ => {
                    self.messages.push(LmMessage {
                        role: "tool".into(),
                        content: Some(result_str.clone()),
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                    });
                    self.new_messages.push(ConversationMessage {
                        role: "tool".into(),
                        content: Some(result_str),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                        timestamp: now,
                    });
                }
            }
        }

        // Handle terminal outcome.
        if let Some((reply, background)) = reply_result {
            if self.send_reply || background {
                self.bot
                    .send_message(ChatId(self.config.allowed_user_id as i64), reply.clone())
                    .await?;
            }
            self.new_messages.push(ConversationMessage {
                role: "assistant".into(),
                content: Some(reply.clone()),
                tool_calls: None,
                tool_call_id: None,
                timestamp: now,
            });
            if !background {
                return Ok(StepAction { should_exit: true, final_reply: Some(reply) });
            }
            // background=true: tool result already pushed unconditionally in the match arm.
            // Inject the continuation hint and fall through so the loop iterates.
            self.push_user_message("[System: Reply sent. Continue working.]");
            tracing::info!(
                "Agentic loop iter {}: multi-call reply_to_user(background=true) — reply sent, continuing",
                iter + 1
            );
            // No return — loop continues naturally.
        }
        if return_none {
            return Ok(StepAction { should_exit: true, final_reply: None });
        }
        // No terminal tool → loop continues with updated context.
        Ok(StepAction { should_exit: false, final_reply: None })
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    pub fn push_user_message(&mut self, content: &str) {
        self.messages.push(LmMessage {
            role: "user".into(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Push an assistant message carrying a tool-calls array to both the LM
    /// context window and the persistence log.
    pub fn push_assistant_call(&mut self, tool_calls_json: Option<serde_json::Value>) {
        let now = Utc::now();
        self.messages.push(LmMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: tool_calls_json.clone(),
            tool_call_id: None,
        });
        self.new_messages.push(ConversationMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: tool_calls_json,
            tool_call_id: None,
            timestamp: now,
        });
    }

    /// Push a tool-result message to both the LM context window and the
    /// persistence log. Always call after `push_assistant_call` so the
    /// assistant→tool pair stays well-formed.
    pub fn push_tool_result(&mut self, call_id: String, content: String) {
        let now = Utc::now();
        self.messages.push(LmMessage {
            role: "tool".into(),
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: Some(call_id.clone()),
        });
        self.new_messages.push(ConversationMessage {
            role: "tool".into(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(call_id),
            timestamp: now,
        });
    }
}
