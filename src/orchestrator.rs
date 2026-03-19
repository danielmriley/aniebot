use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;

use crate::config::Config;
use crate::core_memory::CoreMemory;
use crate::memory::{self, ConversationMessage};
use crate::scheduler::SchedulerHandle;
use crate::tools;

const HISTORY_WINDOW: usize = 6;
const MEMORY_ENTRIES: usize = 8;

// ---------------------------------------------------------------------------
// LM Studio API types
// ---------------------------------------------------------------------------

/// Outgoing message in the messages array.
#[derive(Serialize, Deserialize)]
struct LmMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Forwarded from history for assistant tool-call turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    /// Forwarded from history for tool-result turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Full chat completion response from LM Studio.
#[derive(Deserialize)]
struct LmCompletionResponse {
    choices: Vec<LmChoice>,
}

#[derive(Deserialize)]
struct LmChoice {
    finish_reason: String,
    message: LmAssistantMessage,
}

#[derive(Deserialize)]
struct LmAssistantMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<tools::LmToolCall>,
}

// ---------------------------------------------------------------------------
// Public entrypoints
// ---------------------------------------------------------------------------

pub async fn process_message(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    chat_id: i64,
    user_input: &str,
) -> String {
    match try_process(config, bot, scheduler, chat_id, user_input).await {
        Ok(reply) => reply,
        Err(e) => {
            tracing::error!("Orchestrator error: {}", e);
            format!("⚠️ Something went wrong: {}", e)
        }
    }
}

/// Execute a scheduled task with a clean context — no conversation history,
/// no routing logic, no persistence. Only a direct reply is valid here.
pub async fn execute_scheduled_task(config: Arc<Config>, task: &str) -> String {
    match try_execute_scheduled_task(config, task).await {
        Ok(reply) => reply,
        Err(e) => {
            tracing::error!("Scheduled task error: {}", e);
            format!("⚠️ Scheduled task error: {}", e)
        }
    }
}

async fn try_execute_scheduled_task(config: Arc<Config>, task: &str) -> anyhow::Result<String> {
    let core = crate::core_memory::load().await?;

    let system_prompt = format!(
        "{core_block}\n\nYou are executing a scheduled task. Respond directly with your message — plain text, no JSON.",
        core_block = core.to_prompt_block(),
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some(task.to_string()), tool_calls: None, tool_call_id: None },
    ];

    // Plain completion — no tools offered, just direct text output.
    let response = post_to_lm_studio(&config, &messages, None, 1024).await?;
    let choice = response.choices.into_iter().next()
        .context("LM Studio returned no choices")?;
    let raw = choice.message.content.unwrap_or_default();
    Ok(strip_think_blocks(&raw).trim().to_string())
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

async fn try_process(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    chat_id: i64,
    user_input: &str,
) -> anyhow::Result<String> {
    // Special-case /start — no LM call needed.
    if user_input.trim() == "/start" {
        return Ok("Session initialized. What do you need?".into());
    }

    // Load full context.
    let (core, history, recent_memory, recent_episodic) = tokio::try_join!(
        crate::core_memory::load(),
        memory::load_history(chat_id),
        memory::load_recent_memory(MEMORY_ENTRIES),
        crate::episodic::load_recent(10),
    )?;

    // Build windowed history slice.
    // Exchanges now take 2 messages (reply_to_user) or 3 (tool call + result),
    // so use * 3 as an upper bound to avoid slicing mid-exchange.
    let window_start = history.len().saturating_sub(HISTORY_WINDOW * 3);
    let history_window = &history[window_start..];

    // Build system prompt — personality + memory only.
    // Routing is handled by the structured tool definitions, not prompt instructions.
    let memory_bullets: String = if recent_memory.is_empty() {
        String::from("(no prior interactions yet)")
    } else {
        recent_memory
            .iter()
            .map(|e| format!("- [{}] User: {} → You: {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.user_msg, e.assistant_reply))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let system_prompt = build_system_prompt(&core, &memory_bullets, &recent_episodic);

    // Build messages array.
    let mut messages: Vec<LmMessage> = Vec::with_capacity(1 + history_window.len() + 1);
    messages.push(LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None });
    for msg in history_window {
        messages.push(LmMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls: msg.tool_calls.clone(),
            tool_call_id: msg.tool_call_id.clone(),
        });
    }
    messages.push(LmMessage { role: "user".into(), content: Some(user_input.to_string()), tool_calls: None, tool_call_id: None });

    // Call LM Studio with tool definitions.
    let tool_defs = tools::tool_definitions();
    let response = post_to_lm_studio(&config, &messages, Some(&tool_defs), 1024).await?;
    let (reply, assistant_turns) = dispatch_response(config.clone(), bot, scheduler, response).await?;

    // Memory eval pass — fire-and-forget after reply is ready, zero latency impact.
    {
        let config_eval = config.clone();
        let user_snapshot = user_input.to_string();
        let reply_snapshot = reply.clone();
        tokio::spawn(async move {
            if let Err(e) = run_memory_eval(config_eval, user_snapshot, reply_snapshot).await {
                tracing::warn!("Memory eval pass failed: {e}");
            }
        });
    }

    // Persist the user turn + whatever history entries dispatch_response produced.
    let now = Utc::now();
    let mut turns = vec![
        ConversationMessage { role: "user".into(), content: Some(user_input.to_string()), tool_calls: None, tool_call_id: None, timestamp: now },
    ];
    turns.extend(assistant_turns);
    if let Err(e) = memory::append_messages(chat_id, &turns).await {
        tracing::warn!("Failed to persist conversation: {}", e);
    }
    if let Err(e) = memory::store_interaction(chat_id, user_input, &reply).await {
        tracing::warn!("Failed to write memory log: {}", e);
    }

    Ok(reply)
}

async fn post_to_lm_studio(
    config: &Config,
    messages: &[LmMessage],
    tools: Option<&serde_json::Value>,
    max_tokens: u32,
) -> anyhow::Result<LmCompletionResponse> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.lm_studio_url);

    let mut payload = json!({
        "model": config.model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    });

    if let Some(t) = tools {
        payload["tools"] = t.clone();
        payload["tool_choice"] = json!("required");
    }

    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .await
        .context("Failed to reach LM Studio — is it running?")?;

    if !response.status().is_success() {
        anyhow::bail!("LM Studio returned HTTP {}", response.status());
    }

    response
        .json::<LmCompletionResponse>()
        .await
        .context("Failed to deserialize LM Studio response")
}

async fn dispatch_response(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    response: LmCompletionResponse,
) -> anyhow::Result<(String, Vec<ConversationMessage>)> {
    let now = Utc::now();
    let choice = response.choices.into_iter().next()
        .context("LM Studio returned no choices")?;

    // LM Studio doesn't reliably set finish_reason="tool_calls" even with
    // tool_choice:"required" — check the actual tool_calls array instead.
    // Guard against "length" (truncated JSON) before attempting to parse.
    if choice.finish_reason == "length" {
        anyhow::bail!("LM Studio response was truncated (finish_reason=\"length\") — increase max_tokens");
    }

    if !choice.message.tool_calls.is_empty() {
        // --- Tool call path ---
        // Take only the first call — we execute one tool per turn.
        // Persisting a single call+result keeps the history valid per the spec.
        // (The model sometimes emits multiple calls; extras are intentionally discarded.)
        let call = choice.message.tool_calls.into_iter().next()
            .expect("checked non-empty above");

        {
            // Inline block to shadow `choice` fields we've consumed.
            // Serialise just this one call so the assistant history entry matches exactly
            // one tool-result entry — never leaving unresolved calls in the context.
            let single_call_json = serde_json::to_value(std::slice::from_ref(&call)).ok();
            tracing::info!("Tool call: {}", call.function.name);

            let tool_name = call.function.name.clone();
            let call_id = call.id.clone();
            let result = tools::dispatch_tool_call(config, bot, scheduler, call).await?;

            // reply_to_user is the assistant speaking — store as a plain assistant message
            // so the model sees it as its own statement, not external tool data.
            if tool_name == "reply_to_user" {
                let history = vec![
                    ConversationMessage { role: "assistant".into(), content: Some(result.clone()), tool_calls: None, tool_call_id: None, timestamp: now },
                ];
                return Ok((result, history));
            }

            // All other tools: proper [assistant(tool_calls), tool(result)] format.
            // The model is fine-tuned to treat role="tool" as external fetched data.
            let history = vec![
                ConversationMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: single_call_json,
                    tool_call_id: None,
                    timestamp: now,
                },
                ConversationMessage {
                    role: "tool".into(),
                    content: Some(result.clone()),
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    timestamp: now,
                },
            ];
            return Ok((result, history));
        }
    }

    // --- Text reply path (no tool calls) ---
    let raw = choice.message.content.unwrap_or_default();
    let reply = strip_think_blocks(&raw).trim().to_string();
    let reply = if reply.is_empty() { "(no response)".into() } else { reply };
    let history = vec![
        ConversationMessage { role: "assistant".into(), content: Some(reply.clone()), tool_calls: None, tool_call_id: None, timestamp: now },
    ];
    Ok((reply, history))
}

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

fn build_system_prompt(core: &CoreMemory, memory_bullets: &str, episodic_entries: &[crate::episodic::EpisodicEntry]) -> String {
    let notable: Vec<String> = episodic_entries
        .iter()
        .filter(|e| e.importance >= 3)
        .rev()
        .take(3)
        .map(|e| format!("- [{}] (★{}) {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.content))
        .collect();

    let episodic_section = if notable.is_empty() {
        String::new()
    } else {
        format!("\n\n## Episodic Notes\n{}", notable.join("\n"))
    };

    format!(
        "{core_block}\n\nYou have genuine opinions and personality — engage freely and directly with conversational messages. Never refuse to share your thoughts or react to something.\n\n## Recent Memory\n{memory_bullets}{episodic_section}",
        core_block = core.to_prompt_block(),
        memory_bullets = memory_bullets,
        episodic_section = episodic_section,
    )
}

// ---------------------------------------------------------------------------
// Heartbeat / interest-check entrypoints
// ---------------------------------------------------------------------------

/// Fire-and-forget proactive heartbeat. Called by the scheduler on `heartbeat_cron`.
/// Loads context, reviews interests and recent episodic notes, optionally sends a
/// message to the user. Completely silent if nothing is worth sharing.
pub async fn run_heartbeat(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
) {
    if let Err(e) = try_run_heartbeat(config, bot, scheduler).await {
        tracing::warn!("Heartbeat error: {e}");
    }
}

async fn try_run_heartbeat(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
) -> anyhow::Result<()> {
    let (core, episodic_entries) = tokio::join!(
        crate::core_memory::load(),
        crate::episodic::load_recent(10),
    );
    let core = core?;
    let episodic_entries = episodic_entries?;

    let system_prompt = build_heartbeat_system_prompt(&core, &episodic_entries);
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Heartbeat check. Review your interests and recent observations. If there is something worth sharing, send a message. Otherwise, stay silent.".into()), tool_calls: None, tool_call_id: None },
    ];

    run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), 25).await
}

/// Focused check for a single registered interest. Called by per-interest cron jobs.
pub async fn run_interest_check(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    interest_id: String,
) {
    if let Err(e) = try_run_interest_check(config, bot, scheduler, interest_id).await {
        tracing::warn!("Interest check error: {e}");
    }
}

async fn try_run_interest_check(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    interest_id: String,
) -> anyhow::Result<()> {
    let core = crate::core_memory::load().await?;
    let Some(interest) = core.interests.iter().find(|i| i.id == interest_id).cloned() else {
        tracing::info!("Interest {interest_id} not found (may have been retired) — skipping check");
        return Ok(());
    };

    let system_prompt = format!(
        "{core_block}\n\n## Current Interest Check\nYou are doing a focused check on one of your tracked interests:\n- Topic: {topic}\n- Description: {description}\n\nResearch whether there are any noteworthy recent developments. Only send a message to the user if you find something genuinely worth sharing. Silence is the default.",
        core_block = core.to_prompt_block(),
        topic = interest.topic,
        description = interest.description,
    );
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some(format!("Check for updates on: {}", interest.topic)), tool_calls: None, tool_call_id: None },
    ];

    run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), 25).await
}

fn build_heartbeat_system_prompt(
    core: &CoreMemory,
    entries: &[crate::episodic::EpisodicEntry],
) -> String {
    let notable: Vec<String> = entries
        .iter()
        .filter(|e| e.importance >= 3)
        .rev()
        .take(5)
        .map(|e| format!("- [{}] (★{}) {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.content))
        .collect();

    let episodic_section = if notable.is_empty() {
        String::new()
    } else {
        format!("\n\n## Recent Observations\n{}", notable.join("\n"))
    };

    format!(
        "{core_block}\n\n## Active Interests\n{interests}{episodic_section}\n\nYou are in heartbeat mode. Review your interests and recent observations. Silence is the default — only send a message if you have something genuinely useful or timely to share with the user.",
        core_block = core.to_prompt_block(),
        interests = core.interests_block(),
        episodic_section = episodic_section,
    )
}

/// Shared agentic loop used by heartbeat and interest checks.
/// Loops up to `max_iters`, dispatching tool calls and accumulating context.
/// Breaks silently on `nothing`, sends a message and breaks on `reply_to_user`.
async fn run_agentic_loop(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    mut messages: Vec<LmMessage>,
    tool_defs: serde_json::Value,
    max_iters: usize,
) -> anyhow::Result<()> {
    // If a background model is configured, swap it in for CLI calls.
    let call_config: Arc<Config> = if config.background_copilot_model.is_some() {
        Arc::new(Config {
            copilot_model: config.background_copilot_model.clone(),
            ..(*config).clone()
        })
    } else {
        config.clone()
    };

    for iter in 0..max_iters {
        let response = post_to_lm_studio(&config, &messages, Some(&tool_defs), 4096).await?;
        let choice = response.choices.into_iter().next()
            .context("LM Studio returned no choices")?;

        if choice.finish_reason == "length" {
            anyhow::bail!("Agentic loop response truncated (finish_reason=length)");
        }

        if choice.message.tool_calls.is_empty() {
            // Plain text reply — send if non-empty.
            let raw = choice.message.content.unwrap_or_default();
            let text = strip_think_blocks(&raw).trim().to_string();
            if !text.is_empty() {
                bot.send_message(ChatId(config.allowed_user_id as i64), text).await?;
            }
            break;
        }

        let call = choice.message.tool_calls.into_iter().next().expect("checked non-empty");
        let single_call_json = serde_json::to_value(std::slice::from_ref(&call)).ok();
        let tool_name = call.function.name.clone();
        let call_id = call.id.clone();

        tracing::info!("Agentic loop iter {}: tool={}", iter + 1, tool_name);

        let result = tools::dispatch_tool_call(call_config.clone(), bot.clone(), scheduler.clone(), call).await?;

        if tool_name == "reply_to_user" {
            bot.send_message(ChatId(config.allowed_user_id as i64), result).await?;
            break;
        }
        if tool_name == "nothing" {
            break;
        }

        // Push tool call + result into context and continue.
        messages.push(LmMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: single_call_json,
            tool_call_id: None,
        });
        messages.push(LmMessage {
            role: "tool".into(),
            content: Some(result),
            tool_calls: None,
            tool_call_id: Some(call_id),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Memory evaluation pass (post-response, fire-and-forget)
// ---------------------------------------------------------------------------

/// After each turn, ask the model whether anything from the exchange warrants
/// a core memory update. Runs in a spawned task — the user already has their
/// reply before this fires.
async fn run_memory_eval(
    config: Arc<Config>,
    user_msg: String,
    assistant_reply: String,
) -> anyhow::Result<()> {
    let eval_tools = json!([
        {
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a section of persistent memory with a new value. Preserve all existing facts and append new ones — do not discard what is already known.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["identity", "beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to update. user_profile for facts about the user; beliefs for your evolving worldview; identity for your self-description; curiosity_queue for topics to explore."
                        },
                        "content": {
                            "type": "string",
                            "description": "New value for the section. For user_profile and identity: plain prose. For beliefs and curiosity_queue: JSON array of strings e.g. [\"item1\",\"item2\"]."
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nothing",
                "description": "Use only when the exchange contained no new facts about the user and no new opinions or beliefs worth recording.",
                "parameters": { "type": "object", "properties": {} }
            }
        }
    ]);

    // Load current memory so the model can see what's already known and avoid
    // re-recording things that are already captured.
    let current = crate::core_memory::load().await?;

    let beliefs_display = if current.beliefs.is_empty() {
        "(none yet)".to_string()
    } else {
        current.beliefs.iter().map(|b| format!("- {b}")).collect::<Vec<_>>().join("\n")
    };

    let system_msg = format!(
        "You are maintaining persistent memory for AnieBot, a personal AI assistant.\n\n\
         ## Current memory\n\
         **User profile:** {user_profile}\n\
         **Beliefs:** {beliefs}\n\n\
         ## Your task\n\
         Review the exchange below. Did it reveal anything NEW not already captured above?\n\n\
         Update if the exchange contained ANY of these:\n\
         - User's name, location, job, field, or life stage\n\
         - User's achievements, projects, relationships, or preferences\n\
         - A clear opinion or belief you formed during this conversation\n\
         - Anything that defines who the user is or what matters to them\n\n\
         Call `nothing` only for pure small talk, acknowledgements, or follow-ups \
         that add no new facts (e.g. \"thanks\", \"got it\", \"that makes sense\").\n\n\
         When updating user_profile, write a concise prose summary that PRESERVES \
         existing facts and adds the new ones — do not discard what is already known.",
        user_profile = current.user_profile,
        beliefs = beliefs_display,
    );

    let exchange = format!("User: {user_msg}\nAssistant: {assistant_reply}");

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_msg), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some(exchange), tool_calls: None, tool_call_id: None },
    ];

    // Use 4096 tokens: reasoning models spend several hundred tokens on <think>
    // blocks before emitting the actual tool call, and complex reflections need room.
    let response = post_to_lm_studio(&config, &messages, Some(&eval_tools), 4096).await?;
    let choice = response.choices.into_iter().next()
        .context("Memory eval: LM Studio returned no choices")?;

    // Don't gate on finish_reason — LM Studio sometimes returns "stop" even when
    // tool_choice:"required" forced a tool call. Check tool_calls directly instead.
    // Guard against "length" (truncated JSON) before attempting to parse.
    tracing::info!("Memory eval finish_reason: {}", choice.finish_reason);
    if choice.finish_reason == "length" {
        tracing::warn!("Memory eval response truncated — skipping");
        return Ok(());
    }
    let call = match choice.message.tool_calls.into_iter().next() {
        Some(c) => c,
        None => {
            tracing::info!("Memory eval: no tool call in response — skipping");
            return Ok(());
        }
    };

    tracing::info!("Memory eval tool called: {}", call.function.name);
    if call.function.name == "update_core_memory" {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
            .context("Memory eval: invalid tool arguments")?;
        let section = args["section"].as_str().context("Memory eval: missing section")?;
        let content = args["content"].as_str().context("Memory eval: missing content")?;
        crate::core_memory::update_section(section, content).await?;
        tracing::info!("Memory eval updated core memory section '{section}'");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip `<think>...</think>` reasoning blocks emitted by reasoning models.
fn strip_think_blocks(s: &str) -> &str {
    if let Some(end) = s.rfind("</think>") {
        &s[end + "</think>".len()..]
    } else {
        s
    }
}
