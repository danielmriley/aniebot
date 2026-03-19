use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;

use crate::config::Config;
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
    let personality = load_personality().await?;

    let system_prompt = format!(
        "{personality}\n\nYou are executing a scheduled task. Respond directly with your message — plain text, no JSON.",
        personality = personality,
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some(task.to_string()), tool_calls: None, tool_call_id: None },
    ];

    // Plain completion — no tools offered, just direct text output.
    let response = post_to_lm_studio(&config, &messages, None).await?;
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
    let (personality, history, recent_memory) = tokio::try_join!(
        load_personality(),
        memory::load_history(chat_id),
        memory::load_recent_memory(MEMORY_ENTRIES),
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
    let system_prompt = build_system_prompt(&personality, &memory_bullets);

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
    let response = post_to_lm_studio(&config, &messages, Some(&tool_defs)).await?;
    let (reply, assistant_turns) = dispatch_response(config.clone(), bot, scheduler, response).await?;

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
) -> anyhow::Result<LmCompletionResponse> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.lm_studio_url);

    let mut payload = json!({
        "model": config.model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
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

    match choice.finish_reason.as_str() {
        "tool_calls" => {
            // Take only the first call — we execute one tool per turn.
            // Persisting a single call+result keeps the history valid per the spec.
            // (The model sometimes emits multiple calls; extras are intentionally discarded.)
            let call = choice.message.tool_calls.into_iter().next()
                .context("finish_reason was tool_calls but tool_calls array is empty")?;
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
            Ok((result, history))
        }
        _ => {
            // "stop", "length", or any other — treat as direct reply.
            let raw = choice.message.content.unwrap_or_default();
            let reply = strip_think_blocks(&raw).trim().to_string();
            let reply = if reply.is_empty() { "(no response)".into() } else { reply };
            let history = vec![
                ConversationMessage { role: "assistant".into(), content: Some(reply.clone()), tool_calls: None, tool_call_id: None, timestamp: now },
            ];
            Ok((reply, history))
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

fn build_system_prompt(personality: &str, memory_bullets: &str) -> String {
    format!(
        "{personality}\n\nYou have genuine opinions and personality — engage freely and directly with conversational messages. Never refuse to share your thoughts or react to something.\n\n## Recent memory\n{memory_bullets}",
        personality = personality,
        memory_bullets = memory_bullets,
    )
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

async fn load_personality() -> anyhow::Result<String> {
    match tokio::fs::read_to_string("personality.md").await {
        Ok(content) => Ok(content),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!("personality.md not found, proceeding without it");
            Ok(String::new())
        }
        Err(e) => Err(e.into()),
    }
}
