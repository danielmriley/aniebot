use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::cli_wrapper;
use crate::config::Config;
use crate::memory::{self, ConversationMessage};

const HISTORY_WINDOW: usize = 20;
const MEMORY_ENTRIES: usize = 5;

// ---------------------------------------------------------------------------
// LM Studio response schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OrchestratorResponse {
    action: String, // "direct_reply" or "delegate_cli"
    reply: Option<String>,
    task: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct LmMessage {
    role: String,
    content: String,
}

// ---------------------------------------------------------------------------
// Public entrypoint
// ---------------------------------------------------------------------------

pub async fn process_message(config: Arc<Config>, chat_id: i64, user_input: &str) -> String {
    match try_process(config, chat_id, user_input).await {
        Ok(reply) => reply,
        Err(e) => {
            tracing::error!("Orchestrator error: {}", e);
            format!("⚠️ Something went wrong: {}", e)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

async fn try_process(config: Arc<Config>, chat_id: i64, user_input: &str) -> anyhow::Result<String> {
    // 1. Load context — personality, conversation history, recent memory.
    let (personality, history, recent_memory) = tokio::try_join!(
        load_personality(),
        memory::load_history(chat_id),
        memory::load_recent_memory(MEMORY_ENTRIES),
    )?;

    // 2. Build the windowed history slice for the prompt.
    let window_start = history.len().saturating_sub(HISTORY_WINDOW * 2);
    let history_window = &history[window_start..];

    // 3. Build the system prompt.
    let memory_bullets: String = if recent_memory.is_empty() {
        String::from("(no prior interactions yet)")
    } else {
        recent_memory
            .iter()
            .map(|e| format!("- [{}] User: {} → You: {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.user_msg, e.assistant_reply))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let system_prompt = format!(
        r#"{personality}

## Recent memory
{memory_bullets}

## Response format
You must reply with ONLY a single JSON object — no markdown, no explanation, no code fences.
Choose one of these two shapes:
{{"action":"direct_reply","reply":"your reply here"}}
{{"action":"delegate_cli","task":"precise task description for copilot"}}

Use "delegate_cli" when the user wants ANY of the following:
- Code written, edited, or explained
- Files created, read, or modified
- Shell commands run
- Real-time or current information: weather, news, stock prices, sports scores, anything that changes over time
- Web searches or looking something up online
- Anything you are unsure about or that requires information beyond your training data
The delegate tool (copilot) has full internet access, web search, and can run shell commands — always prefer delegating over guessing.
Use "direct_reply" ONLY for pure conversation, opinions, or things you are completely certain about from your own knowledge.

## Examples
User: "what time is it?"
{{"action":"delegate_cli","task":"Run the `date` command and report the current date and time."}}

User: "what's the weather tomorrow in Apopka Florida?"
{{"action":"delegate_cli","task":"Search the web for the weather forecast for tomorrow in Apopka, Florida and summarize it."}}

User: "add error handling to main.rs"
{{"action":"delegate_cli","task":"Add proper error handling to main.rs in the current workspace. Use anyhow for error propagation."}}

User: "how are you?"
{{"action":"direct_reply","reply":"Functioning within normal parameters. What do you need?"}}"#,
        personality = personality,
        memory_bullets = memory_bullets,
    );

    // 4. Build the messages array: system + history window + current user message.
    let mut messages: Vec<LmMessage> = Vec::with_capacity(1 + history_window.len() + 1);
    messages.push(LmMessage { role: "system".into(), content: system_prompt });
    for msg in history_window {
        messages.push(LmMessage { role: msg.role.clone(), content: msg.content.clone() });
    }
    messages.push(LmMessage { role: "user".into(), content: user_input.to_string() });

    // 5. Call LM Studio. Retry once on JSON parse failure.
    let reply = match call_lm_studio(&config, &mut messages).await {
        Ok(reply) => reply,
        Err(e) => {
            tracing::error!("LM Studio error: {}", e);
            return Err(e);
        }
    };

    // 6. Persist both turns to conversation history and the global memory log.
    let now = Utc::now();
    let turns = vec![
        ConversationMessage { role: "user".into(), content: user_input.to_string(), timestamp: now },
        ConversationMessage { role: "assistant".into(), content: reply.clone(), timestamp: now },
    ];
    if let Err(e) = memory::append_messages(chat_id, &turns).await {
        tracing::warn!("Failed to persist conversation: {}", e);
    }
    if let Err(e) = memory::store_interaction(chat_id, user_input, &reply).await {
        tracing::warn!("Failed to write memory log: {}", e);
    }

    Ok(reply)
}

async fn call_lm_studio(config: &Config, messages: &mut Vec<LmMessage>) -> anyhow::Result<String> {
    let raw = post_to_lm_studio(config, messages).await?;

    // First parse attempt.
    match parse_and_dispatch(config, &raw).await {
        Ok(reply) => return Ok(reply),
        Err(_) => {
            tracing::warn!("LM Studio returned non-JSON, retrying with correction prompt");
        }
    }

    // Retry: append the bad reply and a correction instruction.
    messages.push(LmMessage { role: "assistant".into(), content: raw.clone() });
    messages.push(LmMessage {
        role: "user".into(),
        content: "Your response was not valid JSON. Reply with ONLY the JSON object, nothing else. No markdown, no backticks, no explanation.".into(),
    });

    let raw2 = post_to_lm_studio(config, messages).await?;

    match parse_and_dispatch(config, &raw2).await {
        Ok(reply) => Ok(reply),
        Err(_) => {
            // Graceful degradation: treat whatever the model said as a direct reply.
            tracing::warn!("LM Studio retry also failed to produce JSON — using raw content as reply");
            Ok(raw2)
        }
    }
}

async fn post_to_lm_studio(config: &Config, messages: &[LmMessage]) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.lm_studio_url);

    let payload = json!({
        "model": config.model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 400,
    });

    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .await
        .context("Failed to reach LM Studio — is it running?")?;

    if !response.status().is_success() {
        anyhow::bail!("LM Studio returned HTTP {}", response.status());
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse LM Studio response")?;

    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .context("LM Studio response missing content field")?
        .trim()
        .to_string();

    Ok(content)
}

async fn parse_and_dispatch(config: &Config, raw: &str) -> anyhow::Result<String> {
    // Strip <think>...</think> reasoning block emitted by reasoning models.
    let after_think = if let Some(end) = raw.find("</think>") {
        &raw[end + "</think>".len()..]
    } else {
        raw
    };

    // Strip optional markdown code fences (some models wrap JSON in ```json ... ```)
    let cleaned = after_think
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let parsed: OrchestratorResponse = serde_json::from_str(cleaned)
        .context("Response is not a valid OrchestratorResponse JSON")?;

    match parsed.action.as_str() {
        "direct_reply" => {
            let reply = parsed.reply.unwrap_or_else(|| "(empty reply)".into());
            Ok(reply)
        }
        "delegate_cli" => {
            let task = parsed.task.context("delegate_cli action missing 'task' field")?;
            tracing::info!("Delegating to copilot: {}", task);
            match cli_wrapper::run(config, &task).await {
                Ok(output) => Ok(format!("✅ Done!\n\n{}", output)),
                Err(e) => Ok(format!("❌ CLI error: {}", e)),
            }
        }
        other => anyhow::bail!("Unknown action: {}", other),
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
