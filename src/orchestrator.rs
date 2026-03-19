use std::sync::Arc;
use std::time::Duration;

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
    let (reply, assistant_turns) = dispatch_response(config.clone(), bot.clone(), scheduler.clone(), response).await?;

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

    // Consolidation trigger — fire-and-forget if enough new episodic entries have
    // accumulated since the last consolidation pass.
    {
        let config_cons = config.clone();
        let bot_cons = bot.clone();
        let scheduler_cons = scheduler.clone();
        tokio::spawn(async move {
            let since = match crate::core_memory::load().await {
                Ok(cm) => cm.last_consolidation_at.unwrap_or(chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap()),
                Err(_) => return,
            };
            match crate::episodic::count_since(since).await {
                Ok(n) if n >= config_cons.consolidation_threshold => {
                    tracing::info!("Consolidation triggered ({n} entries since last pass)");
                    run_consolidation(config_cons, bot_cons, scheduler_cons).await;
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("count_since failed: {e}"),
            }
        });
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
        .timeout(Duration::from_secs(900))
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

    let now_utc = Utc::now().format("%Y-%m-%dT%H:%M:%SZ");

    format!(
        "{core_block}\n\nCurrent UTC time: {now_utc}\n\nYou have genuine opinions and personality — engage freely and directly with conversational messages. Never refuse to share your thoughts or react to something.\n\n## Recent Memory\n{memory_bullets}{episodic_section}",
        core_block = core.to_prompt_block(),
        now_utc = now_utc,
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
    let (core, all_recent) = tokio::join!(
        crate::core_memory::load(),
        crate::episodic::load_recent(50),
    );
    let core = core?;
    let all_recent = all_recent?;

    // Split recent episodic entries into general observations and prior heartbeat sends/checks.
    // Only treat entries within the last 6 hours as "recent" — older entries shouldn't suppress
    // legitimate new updates on the same topic.
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(6);
    let recent_sends: Vec<_> = all_recent
        .iter()
        .filter(|e| e.tags.contains(&"heartbeat-sent".to_string()) && e.timestamp > cutoff)
        .rev()
        .take(5)
        .cloned()
        .collect();
    let recent_checks: Vec<_> = all_recent
        .iter()
        .filter(|e| e.tags.contains(&"heartbeat-checked".to_string()) && e.timestamp > cutoff)
        .rev()
        .take(10)
        .cloned()
        .collect();

    let system_prompt = build_heartbeat_system_prompt(&core, &all_recent, &recent_sends, &recent_checks);
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Heartbeat check. Review your interests and recent observations. If there is something worth sharing, send a message. Otherwise, stay silent.".into()), tool_calls: None, tool_call_id: None },
    ];

    let sent = run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), 12).await?;

    // Always write a heartbeat-checked entry with topics covered, so the next heartbeat cycle
    // can skip re-researching interests that were already checked and found nothing new.
    {
        let interest_topics = core.interests.iter().map(|i| i.topic.as_str()).collect::<Vec<_>>().join(", ");
        let content = if interest_topics.is_empty() {
            "[heartbeat checked] No active interests.".to_string()
        } else {
            format!("[heartbeat checked] Topics checked: {interest_topics}")
        };
        let entry = crate::episodic::EpisodicEntry {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            tags: vec!["heartbeat-checked".to_string()],
            importance: 1,
            timestamp: chrono::Utc::now(),
            promoted: false,
        };
        let _ = crate::episodic::append(&entry).await;
    }

    // Auto-log whatever was sent so the next heartbeat can see it and avoid repeating.
    // Include the interest topics covered so the next cycle can match by name, not prose.
    if let Some(text) = sent {
        let interest_topics = core.interests.iter().map(|i| i.topic.as_str()).collect::<Vec<_>>().join(", ");
        let summary = text.chars().take(150).collect::<String>();
        let content = if interest_topics.is_empty() {
            format!("[heartbeat sent] {summary}")
        } else {
            format!("[heartbeat sent] Topics covered: {interest_topics}. Summary: {summary}")
        };
        let entry = crate::episodic::EpisodicEntry {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            tags: vec!["heartbeat-sent".to_string()],
            importance: 2,
            timestamp: chrono::Utc::now(),
            promoted: false,
        };
        let _ = crate::episodic::append(&entry).await;
    }

    Ok(())
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

    run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), 12).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Consolidation
// ---------------------------------------------------------------------------

/// Fire-and-forget consolidation pass. Called from try_process when the episodic
/// entry count since last consolidation exceeds CONSOLIDATION_THRESHOLD.
pub async fn run_consolidation(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
) {
    if let Err(e) = try_run_consolidation(config, bot, scheduler).await {
        tracing::warn!("Consolidation error: {e}");
    }
}

async fn try_run_consolidation(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
) -> anyhow::Result<()> {
    tracing::info!("Consolidation pass starting");

    let (core, entries) = tokio::join!(
        crate::core_memory::load(),
        crate::episodic::load_recent(20),
    );
    let core = core?;
    let entries = entries?;

    let notable: Vec<String> = entries
        .iter()
        .filter(|e| e.importance >= 2)
        .rev()
        .take(20)
        .map(|e| format!("- [{}] (★{}) [{}] {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.tags.join(", "), e.content))
        .collect();

    let episodic_block = if notable.is_empty() {
        "(no recent observations)".to_string()
    } else {
        notable.join("\n")
    };

    let high_sig: Vec<String> = entries
        .iter()
        .filter(|e| e.importance >= 4)
        .rev()
        .take(10)
        .map(|e| format!("- [{}] (★{}) [{}] {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.tags.join(", "), e.content))
        .collect();

    let high_sig_section = if high_sig.is_empty() {
        String::new()
    } else {
        format!("\n\n## High-Significance Episodes\n{}", high_sig.join("\n"))
    };

    let identity_question = if high_sig.is_empty() {
        ""
    } else {
        "\n5. Did any high-significance event fundamentally change who you are or what you stand for? → update_core_memory(\"identity\", ...) [minimum surgical edit — add, revise, or remove only the specific clause that changed; do not rewrite wholesale]"
    };

    let system_prompt = format!(
        "{core_block}\n\n## Recent Episodic Observations\n{episodic_block}{high_sig_section}\n\n\
You are in consolidation mode. Make a single, complete pass through these observations and call tools as needed:\n\
1. Did you learn anything new or significant about the user? → update_core_memory(\"user_profile\", ...) [replace the full field with the updated text]\n\
2. Did anything refine your beliefs or opinions? → update_core_memory(\"beliefs\", ...) [JSON array string, e.g. \"[\\\"Formal verification is underutilized in industry\\\", \\\"Daniel values deep work over shallow multitasking\\\"]\"] Be direct — record concrete first-person conclusions, not vague observations.\n\
3. Any observations worth preserving permanently? → remember(content, tags, 5)\n\
4. Did any unresolved questions or interesting topics come up worth investigating later? → update_core_memory(\"curiosity_queue\", ...) [JSON array of short topic strings, e.g. \"[\\\"How does Lean 4 compare to Coq for software verification?\\\"]\"]. Add to existing items, don't replace them.\n\
5. Any interests to add or retire? → add_interest / retire_interest{identity_question}\n\
\nIMPORTANT: Make ALL your updates now in this one pass. Be conservative — only update if something is genuinely new. Once you have made all your updates (or if nothing warrants updating), you MUST call nothing to end the consolidation. Do not call nothing before you have considered all the questions above.",
        core_block = core.to_prompt_block(),
        episodic_block = episodic_block,
        high_sig_section = high_sig_section,
        identity_question = identity_question,
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Consolidation cycle. Reflect and update as needed.".into()), tool_calls: None, tool_call_id: None },
    ];

    run_agentic_loop(config.clone(), bot, scheduler, messages, tools::consolidation_tool_definitions(), 10).await?; // return value ignored for consolidation

    // Reload core memory after LM pass (model may have added/retired interests).
    // Apply health decay: -10 per consolidation pass for each interest whose topic
    // isn't mentioned in any of the loaded episodic entries. Auto-retire at 0.
    let mut cm_after = crate::core_memory::load().await?;
    let mut retired_ids: Vec<String> = vec![];
    for interest in cm_after.interests.iter_mut() {
        let topic_lower = interest.topic.to_lowercase();
        let mentioned = entries.iter().any(|e| e.content.to_lowercase().contains(&topic_lower));
        if !mentioned {
            interest.health = interest.health.saturating_sub(10);
            if interest.health == 0 {
                retired_ids.push(interest.id.clone());
            }
        } else {
            interest.last_seen_date = chrono::Utc::now().date_naive().to_string();
        }
    }
    cm_after.interests.retain(|i| !retired_ids.contains(&i.id));
    for id in &retired_ids {
        tracing::info!("Interest {id} health reached 0 — auto-retired");
    }
    cm_after.last_consolidation_at = Some(chrono::Utc::now());
    crate::core_memory::save(&cm_after).await?;

    // Promote high-importance episodic entries to archival.
    match crate::episodic::promote_to_archival(4).await {
        Ok(n) if n > 0 => tracing::info!("Promoted {n} episodic entries to archival"),
        Ok(_) => {}
        Err(e) => tracing::warn!("Archival promotion failed: {e}"),
    }

    tracing::info!("Consolidation pass complete");
    Ok(())
}

fn build_heartbeat_system_prompt(
    core: &CoreMemory,
    all_entries: &[crate::episodic::EpisodicEntry],
    recent_sends: &[crate::episodic::EpisodicEntry],
    recent_checks: &[crate::episodic::EpisodicEntry],
) -> String {
    let notable: Vec<String> = all_entries
        .iter()
        .filter(|e| e.importance >= 3 && !e.tags.contains(&"heartbeat-sent".to_string()))
        .rev()
        .take(5)
        .map(|e| format!("- [{}] (★{}) {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.content))
        .collect();

    let episodic_section = if notable.is_empty() {
        String::new()
    } else {
        format!("\n\n## Recent Observations\n{}", notable.join("\n"))
    };

    let sends_section = if recent_sends.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = recent_sends
            .iter()
            .map(|e| format!("- [{}] {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.content))
            .collect();
        format!("\n\n## Recently Sent to You\n{}", lines.join("\n"))
    };

    let checks_section = if recent_checks.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = recent_checks
            .iter()
            .map(|e| format!("- [{}] {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.content))
            .collect();
        format!("\n\n## Recently Checked (nothing new found)\n{}", lines.join("\n"))
    };

    format!(
        "{core_block}\n\n## Active Interests\n{interests}{episodic_section}{sends_section}{checks_section}\n\n\
You are in heartbeat mode. Follow these steps in order:\n\
1. Review '## Recently Sent to You' and '## Recently Checked' above. Any interest whose topic appears in either list was already covered within the last 6 hours — SKIP delegate_cli for it entirely. Do not re-research it.\n\
2. For interests NOT recently covered or checked, use delegate_cli to check for new or noteworthy developments.\n\
3. If your curiosity queue has items, pick ONE and investigate it with delegate_cli. If fully resolved, remove it with update_core_memory(\"curiosity_queue\", ...).\n\
4. After completing any series of searches or tool calls, call reflect to record what you found and decide whether to continue. Only call reply_to_user if you found something genuinely new or useful. Default to nothing — silence is the right choice when nothing significant has changed.",
        core_block = core.to_prompt_block(),
        interests = core.interests_block(),
        episodic_section = episodic_section,
        sends_section = sends_section,
        checks_section = checks_section,
    )
}

/// Shared agentic loop used by heartbeat and interest checks.
/// Loops up to `max_iters`, dispatching tool calls and accumulating context.
/// Returns the text that was sent to the user, if any.
async fn run_agentic_loop(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    mut messages: Vec<LmMessage>,
    tool_defs: serde_json::Value,
    max_iters: usize,
) -> anyhow::Result<Option<String>> {
    // If a background model is configured, swap it in for CLI calls.
    let call_config: Arc<Config> = if config.background_copilot_model.is_some() {
        Arc::new(Config {
            copilot_model: config.background_copilot_model.clone(),
            ..(*config).clone()
        })
    } else {
        config.clone()
    };

    let mut seen_reflections: std::collections::HashSet<String> = std::collections::HashSet::new();

    for iter in 0..max_iters {
        tracing::info!("Agentic loop iter {}: awaiting LM...", iter + 1);
        let lm_start = std::time::Instant::now();
        let response = post_to_lm_studio(&config, &messages, Some(&tool_defs), 4096).await?;
        let lm_elapsed = lm_start.elapsed();
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
                bot.send_message(ChatId(config.allowed_user_id as i64), text.clone()).await?;
                return Ok(Some(text));
            }
            return Ok(None);
        }

        let call = choice.message.tool_calls.into_iter().next().expect("checked non-empty");
        let single_call_json = serde_json::to_value(std::slice::from_ref(&call)).ok();
        let tool_name = call.function.name.clone();
        let call_id = call.id.clone();
        let raw_args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap_or_default();

        // Exact-duplicate backstop: if the model makes the identical reflect call twice, exit.
        if tool_name == "reflect" {
            if !seen_reflections.insert(call.function.arguments.clone()) {
                tracing::warn!("Agentic loop early exit: duplicate reflect call at iter {}", iter + 1);
                return Ok(None);
            }
        }

        tracing::info!("Agentic loop iter {}: tool={} [LM: {:.1}s]", iter + 1, tool_name, lm_elapsed.as_secs_f32());

        let dispatch_start = std::time::Instant::now();
        let result = tools::dispatch_tool_call(call_config.clone(), bot.clone(), scheduler.clone(), call).await?;
        let dispatch_elapsed = dispatch_start.elapsed();
        tracing::info!("Agentic loop iter {}: tool={} done [tool: {:.1}s]", iter + 1, tool_name, dispatch_elapsed.as_secs_f32());

        if tool_name == "reply_to_user" {
            bot.send_message(ChatId(config.allowed_user_id as i64), result.clone()).await?;
            return Ok(Some(result));
        }
        if tool_name == "nothing" {
            return Ok(None);
        }
        if tool_name == "reflect" {
            if raw_args["done"].as_bool().unwrap_or(false) {
                tracing::info!("Agentic loop: reflect concluded at iter {}", iter + 1);
                return Ok(None);
            }
            // done=false: observation goes into context below, loop continues
        }

        tracing::debug!("Agentic loop iter {} result: {:.500}", iter + 1, result);

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

    Ok(None)
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
                            "enum": ["beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to update. user_profile for facts about the user; beliefs for your evolving worldview; curiosity_queue for topics you want to investigate."
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

    let curiosity_display = if current.curiosity_queue.is_empty() {
        "(none yet)".to_string()
    } else {
        current.curiosity_queue.iter().map(|c| format!("- {c}")).collect::<Vec<_>>().join("\n")
    };

    let system_msg = format!(
        "You are maintaining persistent memory for AnieBot, a personal AI assistant.\n\n\
         ## Current memory\n\
         **User profile:** {user_profile}\n\
         **Beliefs:** {beliefs}\n\
         **Curiosity queue:** {curiosity}\n\n\
         ## Your task\n\
         Review the exchange below. Did it reveal anything NEW not already captured above?\n\n\
         Update user_profile if the exchange contained ANY of these:\n\
         - User's name, location, job, field, or life stage\n\
         - User's achievements, projects, relationships, or preferences\n\
         - Anything that defines who the user is or what matters to them\n\n\
         Update beliefs if you reached a conclusion or formed an opinion during this exchange — \
         about the world, technology, society, or this user's situation. Be direct: record the belief \
         as a concise first-person statement (e.g. \"Async Rust is worth the complexity for long-lived services\").\n\n\
         Update curiosity_queue if something came up that you genuinely want to investigate or learn more about — \
         an unresolved question, an interesting topic, something the user mentioned that you'd like to follow up on. \
         Add the full existing list plus the new item as a JSON array.\n\n\
         Call `nothing` only for pure small talk, acknowledgements, or follow-ups \
         that add no new facts (e.g. \"thanks\", \"got it\", \"that makes sense\").\n\n\
         When updating user_profile, write a concise prose summary that PRESERVES \
         existing facts and adds the new ones — do not discard what is already known.",
        user_profile = current.user_profile,
        beliefs = beliefs_display,
        curiosity = curiosity_display,
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
