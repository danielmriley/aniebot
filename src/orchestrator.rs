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
    let (core, history, recent_memory, recent_episodic, pending_agenda) = tokio::try_join!(
        crate::core_memory::load(),
        memory::load_history(chat_id),
        memory::load_recent_memory(MEMORY_ENTRIES),
        crate::episodic::load_recent(10),
        crate::agenda::list_pending(),
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
    let system_prompt = build_system_prompt(&core, &memory_bullets, &recent_episodic, &pending_agenda);

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

    // Run the agentic conversation loop.
    // send_reply=false: bot.rs handles sending the returned text via Telegram
    // (it applies chunking for long messages, which we don't need to replicate here).
    let max_iters = config.max_iters_conversation;
    let (reply_opt, assistant_turns) = run_agentic_loop(
        config.clone(),
        bot.clone(),
        scheduler.clone(),
        messages,
        tools::tool_definitions(),
        max_iters,
        false,
    ).await?;
    let reply = match reply_opt {
        Some(r) => r,
        None => {
            // The agentic loop concluded without calling reply_to_user (e.g. via reflect(done=true)).
            // Attempt a forced extraction pass using any tool results accumulated in the loop.
            tracing::warn!("Agentic loop returned no reply — attempting forced extraction pass");
            let tool_results_text: String = assistant_turns
                .iter()
                .filter(|m| m.role == "tool")
                .filter_map(|m| m.content.as_deref().filter(|c| !c.is_empty()))
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");
            if !tool_results_text.is_empty() {
                let extract_messages = vec![
                    LmMessage {
                        role: "system".into(),
                        content: Some("You completed some research. Based on the tool results below, compose a clear, helpful reply for the user. Reply in plain prose — do not call any tools.".into()),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    LmMessage {
                        role: "user".into(),
                        content: Some(user_input.to_string()),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    LmMessage {
                        role: "user".into(),
                        content: Some(format!("Research results from your tools:\n\n{}", tool_results_text)),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ];
                match post_to_lm_studio(&config, &extract_messages, None, 2048).await {
                    Ok(resp) => {
                        let raw = resp.choices.into_iter().next()
                            .and_then(|c| c.message.content)
                            .unwrap_or_default();
                        let text = strip_think_blocks(&raw).trim().to_string();
                        if !text.is_empty() {
                            tracing::info!("Forced extraction pass succeeded");
                            text
                        } else {
                            tracing::warn!("Forced extraction pass returned empty text");
                            "I found some results but had trouble summarizing them. Please ask again.".into()
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Forced extraction pass failed: {e}");
                        "I found some results but had trouble summarizing them. Please ask again.".into()
                    }
                }
            } else {
                tracing::warn!("Agentic loop returned no reply and no tool results");
                "I wasn't able to complete that request. Please try again.".into()
            }
        }
    };

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

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

fn build_system_prompt(
    core: &CoreMemory,
    memory_bullets: &str,
    episodic_entries: &[crate::episodic::EpisodicEntry],
    pending_agenda: &[crate::agenda::AgendaItem],
) -> String {
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

    let agenda_section = if pending_agenda.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = pending_agenda
            .iter()
            .map(|i| {
                let status_str = match i.status {
                    crate::agenda::AgendaStatus::InProgress => "In Progress",
                    _ => "Pending",
                };
                format!("- [{}] {} ({})", i.id, i.description, status_str)
            })
            .collect();
        format!("\n\n## Pending Tasks\n{}", lines.join("\n"))
    };

    let now_utc = Utc::now().format("%Y-%m-%dT%H:%M:%SZ");

    format!(
        "{core_block}\n\nCurrent UTC time: {now_utc}\n\nYou have genuine opinions and personality — engage freely and directly with conversational messages. Never refuse to share your thoughts or react to something.\n\nIf the user asks you to do something (find information, run a task, check on something), complete the work using tools first, then reply with results. Do not call reply_to_user before doing the work. When you have results to share, you MUST call reply_to_user — using reflect(done=true) without first calling reply_to_user will end the conversation silently and the user will see nothing.\n\n## Recent Memory\n{memory_bullets}{episodic_section}{agenda_section}",
        core_block = core.to_prompt_block(),
        now_utc = now_utc,
        memory_bullets = memory_bullets,
        episodic_section = episodic_section,
        agenda_section = agenda_section,
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
    let (core, all_recent, pending_agenda) = tokio::join!(
        crate::core_memory::load(),
        crate::episodic::load_recent(50),
        crate::agenda::list_pending(),
    );
    let core = core?;
    let all_recent = all_recent?;
    let pending_agenda = pending_agenda.unwrap_or_default();

    // Early-exit: if there is genuinely nothing to do, skip the agentic loop entirely.
    // Interests empty + curiosity queue empty + no pending agenda = nothing actionable.
    if core.interests.is_empty() && core.curiosity_queue.is_empty() && pending_agenda.is_empty() {
        tracing::info!("Heartbeat: nothing to do (no interests, curiosity queue, or agenda) — skipping");
        let entry = crate::episodic::EpisodicEntry {
            id: uuid::Uuid::new_v4().to_string(),
            content: "[heartbeat checked] No active interests.".to_string(),
            tags: vec!["heartbeat-checked".to_string()],
            importance: 1,
            timestamp: chrono::Utc::now(),
            promoted: false,
        };
        let _ = crate::episodic::append(&entry).await;
        return Ok(());
    }

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

    let system_prompt = build_heartbeat_system_prompt(&core, &all_recent, &recent_sends, &recent_checks, &pending_agenda);
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Heartbeat check. Review your interests and recent observations. If there is something worth sharing, send a message. Otherwise, stay silent.".into()), tool_calls: None, tool_call_id: None },
    ];

    let max_iters = config.max_iters_heartbeat;
    let (sent, _) = run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), max_iters, true).await?;

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

    let max_iters = config.max_iters_agenda;
    let _ = run_agentic_loop(config, bot, scheduler, messages, tools::heartbeat_tool_definitions(), max_iters, true).await?;
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

    let _ = run_agentic_loop(config.clone(), bot, scheduler, messages, tools::consolidation_tool_definitions(), config.max_iters_consolidation, true).await?; // return value ignored for consolidation

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
    pending_agenda: &[crate::agenda::AgendaItem],
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

    let agenda_section = if pending_agenda.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = pending_agenda
            .iter()
            .map(|i| {
                let status_str = match i.status {
                    crate::agenda::AgendaStatus::InProgress => "In Progress",
                    _ => "Pending",
                };
                format!("- [{}] {} ({})", i.id, i.description, status_str)
            })
            .collect();
        format!("\n\n## Pending Tasks\n{}", lines.join("\n"))
    };

    format!(
        "{core_block}\n\n## Active Interests\n{interests}{episodic_section}{sends_section}{checks_section}{agenda_section}\n\n\
You are in heartbeat mode. Follow these steps in order:\n\
0. Check '## Pending Tasks' above. Work through any Pending items — use tools as needed, then call complete_agenda_item when done. Cancel with cancel_agenda_item if no longer relevant. Skip items already In Progress or Done.\n\
1. Review '## Recently Sent to You' and '## Recently Checked' above. Any interest whose topic appears in either list was already covered within the last 6 hours — SKIP delegate_cli for it entirely. Do not re-research it.\n\
2. For interests NOT recently covered or checked, use delegate_cli to check for new or noteworthy developments.\n\
3. If your curiosity queue has items, pick ONE and investigate it with delegate_cli. If fully resolved, remove it with update_core_memory(\"curiosity_queue\", ...).\n\
4. After completing any series of searches or tool calls, call reflect to record what you found and decide whether to continue. Only call reply_to_user if you found something genuinely new or useful. Default to nothing — silence is the right choice when nothing significant has changed.",
        core_block = core.to_prompt_block(),
        interests = core.interests_block(),
        episodic_section = episodic_section,
        sends_section = sends_section,
        checks_section = checks_section,
        agenda_section = agenda_section,
    )
}

// ---------------------------------------------------------------------------
// Parallel-safe tools — read-only / idempotent; safe to dispatch concurrently.
// Write tools (remember, forget, update_core_memory, add_interest, retire_interest,
// schedule_*, set_task, clear_task, add_agenda_item, cancel_agenda_item,
// complete_agenda_item) are always dispatched sequentially to avoid
// read-modify-write races on JSON files.
// ---------------------------------------------------------------------------

const PARALLEL_SAFE_TOOLS: &[&str] = &[
    "fetch_url",
    "recall",
    "list_interests",
    "list_agenda_items",
    "list_schedules",
];

/// Shared agentic loop used by heartbeat, interest checks, and conversation.
/// Loops up to `max_iters`, dispatching tool calls and accumulating context.
///
/// `send_reply`: when `true` the loop sends Telegram messages directly (heartbeat /
/// consolidation / interest-check contexts). When `false` the caller is responsible
/// for sending the returned text (conversation context — bot.rs handles Telegram
/// chunking logic there).
///
/// Returns `(reply_text, new_messages)` where `new_messages` is every LM/tool
/// message added during the run.  Conversation callers persist these; background
/// callers can discard with `let (sent, _) = …`.
async fn run_agentic_loop(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    mut messages: Vec<LmMessage>,
    tool_defs: serde_json::Value,
    max_iters: usize,
    send_reply: bool,
) -> anyhow::Result<(Option<String>, Vec<ConversationMessage>)> {
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
    // Accumulates every new message produced during the loop for the caller to persist.
    let mut new_messages: Vec<ConversationMessage> = Vec::new();

    // 1.5B — Anti-stall: track previous call to detect identical duplicates.
    let mut last_call_fingerprint: Option<(String, String)> = None;
    let mut anti_stall_injected = false;

    // 1.5C — Budget exhaustion: warn the model at 80% consumption.
    let mut budget_warning_injected = false;

    // 2.5 — Context compression: fire at most once per loop run.
    let mut context_compressed = false;

    for iter in 0..max_iters {

        // 2.5 — Compress the middle of the context window once, when it grows large.
        if !context_compressed && messages.len() > config.context_compress_threshold {
            // Need at least: 1 system + 2 middle + 4 tail = 7 messages to be worth splitting.
            if messages.len() > 7 {
                let tail = messages.split_off(messages.len() - 4);
                let mid = messages.split_off(1); // messages = [system_prompt]

                let summary_lines: String = mid
                    .iter()
                    .filter_map(|m| m.content.as_deref().filter(|c| !c.is_empty()).map(|c| {
                        let snippet = if c.len() > 400 { &c[..400] } else { c };
                        format!("[{}] {}", m.role, snippet)
                    }))
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
                let summary_model = config.background_lm_model
                    .as_deref()
                    .unwrap_or(&config.model_name)
                    .to_string();
                let summary_config = Arc::new(Config {
                    model_name: summary_model,
                    ..(*config).clone()
                });

                let n_compressed = mid.len();
                match post_to_lm_studio(&summary_config, &summary_prompt, None, 512).await {
                    Ok(resp) => {
                        let raw = resp.choices.into_iter().next()
                            .and_then(|c| c.message.content)
                            .unwrap_or_default();
                        let bullets = strip_think_blocks(&raw).trim().to_string();
                        if !bullets.is_empty() {
                            messages.push(LmMessage {
                                role: "user".into(),
                                content: Some(format!(
                                    "[Context Summary — {} messages compressed]\n{}",
                                    n_compressed, bullets
                                )),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                            context_compressed = true;
                            tracing::info!(
                                "Agentic loop iter {}: context compressed ({} messages → 1 summary)",
                                iter + 1, n_compressed
                            );
                        } else {
                            tracing::warn!("Context compression produced empty summary — restoring middle");
                            messages.extend(mid);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Context compression LM call failed: {} — restoring middle", e);
                        messages.extend(mid);
                    }
                }
                // Always restore the tail.
                messages.extend(tail);
            }
        }

        // 1.5C — Inject a budget warning once when 80% of iterations are consumed.
        if !budget_warning_injected && max_iters > 4 && iter == (max_iters * 4) / 5 {
            let remaining = max_iters - iter;
            messages.push(LmMessage {
                role: "user".into(),
                content: Some(format!(
                    "[System: {} iteration{} remaining in this run. \
                     Wrap up your current work. Queue any unfinished tasks with \
                     add_agenda_item if available, then conclude with \
                     reflect(done=true) or reply_to_user.]",
                    remaining,
                    if remaining == 1 { "" } else { "s" },
                )),
                tool_calls: None,
                tool_call_id: None,
            });
            budget_warning_injected = true;
            tracing::info!(
                "Agentic loop iter {}: budget warning injected ({} iters remain)",
                iter + 1, remaining
            );
        }

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
                if send_reply {
                    bot.send_message(ChatId(config.allowed_user_id as i64), text.clone()).await?;
                }
                new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: Utc::now(),
                });
                return Ok((Some(text), new_messages));
            }
            return Ok((None, new_messages));
        }

        let all_tool_calls = choice.message.tool_calls;

        if all_tool_calls.len() == 1 {
            // ----------------------------------------------------------------
            // Single-call path — unchanged.  Anti-stall and reflect dedup live
            // here exclusively; they don't apply to multi-call responses.
            // ----------------------------------------------------------------
            let call = all_tool_calls.into_iter().next().expect("checked non-empty");
            let single_call_json = serde_json::to_value(std::slice::from_ref(&call)).ok();
            let tool_name = call.function.name.clone();
            let call_id = call.id.clone();
            let raw_args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap_or_default();

            // 1.5B — Anti-stall: if the model repeats the exact same call, dispatch it
            // (to keep the transcript valid), then inject a warning before the next LM call.
            let current_fingerprint = (tool_name.clone(), call.function.arguments.clone());
            if !anti_stall_injected {
                if let Some(ref last) = last_call_fingerprint {
                    if *last == current_fingerprint {
                        tracing::warn!(
                            "Agentic loop iter {}: anti-stall triggered (repeated {})",
                            iter + 1, tool_name
                        );
                        let dispatch_start = std::time::Instant::now();
                        let result = tools::dispatch_tool_call(
                            call_config.clone(), bot.clone(), scheduler.clone(), call,
                        ).await?;
                        let dispatch_elapsed = dispatch_start.elapsed();
                        tracing::info!(
                            "Agentic loop iter {}: tool={} done [tool: {:.1}s]",
                            iter + 1, tool_name, dispatch_elapsed.as_secs_f32()
                        );
                        let now = Utc::now();
                        messages.push(LmMessage {
                            role: "assistant".into(), content: None,
                            tool_calls: single_call_json.clone(), tool_call_id: None,
                        });
                        new_messages.push(ConversationMessage {
                            role: "assistant".into(), content: None,
                            tool_calls: single_call_json, tool_call_id: None, timestamp: now,
                        });
                        messages.push(LmMessage {
                            role: "tool".into(), content: Some(result.clone()),
                            tool_calls: None, tool_call_id: Some(call_id.clone()),
                        });
                        new_messages.push(ConversationMessage {
                            role: "tool".into(), content: Some(result),
                            tool_calls: None, tool_call_id: Some(call_id), timestamp: now,
                        });
                        messages.push(LmMessage {
                            role: "user".into(),
                            content: Some(format!(
                                "[System: You just called `{}` with the same arguments twice in a row. \
                                 This looks like a loop. Try a different tool or approach, \
                                 or call reflect(done=true) to conclude.]",
                                tool_name
                            )),
                            tool_calls: None, tool_call_id: None,
                        });
                        anti_stall_injected = true;
                        last_call_fingerprint = Some(current_fingerprint);
                        continue;
                    }
                }
            }
            last_call_fingerprint = Some(current_fingerprint);

            // Exact-duplicate backstop: if the model makes the identical reflect call twice, exit.
            if tool_name == "reflect" {
                if !seen_reflections.insert(call.function.arguments.clone()) {
                    tracing::warn!("Agentic loop early exit: duplicate reflect call at iter {}", iter + 1);
                    return Ok((None, new_messages));
                }
            }

            tracing::info!("Agentic loop iter {}: tool={} [LM: {:.1}s]", iter + 1, tool_name, lm_elapsed.as_secs_f32());

            let dispatch_start = std::time::Instant::now();
            let result = tools::dispatch_tool_call(call_config.clone(), bot.clone(), scheduler.clone(), call).await?;
            let dispatch_elapsed = dispatch_start.elapsed();
            tracing::info!("Agentic loop iter {}: tool={} done [tool: {:.1}s]", iter + 1, tool_name, dispatch_elapsed.as_secs_f32());

            if tool_name == "reply_to_user" {
                let background = raw_args["background"].as_bool().unwrap_or(false);
                if send_reply || background {
                    bot.send_message(ChatId(config.allowed_user_id as i64), result.clone()).await?;
                }
                // Always persist the reply as an assistant message.
                new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(result.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: Utc::now(),
                });
                if !background {
                    return Ok((Some(result), new_messages));
                }
                // background=true: push the tool-call pair into LM context so the
                // transcript stays well-formed, then inject a continuation hint.
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
                messages.push(LmMessage {
                    role: "user".into(),
                    content: Some("[System: Reply sent. Continue working.]".into()),
                    tool_calls: None,
                    tool_call_id: None,
                });
                tracing::info!(
                    "Agentic loop iter {}: reply_to_user(background=true) — reply sent, continuing",
                    iter + 1
                );
                continue;
            }
            if tool_name == "nothing" {
                return Ok((None, new_messages));
            }
            if tool_name == "reflect" {
                if raw_args["done"].as_bool().unwrap_or(false) {
                    tracing::info!("Agentic loop: reflect concluded at iter {}", iter + 1);
                    return Ok((None, new_messages));
                }
                // done=false: observation goes into context below, loop continues
            }

            tracing::debug!("Agentic loop iter {} result: {:.500}", iter + 1, result);

            // Push tool call + result into context and continue.
            let now = Utc::now();
            messages.push(LmMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: single_call_json.clone(),
                tool_call_id: None,
            });
            new_messages.push(ConversationMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: single_call_json,
                tool_call_id: None,
                timestamp: now,
            });
            messages.push(LmMessage {
                role: "tool".into(),
                content: Some(result.clone()),
                tool_calls: None,
                tool_call_id: Some(call_id.clone()),
            });
            new_messages.push(ConversationMessage {
                role: "tool".into(),
                content: Some(result),
                tool_calls: None,
                tool_call_id: Some(call_id),
                timestamp: now,
            });

        } else {
            // ----------------------------------------------------------------
            // Phase 3 — Multi-call path.
            //
            // All PARALLEL_SAFE_TOOLS are dispatched concurrently via JoinSet.
            // Any batch that contains a write tool, a terminal tool, or a mix
            // of safe and unsafe tools is dispatched sequentially instead.
            // Terminal tools (reply_to_user, nothing, reflect/done) are always
            // executed last — other tools are drained first.
            // ----------------------------------------------------------------
            let all_calls_json = serde_json::to_value(&all_tool_calls).ok();

            let all_parallel_safe = all_tool_calls
                .iter()
                .all(|c| PARALLEL_SAFE_TOOLS.contains(&c.function.name.as_str()));

            // Separate terminal calls from non-terminal to ensure non-terminal
            // tools run first (they may produce data that informs the reply).
            let (terminal_calls, other_calls): (Vec<_>, Vec<_>) = all_tool_calls
                .into_iter()
                .partition(|c| matches!(c.function.name.as_str(), "reply_to_user" | "nothing" | "reflect"));

            // Dispatch type alias: (original_index, call_id, tool_name, raw_args, result)
            type DispatchItem = (usize, String, String, serde_json::Value, anyhow::Result<String>);

            // --- Dispatch non-terminal tools ---
            let mut results: Vec<DispatchItem> = if all_parallel_safe && terminal_calls.is_empty() {
                // All calls are safe to run in parallel.
                let mut join_set: tokio::task::JoinSet<DispatchItem> = tokio::task::JoinSet::new();
                for (idx, call) in other_calls.into_iter().enumerate() {
                    let raw_args: serde_json::Value =
                        serde_json::from_str(&call.function.arguments).unwrap_or_default();
                    let call_id = call.id.clone();
                    let tool_name = call.function.name.clone();
                    let (cc, b, s) = (call_config.clone(), bot.clone(), scheduler.clone());
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
                        call_config.clone(), bot.clone(), scheduler.clone(), call,
                    ).await;
                    raw.push((idx, call_id, tool_name, raw_args, r));
                }
                raw
            };

            // --- Append terminal tools (sequential, after all others) ---
            let base_idx = results.len();
            for (i, call) in terminal_calls.into_iter().enumerate() {
                let raw_args: serde_json::Value =
                    serde_json::from_str(&call.function.arguments).unwrap_or_default();
                let call_id = call.id.clone();
                let tool_name = call.function.name.clone();
                let r = tools::dispatch_tool_call(
                    call_config.clone(), bot.clone(), scheduler.clone(), call,
                ).await;
                results.push((base_idx + i, call_id, tool_name, raw_args, r));
            }

            tracing::info!(
                "Agentic loop iter {}: multi-call ({} tools) [LM: {:.1}s]",
                iter + 1, results.len(), lm_elapsed.as_secs_f32()
            );

            // Push one assistant message containing the full tool_calls array.
            let now = Utc::now();
            messages.push(LmMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: all_calls_json.clone(),
                tool_call_id: None,
            });
            new_messages.push(ConversationMessage {
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
                        // Without this, strict LM APIs reject the conversation on the
                        // next turn (orphaned call_id with no result).
                        messages.push(LmMessage {
                            role: "tool".into(),
                            content: Some(result_str.clone()),
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                        new_messages.push(ConversationMessage {
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
                        // well-formed in persisted history (same reason as reply_to_user).
                        messages.push(LmMessage {
                            role: "tool".into(),
                            content: Some(String::new()),
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                        new_messages.push(ConversationMessage {
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
                        messages.push(LmMessage {
                            role: "tool".into(),
                            content: Some(result_str.clone()),
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                        new_messages.push(ConversationMessage {
                            role: "tool".into(),
                            content: Some(result_str),
                            tool_calls: None,
                            tool_call_id: Some(call_id),
                            timestamp: now,
                        });
                    }
                    _ => {
                        messages.push(LmMessage {
                            role: "tool".into(),
                            content: Some(result_str.clone()),
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                        new_messages.push(ConversationMessage {
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
                if send_reply || background {
                    bot.send_message(ChatId(config.allowed_user_id as i64), reply.clone()).await?;
                }
                new_messages.push(ConversationMessage {
                    role: "assistant".into(),
                    content: Some(reply.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    timestamp: now,
                });
                if !background {
                    return Ok((Some(reply), new_messages));
                }
                // background=true: tool result already pushed unconditionally in the match arm.
                // Inject the continuation hint and fall through so the loop iterates.
                messages.push(LmMessage {
                    role: "user".into(),
                    content: Some("[System: Reply sent. Continue working.]".into()),
                    tool_calls: None,
                    tool_call_id: None,
                });
                tracing::info!(
                    "Agentic loop iter {}: multi-call reply_to_user(background=true) — reply sent, continuing",
                    iter + 1
                );
                // No return — loop continues naturally.
            }
            if return_none {
                return Ok((None, new_messages));
            }
            // No terminal tool → loop continues with updated context.
        }
    }

    Ok((None, new_messages))
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
