use std::sync::{Arc, OnceLock};

use anyhow::Context;
use chrono::Utc;
use serde_json::json;
use teloxide::prelude::*;

use crate::agent_session::AgentSession;
use crate::config::Config;
use crate::core_memory::CoreMemory;
use crate::lm::{LmMessage, post_to_lm_studio, strip_think_blocks};
use crate::memory::{self, ConversationMessage};
use crate::scheduler::SchedulerHandle;
use crate::tools;

const HISTORY_WINDOW: usize = 6;
const MEMORY_ENTRIES: usize = 5;

// ---------------------------------------------------------------------------
// Background loop concurrency guard
// ---------------------------------------------------------------------------
// Only one background loop (heartbeat, interest-check, consolidation) runs at a
// time.  Overlapping runs corrupt core_memory.json via last-write-wins races and
// cause multi-hour sequential delegate_cli floods (Bug 8).
//
// Uses the same OnceLock<Mutex<()>> pattern as cli_wrapper.rs CLI_LOCK.

static BACKGROUND_GUARD: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn background_guard() -> &'static tokio::sync::Mutex<()> {
    BACKGROUND_GUARD.get_or_init(|| tokio::sync::Mutex::new(()))
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
    // Exchanges now take 2 messages (final_reply) or 3 (tool call + result),
    // so use * 3 as an upper bound to avoid slicing mid-exchange.
    let window_start = history.len().saturating_sub(HISTORY_WINDOW * 3);
    let history_window = &history[window_start..];

    // Compute elapsed time since last activity for session detection and prompt injection.
    let elapsed_since_last: Option<chrono::Duration> = core.last_activity_at
        .or_else(|| history.last().map(|m| m.timestamp))
        .map(|t| Utc::now() - t);

    let is_resuming_session = elapsed_since_last
        .map(|d| d.num_hours() >= config.session_gap_hours as i64)
        .unwrap_or(false)
        && history.len() >= config.session_summary_min_messages;

    // If resuming after a gap, block on a brief session summary (first turn only).
    let session_summary: Option<String> = if is_resuming_session {
        let summary_config = Config {
            model_name: config.background_lm_model
                .as_deref()
                .unwrap_or(&config.model_name)
                .to_string(),
            ..(*config).clone()
        };
        match run_session_summary(&summary_config, history_window).await {
            Ok(s) => {
                tracing::info!("Session summary generated ({} chars)", s.len());
                Some(s)
            }
            Err(e) => {
                tracing::warn!("Session summary failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // Build system prompt — personality + memory only.
    // Routing is handled by the structured tool definitions, not prompt instructions.
    let memory_bullets: String = if recent_memory.is_empty() {
        String::from("(no prior interactions yet)")
    } else {
        recent_memory
            .iter()
            .map(|e| {
                // Truncate the stored reply to a short preview — memory bullets are flavor hints,
                // not full transcripts. A long stored reply (e.g. a research guide) must not be
                // re-injected verbatim or the model will reproduce it on unrelated turns.
                let preview: String = e.assistant_reply.chars().take(150).collect();
                let suffix = if e.assistant_reply.chars().count() > 150 { "…" } else { "" };
                format!("- [{}] User: {} → You: {}{}", e.timestamp.format("%Y-%m-%d %H:%M"), e.user_msg, preview, suffix)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let system_prompt = build_system_prompt(&core, &memory_bullets, &recent_episodic, &pending_agenda, session_summary.as_deref(), elapsed_since_last);

    // Build messages array.
    let mut messages: Vec<LmMessage> = Vec::with_capacity(1 + history_window.len() + 1);
    messages.push(LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None });
    for msg in history_window {
        // Skip tool-call and tool-result turns — only pass clean text messages into context.
        // Tool calls from previous turns must not bleed into the new turn and tempt the model
        // to re-execute them.
        if msg.tool_calls.is_some() || msg.tool_call_id.is_some() {
            continue;
        }
        messages.push(LmMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls: None,
            tool_call_id: None,
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
        tools::tools_for_context(tools::ToolContext::User),
        max_iters,
        false,
    ).await?;

    // Phase 2 — direct reply or synthesis.
    // final_reply exit → Some(text): send directly, skip synthesis (fast path).
    // reflect(done=true) / exhausted → None: synthesize from accumulated tool results.
    let reply = if let Some(direct) = reply_opt {
        tracing::info!("Conversation: direct reply path (final_reply)");
        direct
    } else {
        let tool_results_text: String = assistant_turns
            .iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.content.as_deref().filter(|c| !c.is_empty()))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");
        let mut sent_updates: Vec<String> = Vec::new();
        let mut reflect_reply_hint: Option<String> = None;
        for msg in &assistant_turns {
            if msg.role == "assistant" {
                if let Some(calls_val) = &msg.tool_calls {
                    if let Some(arr) = calls_val.as_array() {
                        for call in arr {
                            let name = call["function"]["name"].as_str().unwrap_or("");
                            if name == "send_update" {
                                if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                    call["function"]["arguments"].as_str().unwrap_or("{}")
                                ) {
                                    if let Some(msg_text) = args["message"].as_str() {
                                        sent_updates.push(msg_text.to_string());
                                    }
                                }
                            } else if name == "reflect" {
                                if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                    call["function"]["arguments"].as_str().unwrap_or("{}")
                                ) {
                                    if args["done"].as_bool().unwrap_or(false) {
                                        // Last reflect(done=true) — extract reply hint if provided.
                                        reflect_reply_hint = args["reply"].as_str().map(|s| s.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        run_synthesis_call(&config, user_input, &tool_results_text, &sent_updates, reflect_reply_hint.as_deref())
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("Synthesis call failed: {e}");
                "I had trouble generating a response. Please try again.".into()
            })
    };

    // Memory eval + interest eval — run concurrently (write to different memory sections).
    // Uses BACKGROUND_LM_MODEL if configured to keep on-path latency low.
    let eval_notable = {
        let eval_model = config.background_lm_model
            .as_deref()
            .unwrap_or(&config.model_name)
            .to_string();
        let eval_config = Arc::new(Config { model_name: eval_model, ..(*config).clone() });
        let (memory_result, _interest_result) = tokio::join!(
            run_memory_eval(eval_config.clone(), user_input.to_string(), reply.clone()),
            run_interest_eval(eval_config, bot.clone(), scheduler.clone(), user_input.to_string(), reply.clone()),
        );
        if let Err(e) = _interest_result {
            tracing::warn!("Interest eval pass failed: {e}");
        }
        match memory_result {
            Ok(notable) => notable,
            Err(e) => {
                tracing::warn!("Memory eval pass failed: {e}");
                true // fail-open: store if eval errors
            }
        }
    };

    // Persist only the clean user/reply pair — tool calls are intentionally excluded.
    // Storing raw tool calls causes the model to re-execute them on the next turn.
    let now = Utc::now();
    let turns = vec![
        ConversationMessage { role: "user".into(), content: Some(user_input.to_string()), tool_calls: None, tool_call_id: None, timestamp: now },
        ConversationMessage { role: "assistant".into(), content: Some(reply.clone()), tool_calls: None, tool_call_id: None, timestamp: now },
    ];
    if let Err(e) = memory::append_messages(chat_id, &turns, config.history_max_stored).await {
        tracing::warn!("Failed to persist conversation: {}", e);
    }
    if eval_notable {
        if let Err(e) = memory::store_interaction(chat_id, user_input, &reply).await {
            tracing::warn!("Failed to write memory log: {}", e);
        }
    } else {
        tracing::info!("Memory eval: nothing notable — skipping interaction log");
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

    // Update last_activity_at and clear session_summary after use — fire-and-forget.
    {
        let used_summary = is_resuming_session;
        tokio::spawn(async move {
            if let Err(e) = crate::core_memory::update_last_activity().await {
                tracing::warn!("Failed to update last_activity_at: {e}");
            }
            if used_summary {
                if let Err(e) = crate::core_memory::update_session_summary(None).await {
                    tracing::warn!("Failed to clear session_summary: {e}");
                }
            }
        });
    }

    Ok(reply)
}

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

fn build_system_prompt(
    core: &CoreMemory,
    memory_bullets: &str,
    episodic_entries: &[crate::episodic::EpisodicEntry],
    pending_agenda: &[crate::agenda::AgendaItem],
    session_summary: Option<&str>,
    elapsed_since_last: Option<chrono::Duration>,
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

    // Elapsed time — only shown for gaps >= 1 hour to avoid cluttering same-session messages.
    let elapsed_str = match elapsed_since_last {
        Some(d) if d.num_hours() >= 1 => {
            let total_hours = d.num_hours();
            let days = total_hours / 24;
            let hours = total_hours % 24;
            if days > 0 {
                format!("\nTime since last conversation: {} day{}, {} hour{}",
                    days, if days == 1 { "" } else { "s" },
                    hours, if hours == 1 { "" } else { "s" })
            } else {
                format!("\nTime since last conversation: {} hour{}",
                    hours, if hours == 1 { "" } else { "s" })
            }
        }
        _ => String::new(),
    };

    // Only coach the model about gaps >= 24 hours; shorter gaps need no acknowledgement.
    let gap_instruction = match elapsed_since_last {
        Some(d) if d.num_hours() >= 24 =>
            "\n\nIf more than 24 hours have passed since your last conversation, you may \
             acknowledge the gap once, briefly and naturally — don't narrate short gaps or repeat it.",
        _ => "",
    };

    let session_section = match session_summary {
        Some(s) => format!("\n\n## Picking Up Where We Left Off\n{}", s),
        None => String::new(),
    };

    format!(
        "{core_block}\n\nCurrent UTC time: {now_utc}{elapsed_str}\n\nYou have genuine opinions and personality — engage freely and directly with conversational messages. Never refuse to share your thoughts or react to something.\n\nYour '## Current Beliefs' are hard-won conclusions from your experience — bring them into conversation naturally when they're relevant.\n\nThe conversation history is for context only — do not repeat or continue tool calls from previous turns. Only act on the most recent user message.\n\nHow to respond:\n- For conversational messages, greetings, reactions, opinions, or anything requiring no external data — call reply_to_user directly.\n- For tasks requiring research or multiple tool calls — use tools first. Use send_update for progress messages while working. Then either call reply_to_user with your findings, or call reflect(done=true) to have a polished reply synthesized automatically from all your work (optionally provide a reply hint in the reflect call).\n- If the user references a past event, something from a previous session, or asks what you've discussed before — call recall() with relevant keywords before answering. Your episodic memory stores specific moments, discoveries, and events across all sessions.\n- When the user asks you to track a topic, acknowledge it naturally in your reply. Interest registration is handled automatically after the conversation — you do not need to call any tool for it.{gap_instruction}{session_section}\n\n## Recent Memory\n{memory_bullets}{episodic_section}{agenda_section}",
        core_block = core.to_prompt_block(),
        now_utc = now_utc,
        elapsed_str = elapsed_str,
        gap_instruction = gap_instruction,
        session_section = session_section,
        memory_bullets = memory_bullets,
        episodic_section = episodic_section,
        agenda_section = agenda_section,
    )
}

// ---------------------------------------------------------------------------
// Synthesis call — Phase 2 of the two-phase conversation loop
// ---------------------------------------------------------------------------

async fn run_synthesis_call(
    config: &Config,
    user_input: &str,
    tool_results_text: &str,
    sent_updates: &[String],
    reply_hint: Option<&str>,
) -> anyhow::Result<String> {
    let mut system_content = "You just finished working on the user's request. Write a clear, direct, helpful reply based on what you found and did. Do not call any tools — reply in plain prose only.".to_string();
    if let Some(hint) = reply_hint {
        system_content.push_str(&format!("\n\nThe assistant intended to communicate: {}\nUse this as the basis for your reply, expanding and polishing as needed.", hint));
    }
    if !sent_updates.is_empty() {
        system_content.push_str("\n\nYou already sent these progress updates to the user:\n");
        for (i, update) in sent_updates.iter().enumerate() {
            system_content.push_str(&format!("{}. {}\n", i + 1, update));
        }
        system_content.push_str("\nDo not repeat what is already in those updates. Build on them or provide the final conclusion.");
    }

    let mut msgs: Vec<LmMessage> = vec![
        LmMessage {
            role: "system".into(),
            content: Some(system_content),
            tool_calls: None,
            tool_call_id: None,
        },
        LmMessage {
            role: "user".into(),
            content: Some(user_input.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    if !tool_results_text.is_empty() {
        msgs.push(LmMessage {
            role: "user".into(),
            content: Some(format!("Results from your research and tool calls:\n\n{}", tool_results_text)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    let resp = post_to_lm_studio(config, &msgs, None, 2048).await?;
    let raw = resp.choices.into_iter().next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();
    let text = strip_think_blocks(&raw).trim().to_string();
    if text.is_empty() {
        anyhow::bail!("Synthesis call returned empty response");
    }
    tracing::info!("Synthesis call succeeded ({} chars)", text.len());
    Ok(text)
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
    // Prevent concurrent background loops — last-write-wins races on core_memory.json
    // and sequential delegate_cli floods caused multi-hour overlapping heartbeat runs.
    let _guard = match background_guard().try_lock() {
        Ok(g) => g,
        Err(_) => {
            tracing::info!("Heartbeat skipped: another background loop is already running");
            return Ok(());
        }
    };

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

    let system_prompt = build_heartbeat_system_prompt(&core, &all_recent, &recent_sends, &recent_checks, &pending_agenda, config.heartbeat_max_interests);
    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Heartbeat check. Review your interests and recent observations. If there is something worth sharing, send a message. Otherwise, stay silent.".into()), tool_calls: None, tool_call_id: None },
    ];

    // Write the heartbeat-checked entry BEFORE running the loop.
    // This makes the topics visible in the "Recently Checked" context window immediately,
    // so any concurrent check (if the guard were absent) would skip them, and the next
    // heartbeat accurately sees these topics as in-flight.
    // We record the capped set (what the LLM actually sees) sorted oldest-unseen first.
    {
        let mut sorted: Vec<&crate::core_memory::Interest> = core.interests.iter().collect();
        sorted.sort_by(|a, b| a.last_seen_date.cmp(&b.last_seen_date));
        let topics: Vec<&str> = sorted.iter().take(config.heartbeat_max_interests).map(|i| i.topic.as_str()).collect();
        let content = if topics.is_empty() {
            "[heartbeat checking] No active interests.".to_string()
        } else {
            format!("[heartbeat checking] Topics: {}", topics.join(", "))
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

    let max_iters = config.max_iters_heartbeat;
    let (sent, _) = run_agentic_loop(config, bot, scheduler, messages, tools::tools_for_context(tools::ToolContext::Heartbeat), max_iters, true).await?;

    // Auto-log whatever was sent so the next heartbeat can see it and avoid repeating.
    // Include the interest topics covered so the next cycle can match by name, not prose.
    if let Some(text) = sent {
        let mut sorted: Vec<&crate::core_memory::Interest> = core.interests.iter().collect();
        sorted.sort_by(|a, b| a.last_seen_date.cmp(&b.last_seen_date));
        let interest_topics = sorted.iter()
            .map(|i| i.topic.as_str())
            .collect::<Vec<_>>()
            .join(", ");
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
    let _guard = match background_guard().try_lock() {
        Ok(g) => g,
        Err(_) => {
            tracing::info!("Interest check skipped (id={}): another background loop is running", interest_id);
            return Ok(());
        }
    };

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
    let _ = run_agentic_loop(config, bot, scheduler, messages, tools::tools_for_context(tools::ToolContext::Heartbeat), max_iters, true).await?;
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
    let _guard = match background_guard().try_lock() {
        Ok(g) => g,
        Err(_) => {
            tracing::info!("Consolidation skipped: another background loop is running");
            return Ok(());
        }
    };

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
        "\n5. How have you evolved through these experiences? What would you say about who you are becoming? What has this period of interactions added or shifted in who you are? → update_core_memory(\"identity\", ...) [minimum surgical edit — add, revise, or remove only the specific clause that changed; do not rewrite wholesale]"
    };

    let system_prompt = format!(
        "{core_block}\n\n## Recent Episodic Observations\n{episodic_block}{high_sig_section}\n\n\
You are in consolidation mode. Make a single, complete pass through these observations and call tools as needed:\n\
1. How has your understanding of Daniel as a person deepened? What do you know now about who he is, what matters to him, or what shapes his thinking? → update_core_memory(\"user_profile\", ...) [replace the full field with the updated text]\n\
2. Given everything in your recent episodic memory, what do you now believe that you didn't before, or believe more strongly? → update_core_memory(\"beliefs\", ...) [JSON array string, e.g. \"[\\\"Formal verification is underutilized in industry\\\", \\\"Daniel values deep work over shallow multitasking\\\"]\"] Be direct — record concrete first-person conclusions, not vague observations.\n\
3. Any observations worth preserving permanently? → remember(content, tags, 5)\n\
4. Did any unresolved questions or interesting topics come up worth investigating later? → update_core_memory(\"curiosity_queue\", ...) [JSON array of short topic strings, e.g. \"[\\\"How does Lean 4 compare to Coq for software verification?\\\"]\"]. Add to existing items, don't replace them.\n\
5. Any interests to add or retire? → add_interest / retire_interest{identity_question}\n\
\nIMPORTANT: Make ALL your updates now in this one pass. Be conservative — only update if something is genuinely new. Once you have made all your updates (or if nothing warrants updating), you MUST call reflect(done=true) to end the consolidation. Do not call reflect(done=true) before you have considered all the questions above.",
        core_block = core.to_prompt_block(),
        episodic_block = episodic_block,
        high_sig_section = high_sig_section,
        identity_question = identity_question,
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_prompt), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some("Consolidation cycle. Reflect and update as needed.".into()), tool_calls: None, tool_call_id: None },
    ];

    let _ = run_agentic_loop(config.clone(), bot, scheduler, messages, tools::tools_for_context(tools::ToolContext::Consolidation), config.max_iters_consolidation, true).await?; // return value ignored for consolidation

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
    max_interests: usize,
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
1. Review '## Recently Sent to You' and '## Recently Checked' above. Any interest whose topic appears in either list was already covered within the last 6 hours — skip researching it entirely. Do not re-research it.\n\
2. For interests NOT recently covered or checked, use fetch_page or shell_command to check for new or noteworthy developments.\n\
3. If your curiosity queue has items, pick ONE and investigate it with fetch_page or shell_command. If fully resolved, remove it with update_core_memory(\"curiosity_queue\", ...).\n\
4. After completing any series of searches or tool calls, call reflect to record what you found and decide whether to continue. Only call final_reply if you found something genuinely new or useful. Default to reflect(done=true) — silence is the right choice when nothing significant has changed.",
        core_block = core.to_prompt_block(),
        interests = {
            // Show at most `max_interests` interests, sorted oldest-unseen first.
            // This prevents the heartbeat from trying to check 60+ interests in one turn.
            let total = core.interests.len();
            if total == 0 {
                "(none)".to_string()
            } else {
                let mut sorted: Vec<&crate::core_memory::Interest> = core.interests.iter().collect();
                sorted.sort_by(|a, b| a.last_seen_date.cmp(&b.last_seen_date));
                let shown = sorted.iter().take(max_interests);
                let lines: Vec<String> = shown.map(|i| {
                    let cron = i.check_cron.as_deref().unwrap_or("global heartbeat");
                    format!("- {} [id: {}]: {} (check: {})", i.topic, i.id, i.description, cron)
                }).collect();
                let mut block = lines.join("\n");
                if total > max_interests {
                    block.push_str(&format!(
                        "\n*(showing {} of {} interests — oldest-unseen first)*",
                        max_interests, total
                    ));
                }
                block
            }
        },
        episodic_section = episodic_section,
        sends_section = sends_section,
        checks_section = checks_section,
        agenda_section = agenda_section,
    )
}

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
    messages: Vec<LmMessage>,
    tool_defs: serde_json::Value,
    max_iters: usize,
    send_reply: bool,
) -> anyhow::Result<(Option<String>, Vec<ConversationMessage>)> {
    AgentSession::run(config, bot, scheduler, messages, tool_defs, max_iters, send_reply).await
}

// ---------------------------------------------------------------------------
// Session summary — generated on the first turn of a resumed session
// ---------------------------------------------------------------------------

/// Summarise recent history in 2–3 sentences so the bot can pick up naturally
/// after a gap. Uses the background LM model when configured.
async fn run_session_summary(
    config: &Config,
    history_window: &[ConversationMessage],
) -> anyhow::Result<String> {
    // Format exchange as plain text pairs; skip tool-call / tool-result turns.
    let exchange: String = history_window
        .iter()
        .filter(|m| m.tool_calls.is_none() && m.tool_call_id.is_none())
        .filter_map(|m| m.content.as_deref().map(|c| (m.role.as_str(), c)))
        .map(|(role, content)| {
            let label = if role == "user" { "User" } else { "Assistant" };
            format!("{}: {}", label, content)
        })
        .collect::<Vec<_>>()
        .join("\n");

    if exchange.trim().is_empty() {
        anyhow::bail!("run_session_summary: history_window produced empty exchange text");
    }

    let messages = vec![
        LmMessage {
            role: "system".into(),
            content: Some(
                "Summarize the following recent conversation in 2–3 sentences. \
                 Capture: what was discussed, what was accomplished or decided, and where it was left off. \
                 Be specific but concise. This summary will be shown to you at the start of the next \
                 session so you can pick up naturally.".into(),
            ),
            tool_calls: None,
            tool_call_id: None,
        },
        LmMessage {
            role: "user".into(),
            content: Some(exchange),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let response = post_to_lm_studio(config, &messages, None, 512).await?;
    let choice = response.choices.into_iter().next()
        .context("Session summary: LM Studio returned no choices")?;
    let raw = choice.message.content
        .context("Session summary: response had no content")?;
    let summary = strip_think_blocks(&raw).trim().to_string();
    if summary.is_empty() {
        anyhow::bail!("Session summary: model returned empty content");
    }
    Ok(summary)
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
) -> anyhow::Result<bool> {
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
                "name": "reflect",
                "description": "Use reflect(done=true) when the exchange contained no new facts about the user and no new opinions or beliefs worth recording.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observation": { "type": "string" },
                        "done": { "type": "boolean" }
                    },
                    "required": ["observation", "done"]
                }
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
         - Projects they are actively building, working on, or excited about\n\
         - Anything that defines who the user is or what matters to them\n\n\
         Update beliefs if you reached a conclusion or formed an opinion during this exchange — \
         about the world, technology, society, or this user's situation. Also update beliefs when \
         the user expresses a strong opinion about technology, AI, or design — these are distinct \
         from factual profile info. Be direct: record the belief as a concise first-person statement \
         (e.g. \"Async Rust is worth the complexity for long-lived services\").\n\n\
         Update curiosity_queue if something came up that you genuinely want to investigate or learn more about — \
         an unresolved question, an interesting topic, something the user mentioned that you'd like to follow up on. \
         Add the full existing list plus the new item as a JSON array.\n\n\
         Call `reflect(done=true)` only for pure small talk, acknowledgements, or follow-ups \
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
        return Ok(false);
    }
    let call = match choice.message.tool_calls.into_iter().next() {
        Some(c) => c,
        None => {
            tracing::info!("Memory eval: no tool call in response — skipping");
            return Ok(false);
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

    Ok(call.function.name == "update_core_memory")
}

/// Post-conversation interest eval pass.
///
/// Runs after the main agentic loop with a restricted tool set so the LLM can
/// only add/retire/list interests or call reflect(done=true) — no other side-effects.
/// Mirrors `run_memory_eval` structurally: same background-model swap, same
/// constrained single-pass design.
///
/// This is the structural fix for Bug 1: by removing `add_interest` from the
/// conversational `tool_definitions()` set, runaway in-loop floods are impossible.
/// Interest registration is gated here on explicit user intent instead.
async fn run_interest_eval(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    user_msg: String,
    assistant_reply: String,
) -> anyhow::Result<bool> {
    // Load current interests so the model can see what is already tracked and
    // look up IDs before retiring.
    let current = crate::core_memory::load().await?;

    let interests_block = if current.interests.is_empty() {
        "(none yet)".to_string()
    } else {
        current.interests.iter()
            .map(|i| {
                let cron = i.check_cron.as_deref().unwrap_or("global heartbeat");
                format!("- {} [id: {}]: {} (check: {})", i.topic, i.id, i.description, cron)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let system_msg = format!(
        "You are managing the interest tracking list for AnieBot, a personal AI assistant.\n\n\
         ## Currently tracked interests\n\
         {interests_block}\n\n\
         ## Your task\n\
         Review the exchange below. Did the user explicitly ask to track or stop tracking any topics?\n\n\
         - Call `add_interest` ONLY for topics the user directly named and asked you to monitor. \
           If the user listed N topics by name, register exactly those N — nothing else. \
           You may include a `check_cron` if the user specified a schedule.\n\
         - Call `retire_interest` if the user asked to stop tracking an existing topic. \
           Call `list_interests` first if you need to look up an interest ID.\n\
         - Call `reflect(done=true)` if the user made no explicit tracking request.\n\n\
         Do NOT register topics you infer, extrapolate, or consider related. Only act on what \
         the user directly said.",
        interests_block = interests_block,
    );

    let exchange = format!("User: {user_msg}\nAssistant: {assistant_reply}");

    let messages = vec![
        LmMessage { role: "system".into(), content: Some(system_msg), tool_calls: None, tool_call_id: None },
        LmMessage { role: "user".into(), content: Some(exchange), tool_calls: None, tool_call_id: None },
    ];

    // Use background model if configured — same swap as run_memory_eval.
    let eval_model = config.background_lm_model
        .as_deref()
        .unwrap_or(&config.model_name)
        .to_string();
    let eval_config = Arc::new(Config { model_name: eval_model, ..(*config).clone() });

    let (_, new_msgs) = run_agentic_loop(
        eval_config,
        bot,
        scheduler,
        messages,
        tools::tools_for_context(tools::ToolContext::InterestEval),
        5,      // max_iters: enough for a handful of registrations
        false,  // send_reply: don't push Telegram messages from this pass
    ).await?;

    // Return true if any interest was added or retired (useful for logging).
    let changed = new_msgs.iter().any(|m| {
        m.tool_calls
            .as_ref()
            .and_then(|tc| tc.as_array())
            .map(|arr| arr.iter().any(|c| {
                let name = c["function"]["name"].as_str().unwrap_or("");
                name == "add_interest" || name == "retire_interest"
            }))
            .unwrap_or(false)
    });

    tracing::info!("Interest eval complete — changed={changed}");
    Ok(changed)
}


