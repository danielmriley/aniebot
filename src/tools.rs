use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;
use uuid::Uuid;

use crate::cli_wrapper;
use crate::config::Config;
use crate::core_memory;
use crate::episodic;
use crate::schedule_store::{self, ScheduleEntry};
use crate::scheduler::{self, SchedulerHandle};

// ---------------------------------------------------------------------------
// LM tool call types (shared with orchestrator)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct LmToolCall {
    pub id: String,
    /// Always "function" per the spec; serialised back into history as-is.
    #[serde(rename = "type", default = "default_tool_type")]
    pub call_type: String,
    pub function: LmFunctionCall,
}

fn default_tool_type() -> String {
    "function".to_string()
}

#[derive(Serialize, Deserialize)]
pub struct LmFunctionCall {
    pub name: String,
    pub arguments: String, // JSON-encoded string
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema sent to LM Studio)
// ---------------------------------------------------------------------------

pub fn tool_definitions() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "delegate_cli",
                "description": "Execute a task using copilot, which has full filesystem access, internet access, and can run shell commands. Use this for: fetching news, weather, stock prices, or any real-time data; reading or writing files; running shell commands; web searches; anything requiring information beyond your training data. Do NOT use this for reactions, thank-yous, follow-up comments, or opinion questions about information you already provided — use reply_to_user for those.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Precise, self-contained task description for copilot. Include all context needed."
                        }
                    },
                    "required": ["task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_task",
                "description": "Create a recurring scheduled reminder or automated task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cron": {
                            "type": "string",
                            "description": "Cron expression in 5-field (MIN HRS DOM MON DOW) or 6-field Quartz format (SEC MIN HRS DOM MON DOW). Examples: daily at 08:00 = '0 8 * * *', weekdays at 09:30 = '30 9 * * Mon-Fri', every 15 minutes = '*/15 * * * *'. A leading seconds field of 0 is added automatically if omitted."
                        },
                        "label": {
                            "type": "string",
                            "description": "Human-readable name for this scheduled job."
                        },
                        "task": {
                            "type": "string",
                            "description": "The prompt that will be run as a task when this job fires."
                        }
                    },
                    "required": ["cron", "label", "task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_schedules",
                "description": "List all currently scheduled jobs.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_schedule",
                "description": "Delete a scheduled job by its UUID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schedule_id": {
                            "type": "string",
                            "description": "The UUID of the schedule to delete."
                        }
                    },
                    "required": ["schedule_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reply_to_user",
                "description": "Send a direct conversational reply to the user. Use this for: greetings; thank-yous and acknowledgements ('thanks', 'got it', 'that's a lot'); reactions to information you just provided; opinion and discussion questions ('what do you think', 'how do you feel'); casual conversation; anything that does not require fetching new external data. When in doubt, use this tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The reply text to send to the user."
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a section of your persistent core memory. Use this when you learn something genuinely new or significant about yourself, your beliefs, or the user. Be conservative — only update when something meaningfully new has been established.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["identity", "beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to update. Use identity for your own self-description; beliefs for your evolving worldview; user_profile for facts about the user; curiosity_queue for topics you want to explore."
                        },
                        "content": {
                            "type": "string",
                            "description": "New value for the section. Plain text for identity/user_profile. JSON array of strings for beliefs/curiosity_queue, e.g. [\"item1\",\"item2\"]."
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Log a specific observation, fact, or event to episodic memory for future recall. Use for things too detailed or transient for core memory — e.g. a specific conversation moment, a user preference detail, or a one-off event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The observation or fact to remember."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional topic tags for later retrieval, e.g. [\"music\", \"preferences\"]."
                        },
                        "importance": {
                            "type": "integer",
                            "description": "Importance 1–5 (5 = most important). Use 3 if unsure."
                        }
                    },
                    "required": ["content", "tags", "importance"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search episodic and archival memory by keyword or tags. Use this before answering questions about past events or user preferences that may not be in the recent conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keyword or phrase to search for in stored memories."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional tags to filter results."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_interest",
                "description": "Register a topic you want to proactively track and receive updates about. The bot will check in on this interest automatically according to the cron schedule (or during every heartbeat if no cron is given).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Short name for the interest, e.g. 'SpaceX launches'."
                        },
                        "description": {
                            "type": "string",
                            "description": "What specifically to look for or track."
                        },
                        "check_cron": {
                            "type": "string",
                            "description": "Optional 6-field Quartz cron for dedicated checks (e.g. '0 0 9 * * Mon' = Monday 09:00 UTC). Omit to rely on the hourly heartbeat."
                        }
                    },
                    "required": ["topic", "description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "retire_interest",
                "description": "Remove a previously registered interest by its ID. Use when the topic is no longer relevant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interest_id": {
                            "type": "string",
                            "description": "The UUID of the interest to retire, as shown in core memory."
                        }
                    },
                    "required": ["interest_id"]
                }
            }
        }
    ])
}

/// Tool definitions for the heartbeat / interest-check agentic loops.
/// Excludes user-specific tools (schedule_task, list_schedules, delete_schedule,
/// update_core_memory, remember, recall) to keep the heartbeat focused.
pub fn heartbeat_tool_definitions() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "delegate_cli",
                "description": "Execute a task using copilot, which has full filesystem access, internet access, and can run shell commands. Use this for fetching news, weather, stock prices, or any real-time data relevant to your active interests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Precise, self-contained task description for copilot."
                        }
                    },
                    "required": ["task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reply_to_user",
                "description": "Send a proactive message to the user. Only use this if you have something genuinely worth sharing — silence is the default.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": { "type": "string", "description": "The message to send." }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_interest",
                "description": "Register a new topic to track proactively.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string", "description": "Short name for the interest." },
                        "description": { "type": "string", "description": "What to look for." },
                        "check_cron": { "type": "string", "description": "Optional 6-field Quartz cron for dedicated checks." }
                    },
                    "required": ["topic", "description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "retire_interest",
                "description": "Remove a previously registered interest by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interest_id": { "type": "string", "description": "UUID of the interest to retire." }
                    },
                    "required": ["interest_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nothing",
                "description": "Do nothing and stay silent. Use this as the default when there is nothing worth sharing with the user right now.",
                "parameters": { "type": "object", "properties": {} }
            }
        }
    ])
}

/// Tool definitions for the consolidation reflection pass.
/// Allows the model to update core memory, persist observations, manage interests.
pub fn consolidation_tool_definitions() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a section of your core memory document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to overwrite. Note: 'identity' is read-only and cannot be changed during consolidation."
                        },
                        "content": {
                            "type": "string",
                            "description": "New content. Plain text for identity/user_profile; JSON array string for beliefs/curiosity_queue."
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Persist an important observation to episodic memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string", "description": "The observation to record." },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Categorization tags." },
                        "importance": { "type": "integer", "description": "Importance 1-5. Use 5 for consolidation-promoted facts." }
                    },
                    "required": ["content", "importance"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_interest",
                "description": "Register a new topic to track proactively.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string", "description": "Short name for the interest." },
                        "description": { "type": "string", "description": "What to look for." },
                        "check_cron": { "type": "string", "description": "Optional cron for dedicated checks." }
                    },
                    "required": ["topic", "description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "retire_interest",
                "description": "Remove a previously registered interest by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interest_id": { "type": "string", "description": "UUID of the interest to retire." }
                    },
                    "required": ["interest_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nothing",
                "description": "Nothing to update from this consolidation cycle.",
                "parameters": { "type": "object", "properties": {} }
            }
        }
    ])
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

pub async fn dispatch_tool_call(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    call: LmToolCall,
) -> anyhow::Result<String> {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .context("Tool call arguments are not valid JSON")?;

    match call.function.name.as_str() {
        "reply_to_user" => {
            let message = args["message"].as_str()
                .context("reply_to_user missing 'message' argument")?;
            Ok(message.to_string())
        }
        "delegate_cli" => {
            let task = args["task"].as_str()
                .context("delegate_cli missing 'task' argument")?;
            tracing::info!("Delegating to copilot: {}", task);
            match cli_wrapper::run(&config, task).await {
                Ok(output) => Ok(format!("✅ Done!\n\n{}", output)),
                Err(e) => Ok(format!("❌ CLI error: {}", e)),
            }
        }
        "schedule_task" => {
            let cron_local = args["cron"].as_str().context("schedule_task missing 'cron'")?;
            let label = args["label"].as_str().unwrap_or("unnamed task");
            let task = args["task"].as_str().context("schedule_task missing 'task'")?;

            // Auto-promote standard 5-field cron (MIN HRS DOM MON DOW) to the
            // 6-field Quartz format (SEC MIN HRS DOM MON DOW) that tokio-cron-scheduler
            // requires.  LLMs commonly emit the 5-field variant, so we silently fix it
            // rather than rejecting it.
            let cron_local_owned;
            let cron_local = match cron_local.split_whitespace().count() {
                5 => {
                    cron_local_owned = format!("0 {}", cron_local);
                    tracing::info!("Auto-promoted 5-field cron to 6-field: {}", cron_local_owned);
                    cron_local_owned.as_str()
                }
                6 => cron_local,
                _ => {
                    return Ok(format!(
                        "❌ Invalid cron expression `{}`: expected 5 or 6 fields \
                         (e.g. '30 9 * * Mon-Fri' or '0 30 9 * * Mon-Fri').",
                        cron_local
                    ));
                }
            };

            let cron = cron_local_to_utc(cron_local);
            if cron != cron_local {
                tracing::info!("Cron converted local→UTC: {} → {}", cron_local, cron);
            }

            let entry = ScheduleEntry {
                id: Uuid::new_v4().to_string(),
                label: label.to_string(),
                cron: cron.clone(),
                task: task.to_string(),
                created_at: Utc::now(),
            };
            let entry_id = entry.id.clone();
            let entry_label = entry.label.clone();

            if let Err(e) = scheduler::add_dynamic_job(&scheduler, bot, config.clone(), entry.clone()).await {
                return Ok(format!("❌ Failed to register job: {}", e));
            }
            if let Err(e) = schedule_store::append(entry).await {
                tracing::warn!("Failed to persist schedule entry: {}", e);
            }

            Ok(format!("✅ Scheduled *{}* (ID: `{}`)\nCron: `{}`", entry_label, entry_id, cron))
        }
        "list_schedules" => {
            let entries = schedule_store::load().await.unwrap_or_default();
            if entries.is_empty() {
                return Ok("No scheduled jobs yet.".into());
            }
            let list = entries
                .iter()
                .map(|e| format!("• *{}*\n  ID: `{}`\n  Cron: `{}`", e.label, e.id, e.cron))
                .collect::<Vec<_>>()
                .join("\n\n");
            Ok(format!("Scheduled jobs:\n\n{}", list))
        }
        "delete_schedule" => {
            let id = args["schedule_id"].as_str()
                .context("delete_schedule missing 'schedule_id'")?;
            let removed_live = scheduler::remove_dynamic_job(&scheduler, id).await.unwrap_or(false);
            let removed_store = schedule_store::remove(id).await.unwrap_or(false);
            if removed_live || removed_store {
                Ok(format!("✅ Deleted schedule `{}`.", id))
            } else {
                Ok(format!("❌ No schedule found with ID `{}`.", id))
            }
        }
        "update_core_memory" => {
            let section = args["section"].as_str()
                .context("update_core_memory missing 'section' argument")?;
            let content = args["content"].as_str()
                .context("update_core_memory missing 'content' argument")?;
            core_memory::update_section(section, content).await?;
            Ok("\u{2713} Core memory updated.".into())
        }
        "remember" => {
            let content = args["content"].as_str()
                .context("remember missing 'content' argument")?;
            let tags: Vec<String> = args["tags"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let importance = args["importance"].as_u64().unwrap_or(3).min(5) as u8;
            let entry = episodic::EpisodicEntry {
                id: Uuid::new_v4().to_string(),
                content: content.to_string(),
                tags,
                importance,
                timestamp: Utc::now(),
                promoted: false,
            };
            episodic::append(&entry).await?;
            Ok("\u{2713} Noted.".into())
        }
        "recall" => {
            let query = args["query"].as_str().unwrap_or("");
            let tags: Vec<String> = args["tags"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let results = episodic::search(query, &tags).await?;
            if results.is_empty() {
                Ok("No matching memories found.".into())
            } else {
                let lines: Vec<String> = results
                    .iter()
                    .map(|e| format!("[{}] (importance: {}) {}", e.timestamp.format("%Y-%m-%d"), e.importance, e.content))
                    .collect();
                Ok(lines.join("\n"))
            }
        }
        "add_interest" => {
            let topic = args["topic"].as_str()
                .context("add_interest missing 'topic' argument")?;
            let description = args["description"].as_str()
                .context("add_interest missing 'description' argument")?;
            // Normalize the optional cron: promote 5-field → 6-field Quartz, then UTC-shift.
            // Treat absent OR empty string as no cron (LLMs sometimes emit "" for optional fields).
            let check_cron: Option<String> = match args["check_cron"].as_str() {
                None | Some("") => None,
                Some(raw) => {
                    let promoted = match raw.split_whitespace().count() {
                        5 => {
                            let s = format!("0 {}", raw);
                            tracing::info!("add_interest: auto-promoted cron to 6-field: {}", s);
                            s
                        }
                        6 => raw.to_string(),
                        _ => {
                            return Ok(format!(
                                "❌ Invalid cron `{}` for interest: expected 5 or 6 fields.",
                                raw
                            ));
                        }
                    };
                    let utc = cron_local_to_utc(&promoted);
                    Some(utc)
                }
            };
            let has_cron = check_cron.is_some();
            let interest = core_memory::add_interest(topic, description, check_cron).await?;
            if has_cron {
                if let Err(e) = crate::scheduler::add_interest_job(&scheduler, bot.clone(), config.clone(), interest).await {
                    tracing::warn!("Failed to register interest cron job: {e}");
                }
            }
            Ok(format!("\u{2713} Interest registered: {topic}"))
        }
        "retire_interest" => {
            let id = args["interest_id"].as_str()
                .context("retire_interest missing 'interest_id' argument")?;
            core_memory::retire_interest(id).await?;
            // Best-effort removal — the job may not exist if the interest had no cron.
            let _ = crate::scheduler::remove_dynamic_job(&scheduler, id).await;
            Ok("\u{2713} Interest retired.".into())
        }
        "nothing" => Ok(String::new()),
        other => anyhow::bail!("Unknown tool name: {}", other),
    }
}

/// Convert a Quartz 6-field cron expression from the system's local timezone to UTC.
/// Only adjusts plain numeric hour (and minute) fields. Returns the original on any
/// parse failure so the scheduler's own validation can surface the error.
///
/// When the HOURS or MINUTES fields use wildcards, ranges, or step patterns the
/// expression is returned unchanged (the scheduler fires in UTC, which equals local
/// time only in the UTC timezone in such cases).
///
/// Example (EDT, UTC-4): "0 30 7 * * *" → "0 30 11 * * *"
fn cron_local_to_utc(cron: &str) -> String {
    let parts: Vec<&str> = cron.split_whitespace().collect();
    if parts.len() != 6 {
        return cron.to_string();
    }

    // Parse hour and minute; bail out (with a warning) if either field uses wildcards/ranges
    // — we can't shift those fields by a fixed offset.
    let Ok(local_h) = parts[2].parse::<i32>() else {
        tracing::debug!(
            "cron_local_to_utc: non-integer HOURS field '{}', skipping timezone conversion",
            parts[2]
        );
        return cron.to_string();
    };
    let Ok(local_m) = parts[1].parse::<i32>() else {
        tracing::debug!(
            "cron_local_to_utc: non-integer MINUTES field '{}', skipping timezone conversion",
            parts[1]
        );
        return cron.to_string();
    };

    // local_minus_utc() is negative for zones west of UTC (e.g. EDT = -14400).
    let offset_secs = chrono::Local::now().offset().local_minus_utc();
    let offset_h = offset_secs / 3600;
    let offset_m = (offset_secs % 3600) / 60;

    // Subtract the offset to convert local → UTC.
    let total_utc_mins = local_h * 60 + local_m - offset_h * 60 - offset_m;
    let utc_h = total_utc_mins.div_euclid(60).rem_euclid(24);
    let utc_m = total_utc_mins.rem_euclid(60);

    format!("{} {} {} {} {} {}", parts[0], utc_m, utc_h, parts[3], parts[4], parts[5])
}
