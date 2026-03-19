use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;
use uuid::Uuid;

use crate::cli_wrapper;
use crate::config::Config;
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
                            "description": "6-field Quartz cron expression: SECONDS MINUTES HOURS DAY-OF-MONTH MONTH DAY-OF-WEEK. Examples: daily at 08:00 = '0 0 8 * * *', weekdays at 09:30 = '0 30 9 * * Mon-Fri', every minute = '0 * * * * *'."
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

            if cron_local.split_whitespace().count() != 6 {
                return Ok(format!(
                    "❌ Invalid cron expression `{}`: expected 6 fields (seconds minutes hours day month weekday).",
                    cron_local
                ));
            }

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
        other => anyhow::bail!("Unknown tool name: {}", other),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a Quartz 6-field cron expression from the system's local timezone to UTC.
/// Only adjusts plain numeric hour (and minute) fields. Returns the original on any
/// parse failure so the scheduler's own validation can surface the error.
///
/// Example (EDT, UTC-4): "0 30 7 * * *" → "0 30 11 * * *"
fn cron_local_to_utc(cron: &str) -> String {
    let parts: Vec<&str> = cron.split_whitespace().collect();
    if parts.len() != 6 {
        return cron.to_string();
    }

    // Parse hour and minute; bail out if either field uses wildcards/ranges.
    let Ok(local_h) = parts[2].parse::<i32>() else { return cron.to_string(); };
    let Ok(local_m) = parts[1].parse::<i32>() else { return cron.to_string(); };

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
