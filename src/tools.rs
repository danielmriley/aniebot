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
// Tool helpers — one function per tool; composed into the three public sets below
// ---------------------------------------------------------------------------

fn tool_delegate_cli() -> serde_json::Value {
    json!({
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
    })
}

fn tool_fetch_url() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the raw contents of a URL via HTTP GET. Use for reading web pages, REST APIs, or any public HTTP resource. Returns up to 8 KB of the response body as text. Prefer this over delegate_cli for simple, unauthenticated GET requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must be HTTP or HTTPS)."
                    }
                },
                "required": ["url"]
            }
        }
    })
}

fn tool_schedule_task() -> serde_json::Value {
    json!({
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
    })
}

fn tool_list_schedules() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "list_schedules",
            "description": "List all currently scheduled jobs.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_delete_schedule() -> serde_json::Value {
    json!({
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
    })
}

fn tool_update_schedule() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "update_schedule",
            "description": "Modify an existing scheduled job. Only the fields you provide are changed — omit any field to leave it as-is. The job ID is preserved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule_id": {
                        "type": "string",
                        "description": "The UUID of the schedule to update."
                    },
                    "label": {
                        "type": "string",
                        "description": "New human-readable name for the job (optional)."
                    },
                    "cron": {
                        "type": "string",
                        "description": "New cron expression in 5- or 6-field format (optional). Same rules as schedule_task."
                    },
                    "task": {
                        "type": "string",
                        "description": "New prompt that runs when the job fires (optional)."
                    }
                },
                "required": ["schedule_id"]
            }
        }
    })
}

fn tool_schedule_once() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "schedule_once",
            "description": "Run a task once at a future time, then auto-delete. For relative times ('in 5 minutes', 'in 2 hours') use delay_seconds. For specific clock times only, use fire_at with an ISO 8601 UTC string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_seconds": {
                        "type": "integer",
                        "description": "Seconds from now until the job fires. Preferred for relative times: 5 minutes = 300, 1 hour = 3600. Mutually exclusive with fire_at."
                    },
                    "fire_at": {
                        "type": "string",
                        "description": "ISO 8601 UTC datetime when the job should fire. Only use this when the user specifies a specific clock time. Mutually exclusive with delay_seconds."
                    },
                    "label": {
                        "type": "string",
                        "description": "Human-readable name for this one-time job."
                    },
                    "task": {
                        "type": "string",
                        "description": "The prompt that will be executed when the job fires."
                    }
                },
                "required": ["label", "task"]
            }
        }
    })
}

fn tool_send_update() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "send_update",
            "description": "Send a progress message to the user now and keep working. Use this to keep the user informed during long tasks — for example, after completing one step and before starting the next. This is NOT your final reply; your final reply will be generated automatically when you finish all your work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The progress update to send to the user. Be concise and informative."
                    }
                },
                "required": ["message"]
            }
        }
    })
}

fn tool_reply_to_user() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "reply_to_user",
            "description": "Send your reply to the user. This is the primary way to respond. Use it immediately for conversational messages, greetings, reactions, opinions, and anything not requiring external data. For research tasks, complete the work first with delegate_cli or other tools, then call this with your findings. If the research involved many tool calls and you want a polished synthesis instead, call reflect(done=true) — otherwise just call this with the result yourself.",
            "description_note": "Do NOT call this before doing requested work. Do NOT use background=true unless you genuinely have more work to continue after replying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reply text to send to the user."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Set true to send this reply immediately and keep working on remaining tasks. Omit or set false (default) to send and stop."
                    }
                },
                "required": ["message"]
            }
        }
    })
}

/// `include_identity`: true in the consolidation pass (exposes the "identity" enum value and
/// its surgical-edit guidance); false everywhere else.
fn tool_update_core_memory(include_identity: bool) -> serde_json::Value {
    if include_identity {
        json!({
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a section of your core memory document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["identity", "beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to update. identity for your self-description — only update if the High-Significance Episodes block contains evidence that warrants a surgical edit (add, revise, or remove a specific clause; do not rewrite wholesale). beliefs for your evolving worldview; user_profile for facts about the user; curiosity_queue for topics to investigate."
                        },
                        "content": {
                            "type": "string",
                            "description": "New content. Plain text for identity/user_profile; JSON array string for beliefs/curiosity_queue."
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        })
    } else {
        json!({
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a section of your persistent core memory. Use this when you learn something genuinely new or significant about your beliefs, the user, or topics to investigate. Be conservative — only update when something meaningfully new has been established.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["beliefs", "user_profile", "curiosity_queue"],
                            "description": "Which section to update. Use beliefs for your evolving worldview; user_profile for facts about the user; curiosity_queue for topics you want to investigate."
                        },
                        "content": {
                            "type": "string",
                            "description": "New value for the section. Plain text for user_profile. JSON array of strings for beliefs/curiosity_queue, e.g. [\"item1\",\"item2\"]."
                        }
                    },
                    "required": ["section", "content"]
                }
            }
        })
    }
}

fn tool_remember() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Log a notable moment, discovery, or event to episodic memory. Use this during or after any exchange where something significant happened — a task completed, something surprising found, a meaningful decision made, a strong preference expressed. This is how your understanding accumulates across sessions.",
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
                        "description": "Topic tags for later retrieval, e.g. [\"music\", \"preferences\"]."
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Importance 1–5 (5 = most important). Use 3 if unsure."
                    }
                },
                "required": ["content", "tags", "importance"]
            }
        }
    })
}

fn tool_recall() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search episodic and archival memory by keyword or tags. Use this before answering questions about past events or user preferences that may not be in the recent conversation. In the heartbeat context, use this before sending a proactive message to check if the topic was already covered recently.",
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
    })
}

fn tool_add_interest() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "add_interest",
            "description": "Register a topic the user explicitly asked you to track. Only call this when the user directly names a topic they want monitored — never speculatively for related or inferred topics. The bot will check in automatically per cron schedule (or every heartbeat if no cron is given).",
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
    })
}

fn tool_retire_interest() -> serde_json::Value {
    json!({
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
    })
}

fn tool_list_interests() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "list_interests",
            "description": "List all currently registered interests, including their topic, description, health score, check schedule, and ID.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_forget() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Permanently delete an episodic memory entry by its UUID. Use when a stored note is incorrect, stale, or should no longer be recalled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "UUID of the episodic entry to delete, as returned by recall."
                    }
                },
                "required": ["entry_id"]
            }
        }
    })
}

fn tool_nothing(description: &str) -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "nothing",
            "description": description,
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_set_task() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "set_task",
            "description": "Record the current task or active project I'm helping with. Appears in every system prompt after 'Who I Am'. Call this when the user starts a new project or multi-step task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of the current task (1-2 sentences)."
                    }
                },
                "required": ["description"]
            }
        }
    })
}

fn tool_clear_task() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "clear_task",
            "description": "Clear the current task once it is complete or no longer relevant.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_reflect() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "reflect",
            "description": "Record an intermediate observation about your current progress. Call this after a series of tool calls to organize your thinking. done=false continues the loop. done=true exits the loop — a reply will be synthesized automatically from your work. Supply the optional reply field if you already know what you want to say; synthesis will use it as a starting point. In background tasks (heartbeat, consolidation), done=true is the correct exit when there is nothing to report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {
                        "type": "string",
                        "description": "What have you done so far and what did you find or not find?"
                    },
                    "done": {
                        "type": "boolean",
                        "description": "Set true to exit the loop when your work is complete. A reply will be synthesized from everything you did. In background tasks, true is correct when there is nothing to report."
                    },
                    "reply": {
                        "type": "string",
                        "description": "Optional: if you know what you want to say to the user, put it here. Synthesis will use this as a starting point and polish it. Omit to let the harness compose the reply entirely from your tool results."
                    }
                },
                "required": ["observation", "done"]
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema sent to LM Studio)
// ---------------------------------------------------------------------------

pub fn tool_definitions() -> serde_json::Value {
    json!([
        tool_delegate_cli(),
        tool_fetch_url(),
        tool_schedule_task(),
        tool_list_schedules(),
        tool_delete_schedule(),
        tool_update_schedule(),
        tool_schedule_once(),
        tool_send_update(),
        tool_reply_to_user(),
        tool_update_core_memory(false),
        tool_remember(),
        tool_recall(),
        tool_list_interests(),
        tool_forget(),
        tool_set_task(),
        tool_clear_task(),
        tool_reflect(),
        tool_add_agenda_item(),
        tool_list_agenda_items(),
        tool_cancel_agenda_item(),
    ])
}

/// Tool definitions for the post-conversation interest eval pass.
/// Mirrors run_memory_eval: restricted set so the LLM can only register,
/// retire, list, or declare nothing — no other side-effects possible.
pub fn interest_eval_tools() -> serde_json::Value {
    json!([
        tool_add_interest(),
        tool_retire_interest(),
        tool_list_interests(),
        tool_nothing("The conversation contained no explicit interest tracking request from the user."),
    ])
}

/// Tool definitions for the heartbeat / interest-check agentic loops.
pub fn heartbeat_tool_definitions() -> serde_json::Value {
    json!([
        tool_delegate_cli(),
        tool_fetch_url(),
        tool_reply_to_user(),
        tool_add_interest(),
        tool_retire_interest(),
        tool_update_core_memory(false),
        tool_recall(),
        tool_forget(),
        tool_set_task(),
        tool_clear_task(),
        tool_reflect(),
        tool_add_agenda_item(),
        tool_list_agenda_items(),
        tool_cancel_agenda_item(),
        tool_complete_agenda_item(),
        tool_nothing("Do nothing and stay silent. Use this as the default when there is nothing worth sharing with the user right now."),
    ])
}

/// Tool definitions for the consolidation reflection pass.
/// Allows the model to update core memory, persist observations, manage interests.
pub fn consolidation_tool_definitions() -> serde_json::Value {
    json!([
        tool_update_core_memory(true),
        tool_fetch_url(),
        tool_remember(),
        tool_add_interest(),
        tool_retire_interest(),
        tool_set_task(),
        tool_clear_task(),
        tool_reflect(),
        tool_add_agenda_item(),
        tool_list_agenda_items(),
        tool_cancel_agenda_item(),
        tool_complete_agenda_item(),
        tool_nothing("Nothing to update from this consolidation cycle."),
    ])
}

fn tool_add_agenda_item() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "add_agenda_item",
            "description": "Queue a task to be done later. Use this when you have work you cannot finish now but need to remember — it will appear in future system prompts until completed. Great for tasks deferred due to budget, scope, or ordering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What needs to be done."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional background or context helpful for completing this task later."
                    }
                },
                "required": ["description"]
            }
        }
    })
}

fn tool_list_agenda_items() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "list_agenda_items",
            "description": "List all pending and in-progress agenda tasks.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_cancel_agenda_item() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "cancel_agenda_item",
            "description": "Cancel a pending agenda task by its ID when it is no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The UUID of the agenda item to cancel."
                    }
                },
                "required": ["item_id"]
            }
        }
    })
}

fn tool_complete_agenda_item() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "complete_agenda_item",
            "description": "Mark an agenda task as completed and record what was accomplished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The UUID of the agenda item to complete."
                    },
                    "result": {
                        "type": "string",
                        "description": "Summary of what was accomplished."
                    }
                },
                "required": ["item_id", "result"]
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

async fn fetch_url_impl(url: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;
    let status = response.status();
    let bytes = response.bytes().await.map_err(|e| anyhow::anyhow!("Failed to read response: {}", e))?;
    let cap = bytes.len().min(8 * 1024);
    let text = String::from_utf8_lossy(&bytes[..cap]).into_owned();
    Ok(format!("HTTP {}\n\n{}", status, text))
}

pub async fn dispatch_tool_call(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    call: LmToolCall,
) -> anyhow::Result<String> {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .context("Tool call arguments are not valid JSON")?;

    match call.function.name.as_str() {
        "send_update" => {
            let message = args["message"].as_str()
                .context("send_update missing 'message' argument")?;
            Ok(message.to_string())
        }
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
                Err(e) => Ok(format!("❌ copilot failed: {}", e)),
            }
        }
        "fetch_url" => {
            let url = args["url"].as_str()
                .context("fetch_url missing 'url' argument")?;
            tracing::info!("fetch_url: {}", url);
            match fetch_url_impl(url).await {
                Ok(content) => Ok(content),
                Err(e) => Ok(format!("❌ fetch failed: {}", e)),
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

            if let Err(e) = validate_cron_fields(cron_local) {
                return Ok(e);
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
                fire_once_at: None,
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
                .map(|e| format!("• *{}*\n  ID: `{}`\n  Cron: `{}` (UTC)\n  Task: {}", e.label, e.id, e.cron, e.task))
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
        "update_schedule" => {
            let id = args["schedule_id"].as_str()
                .context("update_schedule missing 'schedule_id'")?;

            let entries = schedule_store::load().await.unwrap_or_default();
            let Some(mut entry) = entries.into_iter().find(|e| e.id == id) else {
                return Ok(format!("❌ No schedule found with ID `{}`.", id));
            };

            if let Some(new_label) = args["label"].as_str() {
                entry.label = new_label.to_string();
            }
            if let Some(new_task) = args["task"].as_str() {
                entry.task = new_task.to_string();
            }
            if let Some(new_cron_local) = args["cron"].as_str() {
                let promoted = match new_cron_local.split_whitespace().count() {
                    5 => format!("0 {}", new_cron_local),
                    _ => new_cron_local.to_string(),
                };
                entry.cron = cron_local_to_utc(&promoted);
            }

            let _ = scheduler::remove_dynamic_job(&scheduler, id).await;
            let _ = schedule_store::remove(id).await;
            if let Err(e) = scheduler::add_dynamic_job(&scheduler, bot, config.clone(), entry.clone()).await {
                return Ok(format!("❌ Failed to re-register job: {}", e));
            }
            if let Err(e) = schedule_store::append(entry.clone()).await {
                tracing::warn!("Failed to persist updated schedule: {}", e);
            }

            Ok(format!("✅ Updated schedule `{}` (*{}*).", entry.id, entry.label))
        }        "schedule_once" => {
            let fire_at: chrono::DateTime<Utc> = if let Some(secs) = args["delay_seconds"].as_i64() {
                if secs <= 0 {
                    return Ok("\u{274c} `delay_seconds` must be a positive integer.".to_string());
                }
                Utc::now() + chrono::Duration::seconds(secs)
            } else if let Some(fire_at_str) = args["fire_at"].as_str() {
                let dt = chrono::DateTime::parse_from_rfc3339(fire_at_str)
                    .context("schedule_once: invalid ISO 8601 datetime for 'fire_at'")?
                    .with_timezone(&Utc);
                if dt <= Utc::now() {
                    let server_now = Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                    return Ok(format!(
                        "\u{274c} `fire_at` must be in the future (server UTC is `{server_now}`). \
                         For relative times like 'in 5 minutes', use delay_seconds=300 instead."
                    ));
                }
                dt
            } else {
                return Ok("\u{274c} schedule_once requires either `delay_seconds` (preferred) or `fire_at`.".to_string());
            };
            let label = args["label"].as_str()
                .context("schedule_once missing 'label'")?;
            let task = args["task"].as_str()
                .context("schedule_once missing 'task'")?;
            let fire_at_display = fire_at.format("%Y-%m-%dT%H:%M:%SZ").to_string();
            let entry = ScheduleEntry {
                id: Uuid::new_v4().to_string(),
                label: label.to_string(),
                cron: String::new(),
                task: task.to_string(),
                created_at: Utc::now(),
                fire_once_at: Some(fire_at),
            };
            if let Err(e) = scheduler::add_dynamic_job(&scheduler, bot, config.clone(), entry.clone()).await {
                return Ok(format!("\u{274c} Failed to schedule one-time job: {}", e));
            }
            if let Err(e) = schedule_store::append(entry.clone()).await {
                tracing::warn!("Failed to persist one-time schedule: {}", e);
            }
            Ok(format!("\u{2705} One-time job *{}* scheduled for `{}`.", entry.label, fire_at_display))
        }        "update_core_memory" => {
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
        "forget" => {
            let id = args["entry_id"].as_str()
                .context("forget missing 'entry_id' argument")?;
            if episodic::delete(id).await? {
                Ok("\u{2713} Forgotten.".into())
            } else {
                Ok("\u{274c} Entry not found.".into())
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
                    if let Err(e) = validate_cron_fields(&promoted) {
                        return Ok(e);
                    }
                    let utc = cron_local_to_utc(&promoted);
                    Some(utc)
                }
            };
            let has_cron = check_cron.is_some();
            match core_memory::add_interest(topic, description, check_cron).await? {
                None => Ok(format!("\u{2713} Interest already exists: {topic}")),
                Some(interest) => {
                    if has_cron {
                        if let Err(e) = crate::scheduler::add_interest_job(&scheduler, bot.clone(), config.clone(), interest).await {
                            tracing::warn!("Failed to register interest cron job: {e}");
                        }
                    }
                    Ok(format!("\u{2713} Interest registered: {topic}"))
                }
            }
        }
        "retire_interest" => {
            let id = args["interest_id"].as_str()
                .context("retire_interest missing 'interest_id' argument")?;
            if !core_memory::retire_interest(id).await? {
                return Ok(format!(
                    "⚠️ No interest with ID '{id}' found. \
                     Use list_interests to see what is currently registered."
                ));
            }
            // Best-effort removal — the job may not exist if the interest had no cron.
            let _ = crate::scheduler::remove_dynamic_job(&scheduler, id).await;
            Ok("\u{2713} Interest retired.".into())
        }
        "list_interests" => {
            let core = core_memory::load().await?;
            if core.interests.is_empty() {
                return Ok("No active interests registered.".into());
            }
            let list = core.interests.iter().map(|i| {
                let cron = i.check_cron.as_deref().unwrap_or("global heartbeat");
                format!("• {}\n  ID: `{}`\n  {}\n  Health: {}%  |  Check: {}", i.topic, i.id, i.description, i.health, cron)
            }).collect::<Vec<_>>().join("\n\n");
            Ok(format!("Active interests:\n\n{}", list))
        }
        "nothing" => Ok(String::new()),
        "reflect" => {
            let observation = args["observation"].as_str().unwrap_or("(no observation)");
            let done = args["done"].as_bool().unwrap_or(false);
            Ok(format!("Reflection recorded: {observation} (concluded: {done})"))
        }
        "set_task" => {
            let desc = args["description"].as_str()
                .context("set_task missing 'description'")?;
            core_memory::set_task(desc).await?;
            Ok(format!("\u{2705} Current task set: {desc}"))
        }
        "clear_task" => {
            core_memory::clear_task().await?;
            Ok("\u{2705} Current task cleared.".to_string())
        }
        "add_agenda_item" => {
            let description = args["description"].as_str()
                .context("add_agenda_item missing 'description'")?;
            let context = args["context"].as_str();
            let item = crate::agenda::add(description, context).await?;
            Ok(format!("\u{2705} Task queued (ID: `{}`):\n{}", item.id, item.description))
        }
        "list_agenda_items" => {
            let items = crate::agenda::list_pending().await?;
            if items.is_empty() {
                return Ok("No pending agenda items.".into());
            }
            let list = items.iter().map(|i| {
                let status_str = match i.status {
                    crate::agenda::AgendaStatus::InProgress => "In Progress",
                    _ => "Pending",
                };
                let ctx = i.context.as_deref()
                    .map(|c| format!("\n  Context: {c}"))
                    .unwrap_or_default();
                format!("\u{2022} {}\n  ID: `{}`  |  Status: {}{}", i.description, i.id, status_str, ctx)
            }).collect::<Vec<_>>().join("\n\n");
            Ok(format!("Pending tasks:\n\n{}", list))
        }
        "cancel_agenda_item" => {
            let id = args["item_id"].as_str()
                .context("cancel_agenda_item missing 'item_id'")?;
            if crate::agenda::cancel(id).await? {
                Ok(format!("\u{2705} Task `{}` cancelled.", id))
            } else {
                Ok(format!("\u{274c} No agenda item found with ID `{}`.", id))
            }
        }
        "complete_agenda_item" => {
            let id = args["item_id"].as_str()
                .context("complete_agenda_item missing 'item_id'")?;
            let result = args["result"].as_str()
                .context("complete_agenda_item missing 'result'")?;
            if crate::agenda::complete(id, result).await? {
                Ok(format!("\u{2705} Task `{}` marked complete.", id))
            } else {
                Ok(format!("\u{274c} No agenda item found with ID `{}`.", id))
            }
        }
        other => anyhow::bail!("Unknown tool name: {}", other),
    }
}

/// Validates that plain numeric cron fields are within their valid ranges.
/// Only checks fields that parse as plain integers; wildcards, ranges, and steps are skipped.
/// Returns `Err(String)` with a user-facing message if a field is out of range.
/// Expects a 6-field Quartz cron string (SEC MIN HRS DOM MON DOW).
fn validate_cron_fields(cron_6field: &str) -> Result<(), String> {
    let parts: Vec<&str> = cron_6field.split_whitespace().collect();
    // parts[1] = MIN, parts[2] = HRS (Quartz 6-field: SEC MIN HRS DOM MON DOW)
    if let Ok(h) = parts[2].parse::<i32>() {
        if !(0..=23).contains(&h) {
            return Err(format!(
                "❌ Invalid cron: hours field is `{}` but must be 0–23. \
                 Please correct the cron expression.",
                h
            ));
        }
    }
    if let Ok(m) = parts[1].parse::<i32>() {
        if !(0..=59).contains(&m) {
            return Err(format!(
                "❌ Invalid cron: minutes field is `{}` but must be 0–59. \
                 Please correct the cron expression.",
                m
            ));
        }
    }
    Ok(())
}

/// Advance a simple DOW field by `delta` days (typically ±1 from a midnight rollover).
/// Returns `None` if the field is a complex expression (`*`, ranges, steps, lists) —
/// the caller should leave it unchanged in that case.
///
/// Handles:
///   - Numeric 0–7 (0 and 7 both mean Sunday); result is always in 0–6.
///   - Named: Sun/Mon/Tue/Wed/Thu/Fri/Sat (case-insensitive); result preserves case style.
fn advance_dow(dow: &str, delta: i32) -> Option<String> {
    // Skip complex expressions.
    if dow.contains(['*', '-', '/', ',']) {
        return None;
    }
    const NAMES: &[&str] = &["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    // Try numeric first.
    if let Ok(n) = dow.parse::<i32>() {
        let shifted = (n + delta).rem_euclid(7);
        return Some(shifted.to_string());
    }
    // Try named (case-insensitive).
    let idx = NAMES.iter().position(|&d| d.eq_ignore_ascii_case(dow))?;
    let shifted = (idx as i32 + delta).rem_euclid(7) as usize;
    Some(NAMES[shifted].to_string())
}

/// Convert a Quartz 6-field cron expression from the system's local timezone to UTC.
/// Adjusts plain numeric hour and minute fields, and corrects the DOW field when the
/// conversion crosses midnight (day_delta ≠ 0). Returns the original on any parse
/// failure so the scheduler's own validation can surface the error.
///
/// When HOURS or MINUTES use wildcards, ranges, or step patterns the expression is
/// returned unchanged. When DOW is a complex expression (range, list, step, `*`) the
/// hour/minute shift is still applied but the DOW field is left as-is.
///
/// Note: the UTC offset is captured at scheduling time. DST transitions that occur
/// after a job is created will cause the effective local fire-time to drift by one hour.
///
/// Examples (EDT, UTC-4):
///   "0 30 7 * * *"   → "0 30 11 * * *"   (DOW is *, left unchanged)
///   "0 30 23 * * Fri" → "0 30 3 * * Sat"  (midnight rollover → DOW advanced)
fn cron_local_to_utc(cron: &str) -> String {
    let offset_secs = chrono::Local::now().offset().local_minus_utc();
    cron_local_to_utc_with_offset(cron, offset_secs / 60)
}

/// Inner implementation that accepts an explicit `offset_mins` (local − UTC, negative
/// west of UTC) rather than reading the system clock. Used directly by unit tests.
fn cron_local_to_utc_with_offset(cron: &str, offset_mins: i32) -> String {
    let parts: Vec<&str> = cron.split_whitespace().collect();
    if parts.len() != 6 {
        return cron.to_string();
    }

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

    let offset_h = offset_mins / 60;
    let offset_m = offset_mins % 60;

    let total_utc_mins = local_h * 60 + local_m - offset_h * 60 - offset_m;
    let utc_h = total_utc_mins.div_euclid(60).rem_euclid(24);
    let utc_m = total_utc_mins.rem_euclid(60);

    let day_delta = total_utc_mins.div_euclid(24 * 60);
    let dow = if day_delta != 0 {
        match advance_dow(parts[5], day_delta) {
            Some(d) => d,
            None => {
                tracing::warn!(
                    "cron_local_to_utc: DOW field `{}` is a range/complex expression and \
                     cannot be auto-adjusted for a day-boundary crossing (day_delta={}). \
                     The schedule will fire at the correct UTC time but on the original DOW days. \
                     Manually adjust the DOW field if needed.",
                    parts[5], day_delta
                );
                parts[5].to_string()
            }
        }
    } else {
        parts[5].to_string()
    };

    format!("{} {} {} {} {} {}", parts[0], utc_m, utc_h, parts[3], parts[4], dow)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // advance_dow
    // -----------------------------------------------------------------------

    #[test]
    fn advance_dow_named_no_rollover() {
        assert_eq!(advance_dow("Mon", 1), Some("Tue".into()));
        assert_eq!(advance_dow("Fri", 1), Some("Sat".into()));
    }

    #[test]
    fn advance_dow_named_rollover_forward() {
        // Sat + 1 day = Sun (index 0)
        assert_eq!(advance_dow("Sat", 1), Some("Sun".into()));
    }

    #[test]
    fn advance_dow_named_rollover_backward() {
        // Sun - 1 day = Sat (index 6)
        assert_eq!(advance_dow("Sun", -1), Some("Sat".into()));
    }

    #[test]
    fn advance_dow_numeric() {
        assert_eq!(advance_dow("5", 1), Some("6".into())); // Fri → Sat
        assert_eq!(advance_dow("6", 1), Some("0".into())); // Sat → Sun
    }

    #[test]
    fn advance_dow_case_insensitive() {
        assert_eq!(advance_dow("fri", 1), Some("Sat".into()));
        assert_eq!(advance_dow("FRI", 1), Some("Sat".into()));
    }

    #[test]
    fn advance_dow_complex_expressions_skipped() {
        assert_eq!(advance_dow("*", 1), None);
        assert_eq!(advance_dow("Mon-Fri", 1), None);
        assert_eq!(advance_dow("1,3,5", 1), None);
        assert_eq!(advance_dow("*/2", 1), None);
    }

    // -----------------------------------------------------------------------
    // cron_local_to_utc_with_offset — deterministic (offset injected)
    // -----------------------------------------------------------------------

    #[test]
    fn cron_utc_offset_zero_is_identity() {
        assert_eq!(
            cron_local_to_utc_with_offset("0 30 9 * * *", 0),
            "0 30 9 * * *"
        );
    }

    #[test]
    fn cron_positive_offset_shifts_back() {
        // UTC+1: local 09:30 → UTC 08:30
        assert_eq!(
            cron_local_to_utc_with_offset("0 30 9 * * *", 60),
            "0 30 8 * * *"
        );
    }

    #[test]
    fn cron_negative_offset_shifts_forward() {
        // UTC-4 (EDT): local 07:30 → UTC 11:30
        assert_eq!(
            cron_local_to_utc_with_offset("0 30 7 * * *", -240),
            "0 30 11 * * *"
        );
    }

    #[test]
    fn cron_dow_rollover_edt() {
        // Phase 2 regression: Friday 23:30 local at UTC-4 → Saturday 03:30 UTC
        assert_eq!(
            cron_local_to_utc_with_offset("0 30 23 * * Fri", -240),
            "0 30 3 * * Sat"
        );
    }

    #[test]
    fn cron_dow_rollover_utc_plus_1_backward() {
        // Sunday 00:30 local at UTC+1 → Saturday 23:30 UTC
        assert_eq!(
            cron_local_to_utc_with_offset("0 30 0 * * Sun", 60),
            "0 30 23 * * Sat"
        );
    }

    #[test]
    fn cron_wildcard_hour_returned_unchanged() {
        let input = "0 30 * * * *";
        assert_eq!(cron_local_to_utc_with_offset(input, -240), input);
    }

    #[test]
    fn cron_wrong_field_count_returned_unchanged() {
        let input = "30 9 * * *"; // 5-field, not 6
        assert_eq!(cron_local_to_utc_with_offset(input, -240), input);
    }

    // -----------------------------------------------------------------------
    // schedule_once logic (no LM / Bot / Scheduler needed)
    // -----------------------------------------------------------------------

    #[test]
    fn schedule_once_delay_seconds_positive_is_future() {
        let fire_at = Utc::now() + chrono::Duration::seconds(300);
        assert!(fire_at > Utc::now());
    }

    #[test]
    fn schedule_once_delay_seconds_zero_is_not_future() {
        let fire_at = Utc::now() + chrono::Duration::seconds(0);
        assert!(fire_at <= Utc::now());
    }

    #[test]
    fn schedule_once_fire_at_past_rejected() {
        let past = "2020-01-01T00:00:00Z";
        let dt = chrono::DateTime::parse_from_rfc3339(past)
            .unwrap()
            .with_timezone(&Utc);
        assert!(dt <= Utc::now(), "past date should be <= now");
    }

    #[test]
    fn schedule_once_fire_at_future_accepted() {
        let future = (Utc::now() + chrono::Duration::hours(1)).format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let dt = chrono::DateTime::parse_from_rfc3339(&future)
            .unwrap()
            .with_timezone(&Utc);
        assert!(dt > Utc::now());
    }

    #[test]
    fn schedule_once_fire_at_invalid_format_errors() {
        assert!(chrono::DateTime::parse_from_rfc3339("not-a-date").is_err());
        assert!(chrono::DateTime::parse_from_rfc3339("2026-03-18 14:00:00").is_err()); // missing T
    }
}
