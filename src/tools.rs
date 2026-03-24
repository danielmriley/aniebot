use std::sync::Arc;

use anyhow::Context;
use chrono::{Local, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;
use uuid::Uuid;

use crate::config::Config;
use crate::core_memory;
use crate::episodic;
use crate::schedule_store::{self, ScheduleEntry};
use crate::scheduler::{self, SchedulerHandle};

// ---------------------------------------------------------------------------
// Tool metadata — used by AgentSession for parallelism decisions
// ---------------------------------------------------------------------------

pub struct ToolDef {
    pub name: &'static str,
    pub parallel_safe: bool,
}

pub const ALL_TOOL_METADATA: &[ToolDef] = &[
    ToolDef { name: "send_update",          parallel_safe: false },
    ToolDef { name: "final_reply",          parallel_safe: false },
    ToolDef { name: "reflect",              parallel_safe: false },
    ToolDef { name: "read_file",            parallel_safe: true  },
    ToolDef { name: "write_file",           parallel_safe: false },
    ToolDef { name: "edit_file",            parallel_safe: false },
    ToolDef { name: "shell_command",        parallel_safe: false },
    ToolDef { name: "list_dir",             parallel_safe: true  },
    ToolDef { name: "fetch_page",           parallel_safe: true  },
    ToolDef { name: "schedule_task",        parallel_safe: false },
    ToolDef { name: "list_schedules",       parallel_safe: true  },
    ToolDef { name: "delete_schedule",      parallel_safe: false },
    ToolDef { name: "update_schedule",      parallel_safe: false },
    ToolDef { name: "schedule_once",        parallel_safe: false },
    ToolDef { name: "update_core_memory",   parallel_safe: false },
    ToolDef { name: "read_core_memory",     parallel_safe: true  },
    ToolDef { name: "remember",             parallel_safe: false },
    ToolDef { name: "recall",               parallel_safe: true  },
    ToolDef { name: "forget",               parallel_safe: false },
    ToolDef { name: "list_episodic_recent", parallel_safe: true  },
    ToolDef { name: "add_interest",         parallel_safe: false },
    ToolDef { name: "retire_interest",      parallel_safe: false },
    ToolDef { name: "list_interests",       parallel_safe: true  },
    ToolDef { name: "set_task",             parallel_safe: false },
    ToolDef { name: "add_agenda_item",      parallel_safe: false },
    ToolDef { name: "list_agenda_items",    parallel_safe: true  },
    ToolDef { name: "update_agenda_item",   parallel_safe: false },
    ToolDef { name: "cancel_agenda_item",   parallel_safe: false },
    ToolDef { name: "complete_agenda_item", parallel_safe: false },
    ToolDef { name: "get_current_time",     parallel_safe: true  },
];

pub fn is_parallel_safe(name: &str) -> bool {
    ALL_TOOL_METADATA.iter().any(|t| t.name == name && t.parallel_safe)
}

// ---------------------------------------------------------------------------
// Tool context — determines which tools are available in each execution mode
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ToolContext {
    User,
    Heartbeat,
    InterestEval,
    Consolidation,
}

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
// Tool helpers — one function per tool; composed into context sets below
// ---------------------------------------------------------------------------

fn tool_read_file() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns up to `limit` lines starting from `offset`. Omit both to read the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the file." },
                    "offset": { "type": "integer", "description": "Line number to start reading from (1-based, optional)." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (optional)." }
                },
                "required": ["path"]
            }
        }
    })
}

fn tool_write_file() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it or overwriting it entirely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the file." },
                    "content": { "type": "string", "description": "Content to write." }
                },
                "required": ["path", "content"]
            }
        }
    })
}

fn tool_edit_file() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Make a targeted edit to a file by replacing an exact string. Read the file first to get the exact text to replace. Fails if old_string is not found — use the exact verbatim text as it appears in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the file." },
                    "old_string": { "type": "string", "description": "Exact text to find and replace. Must appear verbatim in the file." },
                    "new_string": { "type": "string", "description": "Replacement text." },
                    "replace_all": { "type": "boolean", "description": "If true, replace all occurrences instead of just the first (default false)." }
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    })
}

fn tool_shell_command() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Run a shell command and return combined stdout+stderr. Use for git, grep, curl, python scripts, etc. Output is capped at 16 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": { "type": "string", "description": "Shell command to execute (passed to sh -c)." },
                    "cwd": { "type": "string", "description": "Working directory (optional; defaults to the bot's working directory)." },
                    "timeout_secs": { "type": "integer", "description": "Seconds before the command is killed. Defaults to the server-configured SHELL_COMMAND_TIMEOUT_SECS. Use a higher value for builds, tests, or long-running scripts (max 600)." }
                },
                "required": ["cmd"]
            }
        }
    })
}

fn tool_list_dir() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory. Subdirectory names are shown with a trailing '/'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the directory." }
                },
                "required": ["path"]
            }
        }
    })
}

fn tool_fetch_page() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page. mode='raw' returns the raw response body (up to 8 KB); mode='text' returns a clean markdown rendering via Jina Reader — use this for articles, documentation, or any JS-rendered page. Both modes accept any public HTTP/HTTPS URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The URL to fetch (must be HTTP or HTTPS)." },
                    "mode": {
                        "type": "string",
                        "enum": ["raw", "text"],
                        "description": "raw = plain HTTP response body; text = cleaned markdown via Jina Reader (recommended for most pages)."
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
            "description": "Send a progress message to the user now and keep working. Use this to keep the user informed during long tasks — for example, after completing one step and before starting the next. You must still call final_reply or reflect(done=true) when you are done.",
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

fn tool_final_reply() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "final_reply",
            "description": "Send a complete, polished reply to the user and end this session immediately. Use this when you have finished all your work and are ready to present results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The complete reply to send to the user."
                    }
                },
                "required": ["text"]
            }
        }
    })
}

fn tool_set_task() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "set_task",
            "description": "Record the current task or active project. Appears in every system prompt. Pass an empty string to clear the current task once it is complete or no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of the current task (1-2 sentences). Pass \"\" to clear."
                    }
                },
                "required": ["description"]
            }
        }
    })
}

fn tool_read_core_memory() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "read_core_memory",
            "description": "Read the current state of your core memory (identity, beliefs, user profile, curiosity queue). Useful mid-session after updates to verify what was written.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

fn tool_list_episodic_recent() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "list_episodic_recent",
            "description": "Return the N most-recent episodic memory entries, ordered oldest-first. Useful for consolidation or reviewing recent activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": { "type": "integer", "description": "Number of recent entries to return (default 10, max 50)." }
                }
            }
        }
    })
}

fn tool_update_agenda_item() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "update_agenda_item",
            "description": "Update the status or append a note to a pending agenda item. Use status 'in_progress' when starting work on an item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": { "type": "string", "description": "UUID of the agenda item to update." },
                    "status": { "type": "string", "enum": ["pending", "in_progress"], "description": "New status (optional)." },
                    "note": { "type": "string", "description": "Note to append to the item's context (optional)." }
                },
                "required": ["item_id"]
            }
        }
    })
}

fn tool_get_current_time() -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in both local time and UTC. Use this when you need the current timestamp for scheduling calculations or time-sensitive decisions.",
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
                        "description": "Optional draft reply for the user. When done=true in a user conversation, this is passed to the synthesis step as a starting point. Omit to let the harness compose the reply from your tool results. Ignored in background contexts (heartbeat, consolidation)."
                    }
                },
                "required": ["observation", "done"]
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Tool sets by context (JSON Schema arrays sent to LM Studio)
// ---------------------------------------------------------------------------

pub fn tools_for_context(ctx: ToolContext) -> serde_json::Value {
    match ctx {
        ToolContext::User => json!([
            tool_read_file(),
            tool_write_file(),
            tool_edit_file(),
            tool_shell_command(),
            tool_list_dir(),
            tool_fetch_page(),
            tool_schedule_task(),
            tool_list_schedules(),
            tool_delete_schedule(),
            tool_update_schedule(),
            tool_schedule_once(),
            tool_send_update(),
            tool_final_reply(),
            tool_update_core_memory(false),
            tool_read_core_memory(),
            tool_remember(),
            tool_recall(),
            tool_forget(),
            tool_list_episodic_recent(),
            tool_list_interests(),
            tool_set_task(),
            tool_reflect(),
            tool_add_agenda_item(),
            tool_list_agenda_items(),
            tool_update_agenda_item(),
            tool_cancel_agenda_item(),
            tool_get_current_time(),
        ]),
        ToolContext::Heartbeat => json!([
            tool_read_file(),
            tool_write_file(),
            tool_edit_file(),
            tool_shell_command(),
            tool_list_dir(),
            tool_fetch_page(),
            tool_send_update(),
            tool_final_reply(),
            tool_add_interest(),
            tool_retire_interest(),
            tool_update_core_memory(false),
            tool_read_core_memory(),
            tool_recall(),
            tool_forget(),
            tool_list_episodic_recent(),
            tool_set_task(),
            tool_reflect(),
            tool_add_agenda_item(),
            tool_list_agenda_items(),
            tool_update_agenda_item(),
            tool_cancel_agenda_item(),
            tool_complete_agenda_item(),
            tool_get_current_time(),
        ]),
        ToolContext::InterestEval => json!([
            tool_add_interest(),
            tool_retire_interest(),
            tool_list_interests(),
            tool_reflect(),
        ]),
        ToolContext::Consolidation => json!([
            tool_update_core_memory(true),
            tool_read_core_memory(),
            tool_read_file(),
            tool_write_file(),
            tool_edit_file(),
            tool_fetch_page(),
            tool_final_reply(),
            tool_remember(),
            tool_recall(),
            tool_forget(),
            tool_list_episodic_recent(),
            tool_add_interest(),
            tool_retire_interest(),
            tool_set_task(),
            tool_reflect(),
            tool_add_agenda_item(),
            tool_list_agenda_items(),
            tool_update_agenda_item(),
            tool_cancel_agenda_item(),
            tool_complete_agenda_item(),
            tool_get_current_time(),
        ]),
    }
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

async fn fetch_page_impl(url: &str, mode: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(45))
        .user_agent("AnieBot/1.0")
        .build()
        .context("Failed to build HTTP client")?;
    let (effective_url, cap) = if mode == "text" {
        (format!("https://r.jina.ai/{}", url), 32 * 1024)
    } else {
        (url.to_string(), 8 * 1024)
    };
    let response = client
        .get(&effective_url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;
    let status = response.status();
    let bytes = response.bytes().await.map_err(|e| anyhow::anyhow!("Failed to read response: {}", e))?;
    let capped = bytes.len().min(cap);
    let text = String::from_utf8_lossy(&bytes[..capped]).into_owned();
    if mode == "text" {
        Ok(text)
    } else {
        Ok(format!("HTTP {}\n\n{}", status, text))
    }
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
        "final_reply" => {
            let text = args["text"].as_str().unwrap_or("").to_string();
            Ok(text)
        }
        "read_file" => {
            let path = args["path"].as_str().context("read_file missing 'path'")?;
            tracing::info!("read_file: {}", path);
            match tokio::fs::read_to_string(path).await {
                Ok(content) => {
                    let lines: Vec<&str> = content.lines().collect();
                    let offset = args["offset"].as_u64().map(|n| (n as usize).saturating_sub(1)).unwrap_or(0);
                    let limit = args["limit"].as_u64().map(|n| n as usize).unwrap_or(usize::MAX);
                    let slice: Vec<String> = lines.iter().skip(offset).take(limit)
                        .enumerate()
                        .map(|(i, l)| format!("{}: {}", offset + i + 1, l))
                        .collect();
                    Ok(slice.join("\n"))
                }
                Err(e) => Ok(format!("❌ read_file failed: {}", e)),
            }
        }
        "write_file" => {
            let path = args["path"].as_str().context("write_file missing 'path'")?;
            let content = args["content"].as_str().context("write_file missing 'content'")?;
            tracing::info!("write_file: {}", path);
            // Create parent directories if needed.
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
            }
            match tokio::fs::write(path, content).await {
                Ok(()) => Ok(format!("✅ Written {} bytes to {}", content.len(), path)),
                Err(e) => Ok(format!("❌ write_file failed: {}", e)),
            }
        }
        "edit_file" => {
            let path = args["path"].as_str().context("edit_file missing 'path'")?;
            let old_string = args["old_string"].as_str().context("edit_file missing 'old_string'")?;
            let new_string = args["new_string"].as_str().context("edit_file missing 'new_string'")?;
            let replace_all = args["replace_all"].as_bool().unwrap_or(false);
            tracing::info!("edit_file: {}", path);
            match tokio::fs::read_to_string(path).await {
                Err(e) => Ok(format!("❌ edit_file failed to read {}: {}", path, e)),
                Ok(content) => {
                    let count = content.matches(old_string).count();
                    if count == 0 {
                        return Ok(format!(
                            "❌ old_string not found in {}. Read the file first and use the exact text.",
                            path
                        ));
                    }
                    let new_content = if replace_all {
                        content.replace(old_string, new_string)
                    } else {
                        content.replacen(old_string, new_string, 1)
                    };
                    // Atomic write: write to .tmp then rename.
                    let tmp_path = format!("{}.tmp", path);
                    if let Err(e) = tokio::fs::write(&tmp_path, &new_content).await {
                        return Ok(format!("❌ edit_file failed to write tmp file: {}", e));
                    }
                    if let Err(e) = tokio::fs::rename(&tmp_path, path).await {
                        let _ = tokio::fs::remove_file(&tmp_path).await;
                        return Ok(format!("❌ edit_file failed to rename tmp file: {}", e));
                    }
                    if !replace_all && count > 1 {
                        Ok(format!(
                            "✅ edit applied to {} (replaced first of {} occurrences — use replace_all=true to replace all)",
                            path, count
                        ))
                    } else {
                        Ok(format!("✅ edit applied to {}", path))
                    }
                }
            }
        }
        "shell_command" => {
            let cmd = args["cmd"].as_str().context("shell_command missing 'cmd'")?;
            let cwd = args["cwd"].as_str();
            let default_timeout = config.shell_command_timeout_secs;
            let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(default_timeout).min(600);
            tracing::info!("shell_command (timeout={}s): {}", timeout_secs, cmd);
            let mut command = tokio::process::Command::new("sh");
            command.arg("-c").arg(cmd)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            if let Some(dir) = cwd {
                command.current_dir(dir);
            }
            let timeout = std::time::Duration::from_secs(timeout_secs);
            match tokio::time::timeout(timeout, command.output()).await {
                Err(_) => Ok(format!("❌ shell_command timed out after {}s", timeout_secs)),
                Ok(Err(e)) => Ok(format!("❌ shell_command failed: {}", e)),
                Ok(Ok(out)) => {
                    let cap = 16 * 1024;
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let combined = if stderr.trim().is_empty() {
                        stdout.to_string()
                    } else if stdout.trim().is_empty() {
                        stderr.to_string()
                    } else {
                        format!("{}\n--- stderr ---\n{}", stdout, stderr)
                    };
                    let capped = if combined.len() > cap { &combined[..cap] } else { &combined };
                    let exit = out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into());
                    Ok(format!("exit={}\n{}", exit, capped))
                }
            }
        }
        "list_dir" => {
            let path = args["path"].as_str().context("list_dir missing 'path'")?;
            tracing::info!("list_dir: {}", path);
            match tokio::fs::read_dir(path).await {
                Ok(mut dir) => {
                    let mut entries: Vec<String> = Vec::new();
                    while let Ok(Some(entry)) = dir.next_entry().await {
                        let name = entry.file_name().to_string_lossy().to_string();
                        let suffix = if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) { "/" } else { "" };
                        entries.push(format!("{}{}", name, suffix));
                    }
                    entries.sort();
                    Ok(entries.join("\n"))
                }
                Err(e) => Ok(format!("❌ list_dir failed: {}", e)),
            }
        }
        "fetch_page" => {
            let url = args["url"].as_str().context("fetch_page missing 'url'")?;
            let mode = args["mode"].as_str().unwrap_or("raw");
            tracing::info!("fetch_page: {} (mode={})", url, mode);
            match fetch_page_impl(url, mode).await {
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
        "reflect" => {
            let observation = args["observation"].as_str().unwrap_or("(no observation)");
            let done = args["done"].as_bool().unwrap_or(false);
            Ok(format!("Reflection recorded: {observation} (concluded: {done})"))
        }
        "read_core_memory" => {
            let core = core_memory::load().await?;
            Ok(core.to_prompt_block())
        }
        "list_episodic_recent" => {
            let n = args["n"].as_u64().unwrap_or(10).min(50) as usize;
            let entries = episodic::load_recent(n).await?;
            if entries.is_empty() {
                Ok("No episodic entries yet.".into())
            } else {
                let lines: Vec<String> = entries.iter().map(|e| {
                    format!("[{}] (importance: {}, id: {}) {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.importance, e.id, e.content)
                }).collect();
                Ok(lines.join("\n"))
            }
        }
        "get_current_time" => {
            let local_now = Local::now();
            let utc_now = Utc::now();
            Ok(format!("Local: {}\nUTC:   {}", local_now.format("%Y-%m-%d %H:%M:%S %Z"), utc_now.format("%Y-%m-%dT%H:%M:%SZ")))
        }
        "set_task" => {
            let desc = args["description"].as_str()
                .context("set_task missing 'description'")?;
            if desc.is_empty() {
                core_memory::clear_task().await?;
                Ok("✅ Current task cleared.".to_string())
            } else {
                core_memory::set_task(desc).await?;
                Ok(format!("✅ Current task set: {desc}"))
            }
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
                Ok(format!("✅ Task `{}` cancelled.", id))
            } else {
                Ok(format!("❌ No agenda item found with ID `{}`.", id))
            }
        }
        "update_agenda_item" => {
            let id = args["item_id"].as_str()
                .context("update_agenda_item missing 'item_id'")?;
            let status = args["status"].as_str().and_then(|s| match s {
                "pending" => Some(crate::agenda::AgendaStatus::Pending),
                "in_progress" => Some(crate::agenda::AgendaStatus::InProgress),
                _ => None,
            });
            let note = args["note"].as_str();
            if crate::agenda::update(id, status, note).await? {
                Ok(format!("✅ Agenda item `{}` updated.", id))
            } else {
                Ok(format!("❌ No agenda item found with ID `{}`.", id))
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
