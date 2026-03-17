use std::sync::Arc;

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use teloxide::prelude::*;
use uuid::Uuid;

use crate::cli_wrapper;
use crate::config::Config;
use crate::memory::{self, ConversationMessage};
use crate::schedule_store::{self, ScheduleEntry};
use crate::scheduler::{self, SchedulerHandle};

const HISTORY_WINDOW: usize = 20;
const MEMORY_ENTRIES: usize = 5;

// ---------------------------------------------------------------------------
// LM Studio response schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OrchestratorResponse {
    action: String, // "direct_reply" | "delegate_cli" | "schedule_task" | "list_schedules" | "delete_schedule"
    reply: Option<String>,
    task: Option<String>,
    // --- schedule_task fields ---
    cron: Option<String>,
    label: Option<String>,
    schedule_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct LmMessage {
    role: String,
    content: String,
}

// ---------------------------------------------------------------------------
// Prompt tier classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum PromptTier {
    /// User wants to see or remove existing schedules — minimal prompt, no context.
    ScheduleOps,
    /// User wants to create a new recurring task — personality only, no history.
    ScheduleCreate,
    /// Everything else — full context with history and memory.
    Full,
}

fn classify_intent(input: &str) -> PromptTier {
    let s = input.to_lowercase();
    // ScheduleOps: listing or deleting existing schedules
    if s.contains("list schedule")
        || s.contains("show schedule")
        || s.contains("my schedule")
        || s.contains("my job")
        || s.contains("my reminder")
        || (s.contains("delete") && (s.contains("schedule") || s.contains("reminder") || s.contains("job")))
        || (s.contains("remove") && (s.contains("schedule") || s.contains("reminder") || s.contains("job")))
        || s.contains("cancel reminder")
        || s.contains("cancel schedule")
    {
        return PromptTier::ScheduleOps;
    }
    // ScheduleCreate: setting up a new recurring task
    if (s.contains("remind me") || s.contains("schedule"))
        && (s.contains("every")
            || s.contains(" at ")
            || s.contains("daily")
            || s.contains("weekly")
            || s.contains("morning")
            || s.contains("evening")
            || s.contains("each day")
            || s.contains("each week"))
    {
        return PromptTier::ScheduleCreate;
    }
    PromptTier::Full
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
        r#"{personality}

You are executing a scheduled task. Reply with ONLY this JSON shape — no other actions are valid:
{{"action":"direct_reply","reply":"your message here"}}
The task to perform:"#,
        personality = personality,
    );

    let messages = vec![
        LmMessage { role: "system".into(), content: system_prompt },
        LmMessage { role: "user".into(), content: task.to_string() },
    ];

    // Single attempt — no retry loop needed for this simple path.
    let raw = post_to_lm_studio(&config, &messages).await?;

    // Strip think blocks and fences.
    let after_think = if let Some(end) = raw.find("</think>") {
        &raw[end + "</think>".len()..]
    } else {
        &raw
    };
    let cleaned = after_think
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Accept either a valid JSON direct_reply or fall back to raw text.
    if let Ok(parsed) = serde_json::from_str::<OrchestratorResponse>(cleaned) {
        if let Some(reply) = parsed.reply {
            return Ok(reply);
        }
    }

    // Graceful degradation: use raw content as the reply.
    Ok(raw)
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
    // 1. Classify intent to determine how much context to load.
    let tier = classify_intent(user_input);
    tracing::debug!("Prompt tier: {:?}", tier);

    // 2. Load context conditionally based on tier.
    let personality = match tier {
        PromptTier::ScheduleOps => String::new(),
        _ => load_personality().await?,
    };

    let (history, recent_memory) = match tier {
        PromptTier::Full => {
            tokio::try_join!(
                memory::load_history(chat_id),
                memory::load_recent_memory(MEMORY_ENTRIES),
            )?
        }
        _ => (Vec::new(), Vec::new()),
    };

    // 3. Build the windowed history slice for the prompt.
    let window_start = history.len().saturating_sub(HISTORY_WINDOW * 2);
    let history_window = &history[window_start..];

    // 4. Build the system prompt based on tier.
    let system_prompt = match tier {
        PromptTier::ScheduleOps => schedule_ops_prompt(),
        PromptTier::ScheduleCreate => schedule_create_prompt(&personality),
        PromptTier::Full => {
            let memory_bullets: String = if recent_memory.is_empty() {
                String::from("(no prior interactions yet)")
            } else {
                recent_memory
                    .iter()
                    .map(|e| format!("- [{}] User: {} → You: {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.user_msg, e.assistant_reply))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            full_prompt(&personality, &memory_bullets)
        }
    };

    // 5. Build the messages array: system + history window + current user message.
    let mut messages: Vec<LmMessage> = Vec::with_capacity(1 + history_window.len() + 1);
    messages.push(LmMessage { role: "system".into(), content: system_prompt });
    for msg in history_window {
        messages.push(LmMessage { role: msg.role.clone(), content: msg.content.clone() });
    }
    messages.push(LmMessage { role: "user".into(), content: user_input.to_string() });

    // 5. Call LM Studio. Retry once on JSON parse failure.
    let reply = match call_lm_studio(config.clone(), bot, scheduler, &mut messages).await {
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

async fn call_lm_studio(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    messages: &mut Vec<LmMessage>,
) -> anyhow::Result<String> {
    let raw = post_to_lm_studio(&config, messages).await?;

    // First parse attempt.
    match parse_and_dispatch(config.clone(), bot.clone(), scheduler.clone(), &raw).await {
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

    let raw2 = post_to_lm_studio(&config, messages).await?;

    match parse_and_dispatch(config, bot, scheduler, &raw2).await {
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
        "max_tokens": 1024,
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

async fn parse_and_dispatch(
    config: Arc<Config>,
    bot: Bot,
    scheduler: Arc<SchedulerHandle>,
    raw: &str,
) -> anyhow::Result<String> {
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
            match cli_wrapper::run(&config, &task).await {
                Ok(output) => Ok(format!("✅ Done!\n\n{}", output)),
                Err(e) => Ok(format!("❌ CLI error: {}", e)),
            }
        }
        "schedule_task" => {
            let cron = parsed.cron.context("schedule_task missing 'cron' field")?;
            let label = parsed.label.unwrap_or_else(|| "unnamed task".into());
            let task = parsed.task.context("schedule_task missing 'task' field")?;

            // Validate the cron expression with a lightweight field-count check.
            // Full validation happens when the job is registered; this catches
            // obviously malformed input (wrong number of fields) without spawning
            // any scheduler-internal tasks.
            if cron.split_whitespace().count() != 6 {
                return Ok(format!(
                    "❌ Invalid cron expression `{}`: expected 6 fields (seconds minutes hours day month weekday).",
                    cron
                ));
            }

            let entry = ScheduleEntry {
                id: Uuid::new_v4().to_string(),
                label: label.clone(),
                cron: cron.clone(),
                task,
                created_at: Utc::now(),
            };

            let entry_id = entry.id.clone();

            if let Err(e) =
                scheduler::add_dynamic_job(&scheduler, bot.clone(), config.clone(), entry.clone())
                    .await
            {
                return Ok(format!("❌ Failed to register job: {}", e));
            }

            if let Err(e) = schedule_store::append(entry).await {
                tracing::warn!("Failed to persist schedule entry: {}", e);
            }

            Ok(format!(
                "✅ Scheduled *{}* (ID: `{}`)\nCron: `{}`",
                label, entry_id, cron
            ))
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
            let id = parsed
                .schedule_id
                .context("delete_schedule missing 'schedule_id' field")?;

            let removed_live =
                scheduler::remove_dynamic_job(&scheduler, &id).await.unwrap_or(false);
            let removed_store = schedule_store::remove(&id).await.unwrap_or(false);

            if removed_live || removed_store {
                Ok(format!("✅ Deleted schedule `{}`.", id))
            } else {
                Ok(format!("❌ No schedule found with ID `{}`.", id))
            }
        }
        other => anyhow::bail!("Unknown action: {}", other),
    }
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

fn schedule_ops_prompt() -> String {
    r#"You manage the user's scheduled jobs.
You must reply with ONLY a single JSON object — no markdown, no explanation, no code fences.
Allowed actions:
{"action":"list_schedules"}
{"action":"delete_schedule","schedule_id":"<uuid>"}
{"action":"direct_reply","reply":"your reply here"}
Use "list_schedules" when the user wants to see their scheduled jobs.
Use "delete_schedule" when the user wants to remove one (you must have the ID).
Use "direct_reply" for anything else."#
        .to_string()
}

fn schedule_create_prompt(personality: &str) -> String {
    format!(
        r#"{personality}

## Response format
You must reply with ONLY a single JSON object — no markdown, no explanation, no code fences.
Allowed actions:
{{"action":"schedule_task","cron":"0 30 9 * * Mon-Fri","label":"human-readable name","task":"The prompt to run when this job fires."}}
{{"action":"direct_reply","reply":"your reply here"}}
IMPORTANT — cron format: 6 fields in Quartz order: SECONDS MINUTES HOURS DAY-OF-MONTH MONTH DAY-OF-WEEK
Examples: every day at 08:00 = "0 0 8 * * *", weekdays at 09:30 = "0 30 9 * * Mon-Fri", every minute = "0 * * * * *"
Use "schedule_task" to create a new recurring reminder or automated task.
Use "direct_reply" for clarification questions or if the user changes their mind."#,
        personality = personality,
    )
}

fn full_prompt(personality: &str, memory_bullets: &str) -> String {
    format!(
        r#"{personality}

## Recent memory
{memory_bullets}

## Response format
You must reply with ONLY a single JSON object — no markdown, no explanation, no code fences.
Choose one of these action shapes:
{{"action":"direct_reply","reply":"your reply here"}}
{{"action":"delegate_cli","task":"precise task description for copilot"}}
{{"action":"schedule_task","cron":"0 30 9 * * Mon-Fri","label":"human-readable name","task":"The prompt to run when this job fires."}}
{{"action":"list_schedules"}}
{{"action":"delete_schedule","schedule_id":"<uuid>"}}

IMPORTANT — cron format: 6 fields in Quartz order: SECONDS MINUTES HOURS DAY-OF-MONTH MONTH DAY-OF-WEEK
Examples: every day at 08:00 = "0 0 8 * * *", weekdays at 09:30 = "0 30 9 * * Mon-Fri", every minute = "0 * * * * *"

Use "delegate_cli" when the user wants ANY of the following:
- Code written, edited, or explained
- Files created, read, or modified
- Shell commands run
- Real-time or current information: weather, news, stock prices, sports scores, anything that changes over time
- Web searches or looking something up online
- Anything you are unsure about or that requires information beyond your training data
The delegate tool (copilot) has full internet access, web search, and can run shell commands — always prefer delegating over guessing.
Use "direct_reply" ONLY for pure conversation, opinions, or things you are completely certain about from your own knowledge.
Use "schedule_task" when the user wants to set up a recurring reminder or automated task.
Use "list_schedules" when the user asks to see their scheduled jobs.
Use "delete_schedule" when the user wants to remove a scheduled job (look up the ID from list_schedules first if needed).

IMPORTANT routing rules — read these before deciding:
1. If information the user is referring to is already present in the conversation history above, do NOT delegate again. Respond directly using what you already know.
2. If the user is asking for your opinion, reaction, or take on something (e.g. "what do you think?", "do you have an opinion?", "interesting, right?"), always use "direct_reply". Opinions never require a web fetch.
3. If the user is simply acknowledging your previous reply (e.g. "thanks", "got it", "wow", "interesting"), respond conversationally with "direct_reply". Do not re-fetch or re-summarize.
4. NEVER say "I can't access files" or "I don't have filesystem access". You delegate file operations to copilot, which runs in the workspace directory with full filesystem access. Any request to read, write, or summarize a file MUST use "delegate_cli".

## Examples
User: "what time is it?"
{{"action":"delegate_cli","task":"Run the `date` command and report the current date and time."}}

User: "what's the weather tomorrow in Apopka Florida?"
{{"action":"delegate_cli","task":"Search the web for the weather forecast for tomorrow in Apopka, Florida and summarize it."}}

User: "add error handling to main.rs"
{{"action":"delegate_cli","task":"Add proper error handling to main.rs in the current workspace. Use anyhow for error propagation."}}

User: "can you read the design_plan2.md file and summarize it?"
{{"action":"delegate_cli","task":"Read the file docs/design_plan2.md in the workspace directory and summarize its contents."}}

User: "can you access files in my workspace?"
{{"action":"direct_reply","reply":"Yes — I delegate file reads and writes to copilot, which runs in the workspace directory with full access."}}

User: "how are you?"
{{"action":"direct_reply","reply":"Functioning within normal parameters. What do you need?"}}

User: "thanks, that's a lot going on!"
{{"action":"direct_reply","reply":"Yeah, it's a busy news day. Anything in particular catch your eye?"}}

User: "what do you think of all of this?"
{{"action":"direct_reply","reply":"My take: ..."}}

User: "do you have an opinion on those stories?"
{{"action":"direct_reply","reply":"A few thoughts: ..."}}

User: "remind me every weekday at 9:30 to do standup"
{{"action":"schedule_task","cron":"0 30 9 * * Mon-Fri","label":"standup reminder","task":"Remind me to do my standup. Be brief and direct."}}

User: "show my scheduled jobs"
{{"action":"list_schedules"}}

User: "delete the standup reminder"
{{"action":"delete_schedule","schedule_id":"<uuid-from-list>"}}
"#,
        personality = personality,
        memory_bullets = memory_bullets,
    )
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
