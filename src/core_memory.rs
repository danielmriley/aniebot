use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const CORE_MEMORY_FILE: &str = "data/core_memory.json";
const CORE_MEMORY_TMP: &str = "data/core_memory.json.tmp";

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct CoreMemory {
    pub identity: String,
    pub beliefs: Vec<String>,
    pub user_profile: String,
    /// Active interests — used by the heartbeat only, NOT injected into the
    /// conversation system prompt to avoid cluttering the context.
    pub interests: Vec<Interest>,
    pub curiosity_queue: Vec<String>,
    /// Timestamp of the last consolidation pass. `None` means never consolidated.
    #[serde(default)]
    pub last_consolidation_at: Option<chrono::DateTime<Utc>>,
    /// Current active task or project being worked on. Injected into every system prompt.
    #[serde(default)]
    pub current_task: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Interest {
    pub id: String,
    pub topic: String,
    pub description: String,
    pub check_cron: Option<String>,
    /// Health score 0-100; decays during consolidation when topic stops appearing.
    pub health: u8,
    pub last_seen_date: String,
}

impl Default for CoreMemory {
    fn default() -> Self {
        Self {
            identity: String::new(),
            beliefs: Vec::new(),
            user_profile: "(no profile yet — will be populated as I learn about you)".into(),
            interests: Vec::new(),
            curiosity_queue: Vec::new(),
            last_consolidation_at: None,
            current_task: None,
        }
    }
}

impl CoreMemory {
    /// Format all sections for injection into the conversation system prompt.
    /// `interests` is intentionally excluded (heartbeat-only).
    pub fn to_prompt_block(&self) -> String {
        let beliefs = if self.beliefs.is_empty() {
            "  (none yet)".to_string()
        } else {
            self.beliefs
                .iter()
                .map(|b| format!("- {b}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let curiosity = if self.curiosity_queue.is_empty() {
            "  (nothing yet)".to_string()
        } else {
            self.curiosity_queue
                .iter()
                .map(|c| format!("- {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let task_block = match &self.current_task {
            Some(t) => format!("\n\n## Current Task\n{t}"),
            None => String::new(),
        };

        format!(
            "## What I Am\n\
             I am AnieBot, a personal AI assistant running inside a custom Rust/tokio harness built by Daniel. \
             I communicate with Daniel via Telegram. Inference is handled by a local LM Studio instance. \
             My available tools are: recall, remember, delegate_cli, schedule_once, schedule_cron, \
             set_task, clear_task, reflect, add_interest, retire_interest, update_core_memory, nothing.\n\n\
             ## Tone & Style\n\
             Be concise and direct. Do not append unsolicited summaries to messages Daniel did not ask to be summarized. \
             A brief acknowledgment like \"Noted.\" is fine; do not repeat back what was just said.\n\n\
             ## Who I Am\n{identity}{task_block}\n\n\
             ## Current Beliefs\n{beliefs}\n\n\
             ## What I Know About You\n{user_profile}\n\n\
             ## Things I'm Curious About\n{curiosity}",
            identity = self.identity,
            task_block = task_block,
            beliefs = beliefs,
            user_profile = self.user_profile,
            curiosity = curiosity,
        )
    }

    /// Format active interests for injection into the heartbeat system prompt.
    pub fn interests_block(&self) -> String {
        if self.interests.is_empty() {
            return "(none)".to_string();
        }
        self.interests
            .iter()
            .map(|i| {
                let cron = i.check_cron.as_deref().unwrap_or("global heartbeat");
                format!("- {} [id: {}]: {} (check: {})", i.topic, i.id, i.description, cron)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Load core memory from disk. Returns `CoreMemory::default()` if the file
/// does not exist yet (first run before seeding).
pub async fn load() -> Result<CoreMemory> {
    match tokio::fs::read_to_string(CORE_MEMORY_FILE).await {
        Ok(content) => {
            serde_json::from_str(&content).context("Failed to deserialize core_memory.json")
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(CoreMemory::default()),
        Err(e) => Err(e.into()),
    }
}

/// Atomically write core memory to disk (tmp → rename).
pub async fn save(cm: &CoreMemory) -> Result<()> {
    let json = serde_json::to_string_pretty(cm).context("Failed to serialize CoreMemory")?;
    tokio::fs::write(CORE_MEMORY_TMP, &json).await
        .context("Failed to write core_memory.json.tmp")?;
    tokio::fs::rename(CORE_MEMORY_TMP, CORE_MEMORY_FILE).await
        .context("Failed to rename core_memory.json.tmp → core_memory.json")?;
    Ok(())
}

/// Load, mutate one section, and save. Called by the `update_core_memory` tool.
pub async fn update_section(section: &str, content: &str) -> Result<()> {
    let mut cm = load().await?;
    match section {
        "identity" => cm.identity = content.to_string(),
        "user_profile" => cm.user_profile = content.to_string(),
        "beliefs" => {
            cm.beliefs = serde_json::from_str::<Vec<String>>(content).context(
                r#"beliefs content must be a JSON array of strings, e.g. ["item1","item2"]"#,
            )?;
        }
        "curiosity_queue" => {
            cm.curiosity_queue = serde_json::from_str::<Vec<String>>(content).context(
                r#"curiosity_queue content must be a JSON array of strings, e.g. ["item1","item2"]"#,
            )?;
        }
        other => anyhow::bail!("Unknown core memory section: {other}"),
    }
    save(&cm).await
}

/// Create a new interest, append it to core memory, and return it.
pub async fn add_interest(
    topic: &str,
    description: &str,
    check_cron: Option<String>,
) -> Result<Interest> {
    let mut cm = load().await?;
    let interest = Interest {
        id: Uuid::new_v4().to_string(),
        topic: topic.to_string(),
        description: description.to_string(),
        check_cron,
        health: 100,
        last_seen_date: Utc::now().date_naive().to_string(),
    };
    cm.interests.push(interest.clone());
    save(&cm).await?;
    Ok(interest)
}

/// Remove an interest by ID from core memory.
pub async fn retire_interest(id: &str) -> Result<()> {
    let mut cm = load().await?;
    cm.interests.retain(|e| e.id != id);
    save(&cm).await
}

/// Set the current active task description.
pub async fn set_task(description: &str) -> Result<()> {
    let mut cm = load().await?;
    cm.current_task = Some(description.to_string());
    save(&cm).await
}

/// Clear the current active task.
pub async fn clear_task() -> Result<()> {
    let mut cm = load().await?;
    cm.current_task = None;
    save(&cm).await
}
