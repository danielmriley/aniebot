use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const CONVERSATIONS_DIR: &str = "data/conversations";
const MEMORY_FILE: &str = "data/memory.json";

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct ConversationMessage {
    pub role: String,
    // null for assistant turns that issued a tool call
    pub content: Option<String>,
    // set on role=="assistant" tool-call turns
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    // set on role=="tool" turns
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
pub struct MemoryEntry {
    pub chat_id: i64,
    pub user_msg: String,
    pub assistant_reply: String,
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Conversation History
// ---------------------------------------------------------------------------

pub async fn load_history(chat_id: i64) -> anyhow::Result<Vec<ConversationMessage>> {
    let path = format!("{}/{}.json", CONVERSATIONS_DIR, chat_id);
    match tokio::fs::read_to_string(&path).await {
        Ok(contents) => match serde_json::from_str(&contents) {
            Ok(messages) => Ok(messages),
            Err(e) => {
                tracing::warn!("Conversation file for {} is corrupt, starting fresh: {}", chat_id, e);
                Ok(vec![])
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}

pub async fn append_messages(
    chat_id: i64,
    messages: &[ConversationMessage],
) -> anyhow::Result<()> {
    let path = format!("{}/{}.json", CONVERSATIONS_DIR, chat_id);
    let tmp_path = format!("{}/{}.json.tmp", CONVERSATIONS_DIR, chat_id);
    let mut history = load_history(chat_id).await?;
    history.extend_from_slice(messages);
    tokio::fs::write(&tmp_path, serde_json::to_vec_pretty(&history)?).await?;
    tokio::fs::rename(&tmp_path, &path).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Global Memory Log
// ---------------------------------------------------------------------------

pub async fn store_interaction(
    chat_id: i64,
    user_msg: &str,
    assistant_reply: &str,
) -> anyhow::Result<()> {
    let entry = MemoryEntry {
        chat_id,
        user_msg: user_msg.to_string(),
        assistant_reply: assistant_reply.to_string(),
        timestamp: Utc::now(),
    };

    let mut log = load_memory_log().await?;
    log.push(entry);
    let tmp = format!("{}.tmp", MEMORY_FILE);
    tokio::fs::write(&tmp, serde_json::to_vec_pretty(&log)?).await?;
    tokio::fs::rename(&tmp, MEMORY_FILE).await?;
    Ok(())
}

pub async fn load_recent_memory(n: usize) -> anyhow::Result<Vec<MemoryEntry>> {
    let log = load_memory_log().await?;
    let start = log.len().saturating_sub(n);
    Ok(log.into_iter().skip(start).collect())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async fn load_memory_log() -> anyhow::Result<Vec<MemoryEntry>> {
    match tokio::fs::read_to_string(MEMORY_FILE).await {
        Ok(contents) => match serde_json::from_str(&contents) {
            Ok(entries) => Ok(entries),
            Err(e) => {
                tracing::warn!("Memory file is corrupt, starting fresh: {}", e);
                Ok(vec![])
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}
