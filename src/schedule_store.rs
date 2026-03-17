use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const PATH: &str = "data/schedules.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleEntry {
    pub id: String,
    pub label: String,
    pub cron: String,
    pub task: String,
    pub created_at: DateTime<Utc>,
}

pub async fn load() -> anyhow::Result<Vec<ScheduleEntry>> {
    match tokio::fs::read_to_string(PATH).await {
        Ok(contents) => serde_json::from_str(&contents).context("Failed to parse schedules.json"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}

pub async fn save(entries: &[ScheduleEntry]) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(entries).context("Failed to serialize schedules")?;
    tokio::fs::write(PATH, json).await.context("Failed to write schedules.json")
}

pub async fn append(entry: ScheduleEntry) -> anyhow::Result<()> {
    let mut entries = load().await?;
    entries.push(entry);
    save(&entries).await
}

/// Remove an entry by its stable ID. Returns `true` if the entry was found and removed.
pub async fn remove(id: &str) -> anyhow::Result<bool> {
    let mut entries = load().await?;
    let before = entries.len();
    entries.retain(|e| e.id != id);
    let removed = entries.len() < before;
    if removed {
        save(&entries).await?;
    }
    Ok(removed)
}
