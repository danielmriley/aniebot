use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const PATH: &str = "data/agenda.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgendaStatus {
    Pending,
    InProgress,
    Done,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgendaItem {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub status: AgendaStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

pub async fn load() -> anyhow::Result<Vec<AgendaItem>> {
    match tokio::fs::read_to_string(PATH).await {
        Ok(contents) => serde_json::from_str(&contents).context("Failed to parse agenda.json"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}

pub async fn save(items: &[AgendaItem]) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(items).context("Failed to serialize agenda")?;
    tokio::fs::write(PATH, json).await.context("Failed to write agenda.json")
}

pub async fn add(description: &str, context: Option<&str>) -> anyhow::Result<AgendaItem> {
    let item = AgendaItem {
        id: uuid::Uuid::new_v4().to_string(),
        created_at: Utc::now(),
        description: description.to_string(),
        context: context.map(|s| s.to_string()),
        status: AgendaStatus::Pending,
        result: None,
        completed_at: None,
    };
    let mut items = load().await?;
    items.push(item.clone());
    save(&items).await?;
    Ok(item)
}

pub async fn complete(id: &str, result: &str) -> anyhow::Result<bool> {
    let mut items = load().await?;
    let Some(item) = items.iter_mut().find(|i| i.id == id) else {
        return Ok(false);
    };
    item.status = AgendaStatus::Done;
    item.result = Some(result.to_string());
    item.completed_at = Some(Utc::now());
    save(&items).await?;
    Ok(true)
}

pub async fn cancel(id: &str) -> anyhow::Result<bool> {
    let mut items = load().await?;
    let Some(item) = items.iter_mut().find(|i| i.id == id) else {
        return Ok(false);
    };
    item.status = AgendaStatus::Cancelled;
    item.completed_at = Some(Utc::now());
    save(&items).await?;
    Ok(true)
}

/// Returns only Pending and InProgress items.
pub async fn list_pending() -> anyhow::Result<Vec<AgendaItem>> {
    let items = load().await?;
    Ok(items
        .into_iter()
        .filter(|i| matches!(i.status, AgendaStatus::Pending | AgendaStatus::InProgress))
        .collect())
}
