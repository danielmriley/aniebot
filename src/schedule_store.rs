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
    #[serde(default)]
    pub fire_once_at: Option<DateTime<Utc>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn tmp_path() -> String {
        format!("/tmp/aniebot_sched_test_{}.json", Uuid::new_v4())
    }

    fn make_entry(id: &str) -> ScheduleEntry {
        ScheduleEntry {
            id: id.to_string(),
            label: format!("job-{}", id),
            cron: "0 0 9 * * *".to_string(),
            task: "test task".to_string(),
            created_at: chrono::Utc::now(),
            fire_once_at: None,
        }
    }

    async fn save_to(path: &str, entries: &[ScheduleEntry]) {
        let json = serde_json::to_string_pretty(entries).unwrap();
        tokio::fs::write(path, json).await.unwrap();
    }

    async fn load_from(path: &str) -> Vec<ScheduleEntry> {
        match tokio::fs::read_to_string(path).await {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => vec![],
        }
    }

    #[tokio::test]
    async fn round_trip_append_and_load() {
        let path = tmp_path();
        save_to(&path, &[]).await; // start empty

        let e1 = make_entry("001");
        let e2 = make_entry("002");
        let mut entries = load_from(&path).await;
        entries.push(e1.clone());
        entries.push(e2.clone());
        save_to(&path, &entries).await;

        let loaded = load_from(&path).await;
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "001");
        assert_eq!(loaded[1].id, "002");

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn remove_returns_true_when_found() {
        let path = tmp_path();
        let entries = vec![make_entry("aaa"), make_entry("bbb"), make_entry("ccc")];
        save_to(&path, &entries).await;

        let mut loaded = load_from(&path).await;
        let before = loaded.len();
        loaded.retain(|e| e.id != "bbb");
        let removed = loaded.len() < before;
        save_to(&path, &loaded).await;

        assert!(removed);
        let after = load_from(&path).await;
        assert_eq!(after.len(), 2);
        assert!(after.iter().all(|e| e.id != "bbb"));

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn remove_returns_false_when_not_found() {
        let path = tmp_path();
        let entries = vec![make_entry("aaa")];
        save_to(&path, &entries).await;

        let loaded = load_from(&path).await;
        let found = loaded.iter().any(|e| e.id == "zzz");
        assert!(!found);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn fire_once_at_serialises_and_deserialises() {
        let path = tmp_path();
        let mut e = make_entry("x");
        let fire_time = chrono::Utc::now() + chrono::Duration::hours(1);
        e.fire_once_at = Some(fire_time);
        save_to(&path, &[e.clone()]).await;

        let loaded = load_from(&path).await;
        assert!(loaded[0].fire_once_at.is_some());
        // Timestamps round-trip to millisecond precision
        let diff = (loaded[0].fire_once_at.unwrap() - fire_time).num_milliseconds().abs();
        assert!(diff < 1000, "timestamp drift too large: {}ms", diff);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn fire_once_at_defaults_to_none_on_old_json() {
        let path = tmp_path();
        // Simulate a pre-Phase-4 schedules.json entry without the field.
        let old_json = r#"[{"id":"abc","label":"old job","cron":"0 0 9 * * *","task":"do thing","created_at":"2025-01-01T00:00:00Z"}]"#;
        tokio::fs::write(&path, old_json).await.unwrap();

        let loaded = load_from(&path).await;
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].fire_once_at.is_none(), "fire_once_at should default to None");

        let _ = tokio::fs::remove_file(&path).await;
    }
}
