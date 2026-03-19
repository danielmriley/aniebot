use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

const EPISODIC_FILE: &str = "data/episodic.jsonl";
const ARCHIVAL_FILE: &str = "data/archival.jsonl";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct EpisodicEntry {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    /// Importance 1–5 (5 = most important).
    pub importance: u8,
    pub timestamp: DateTime<Utc>,
    /// Reserved for Phase 4 — promotion to archival.
    pub promoted: bool,
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Append a single entry to the episodic JSONL file.
pub async fn append(entry: &EpisodicEntry) -> Result<()> {
    let line = serde_json::to_string(entry)? + "\n";
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(EPISODIC_FILE)
        .await?;
    file.write_all(line.as_bytes()).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

async fn read_jsonl(path: &str) -> Result<Vec<EpisodicEntry>> {
    match tokio::fs::read_to_string(path).await {
        Ok(text) => {
            let entries = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str::<EpisodicEntry>(l).ok())
                .collect();
            Ok(entries)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}

/// Return the `n` most-recent episodic entries.
pub async fn load_recent(n: usize) -> Result<Vec<EpisodicEntry>> {
    let entries = read_jsonl(EPISODIC_FILE).await?;
    let start = entries.len().saturating_sub(n);
    Ok(entries[start..].to_vec())
}

/// Count entries with `timestamp >= dt`.
pub async fn count_since(dt: DateTime<Utc>) -> Result<usize> {
    let entries = read_jsonl(EPISODIC_FILE).await?;
    Ok(entries.iter().filter(|e| e.timestamp >= dt).count())
}

/// Search both episodic and archival JSONL files by keyword / tags.
/// Returns up to 10 matches, sorted by importance desc then timestamp desc.
pub async fn search(query: &str, tags: &[String]) -> Result<Vec<EpisodicEntry>> {
    let mut all = read_jsonl(EPISODIC_FILE).await?;
    all.extend(read_jsonl(ARCHIVAL_FILE).await?);

    let query_lower = query.to_lowercase();
    let mut matches: Vec<EpisodicEntry> = all
        .into_iter()
        .filter(|e| {
            let content_match = !query_lower.is_empty()
                && e.content.to_lowercase().contains(&query_lower);
            let tag_match = tags.iter().any(|t| e.tags.contains(t));
            content_match || tag_match
        })
        .collect();

    matches.sort_by(|a, b| {
        b.importance
            .cmp(&a.importance)
            .then_with(|| b.timestamp.cmp(&a.timestamp))
    });
    matches.truncate(10);
    Ok(matches)
}
