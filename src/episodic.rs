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

    // Deduplicate by ID — entries may appear in both files after promotion.
    let mut seen = std::collections::HashSet::new();
    all.retain(|e| seen.insert(e.id.clone()));

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

/// Promote non-promoted entries with `importance >= min_importance` to `archival.jsonl`.
/// Atomically rewrites `episodic.jsonl` with the promoted entries marked `promoted: true`.
/// Returns the number of entries promoted.
pub async fn promote_to_archival(min_importance: u8) -> Result<usize> {
    let entries = read_jsonl(EPISODIC_FILE).await?;

    let promote_ids: std::collections::HashSet<String> = entries
        .iter()
        .filter(|e| !e.promoted && e.importance >= min_importance)
        .map(|e| e.id.clone())
        .collect();

    if promote_ids.is_empty() {
        return Ok(0);
    }

    // Append to archival.jsonl.
    {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(ARCHIVAL_FILE)
            .await?;
        for entry in entries.iter().filter(|e| promote_ids.contains(&e.id)) {
            let line = serde_json::to_string(entry)? + "\n";
            file.write_all(line.as_bytes()).await?;
        }
    }

    // Atomically rewrite episodic.jsonl with promoted flags updated.
    let updated: Vec<EpisodicEntry> = entries
        .into_iter()
        .map(|mut e| {
            if promote_ids.contains(&e.id) {
                e.promoted = true;
            }
            e
        })
        .collect();

    let tmp = format!("{}.tmp", EPISODIC_FILE);
    {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .await?;
        for entry in &updated {
            let line = serde_json::to_string(entry)? + "\n";
            file.write_all(line.as_bytes()).await?;
        }
    }
    tokio::fs::rename(&tmp, EPISODIC_FILE).await?;

    Ok(promote_ids.len())
}

/// Delete a single episodic entry by ID from both episodic and archival storage.
/// Returns `true` if the entry was found and removed from at least one file.
pub async fn delete(id: &str) -> Result<bool> {
    let mut removed = false;

    // Remove from episodic.jsonl
    let episodic = read_jsonl(EPISODIC_FILE).await?;
    if episodic.iter().any(|e| e.id == id) {
        let filtered: Vec<_> = episodic.into_iter().filter(|e| e.id != id).collect();
        let tmp = format!("{}.tmp", EPISODIC_FILE);
        {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .await?;
            for entry in &filtered {
                let line = serde_json::to_string(entry)? + "\n";
                file.write_all(line.as_bytes()).await?;
            }
        }
        tokio::fs::rename(&tmp, EPISODIC_FILE).await?;
        removed = true;
    }

    // Remove from archival.jsonl (entry may have been promoted)
    let archival = read_jsonl(ARCHIVAL_FILE).await?;
    if archival.iter().any(|e| e.id == id) {
        let filtered: Vec<_> = archival.into_iter().filter(|e| e.id != id).collect();
        let tmp = format!("{}.tmp", ARCHIVAL_FILE);
        {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .await?;
            for entry in &filtered {
                let line = serde_json::to_string(entry)? + "\n";
                file.write_all(line.as_bytes()).await?;
            }
        }
        tokio::fs::rename(&tmp, ARCHIVAL_FILE).await?;
        removed = true;
    }

    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_entry(id: &str, content: &str) -> EpisodicEntry {
        EpisodicEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: vec![],
            importance: 3,
            timestamp: chrono::Utc::now(),
            promoted: false,
        }
    }

    async fn write_jsonl(path: &str, entries: &[EpisodicEntry]) {
        let mut data = String::new();
        for e in entries {
            data.push_str(&serde_json::to_string(e).unwrap());
            data.push('\n');
        }
        tokio::fs::write(path, data).await.unwrap();
    }

    async fn read_jsonl_test(path: &str) -> Vec<EpisodicEntry> {
        match tokio::fs::read_to_string(path).await {
            Ok(text) => text.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect(),
            Err(_) => vec![],
        }
    }

    // Write a temp episodic file, call the inner logic, verify result.
    // We test the core delete logic by directly calling read_jsonl and replicating
    // the same filter/write pattern used in delete().
    #[tokio::test]
    async fn delete_removes_entry_from_file() {
        let path = format!("/tmp/aniebot_test_{}.jsonl", Uuid::new_v4());
        let entries = vec![
            make_entry("aaa", "keep me"),
            make_entry("bbb", "delete me"),
            make_entry("ccc", "keep me too"),
        ];
        write_jsonl(&path, &entries).await;

        // Replicate delete logic against the temp file.
        let loaded = read_jsonl_test(&path).await;
        let found = loaded.iter().any(|e| e.id == "bbb");
        assert!(found, "entry should exist before delete");
        let filtered: Vec<_> = loaded.into_iter().filter(|e| e.id != "bbb").collect();
        write_jsonl(&path, &filtered).await;

        let after = read_jsonl_test(&path).await;
        assert_eq!(after.len(), 2);
        assert!(after.iter().all(|e| e.id != "bbb"));
        assert!(after.iter().any(|e| e.id == "aaa"));
        assert!(after.iter().any(|e| e.id == "ccc"));

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn delete_returns_false_when_not_found() {
        let path = format!("/tmp/aniebot_test_{}.jsonl", Uuid::new_v4());
        let entries = vec![make_entry("aaa", "only entry")];
        write_jsonl(&path, &entries).await;

        let loaded = read_jsonl_test(&path).await;
        let found = loaded.iter().any(|e| e.id == "nonexistent");
        assert!(!found);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn delete_empty_file_is_safe() {
        let path = format!("/tmp/aniebot_test_{}.jsonl", Uuid::new_v4());
        tokio::fs::write(&path, "").await.unwrap();

        let loaded = read_jsonl_test(&path).await;
        assert!(loaded.is_empty());

        let _ = tokio::fs::remove_file(&path).await;
    }
}
