use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::Config;
use crate::tools::LmToolCall;

// ---------------------------------------------------------------------------
// LM Studio API types
// ---------------------------------------------------------------------------

/// Outgoing message in the messages array.
#[derive(Serialize, Deserialize)]
pub struct LmMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Forwarded from history for assistant tool-call turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Forwarded from history for tool-result turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Full chat completion response from LM Studio.
#[derive(Deserialize)]
pub struct LmCompletionResponse {
    pub choices: Vec<LmChoice>,
}

#[derive(Deserialize)]
pub struct LmChoice {
    pub finish_reason: String,
    pub message: LmAssistantMessage,
}

#[derive(Deserialize)]
pub struct LmAssistantMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<LmToolCall>,
}

// ---------------------------------------------------------------------------
// LM Studio HTTP client
// ---------------------------------------------------------------------------

pub async fn post_to_lm_studio(
    config: &Config,
    messages: &[LmMessage],
    tools: Option<&serde_json::Value>,
    max_tokens: u32,
) -> anyhow::Result<LmCompletionResponse> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.lm_studio_url);

    let mut payload = json!({
        "model": config.model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    });

    if let Some(t) = tools {
        payload["tools"] = t.clone();
        payload["tool_choice"] = json!("required");
    }

    let response = client
        .post(&url)
        .json(&payload)
        .timeout(Duration::from_secs(900))
        .send()
        .await
        .context("Failed to reach LM Studio — is it running?")?;

    if !response.status().is_success() {
        anyhow::bail!("LM Studio returned HTTP {}", response.status());
    }

    response
        .json::<LmCompletionResponse>()
        .await
        .context("Failed to deserialize LM Studio response")
}

// ---------------------------------------------------------------------------
// Response utilities
// ---------------------------------------------------------------------------

/// Strip `<think>...</think>` reasoning blocks emitted by reasoning models.
pub(crate) fn strip_think_blocks(s: &str) -> &str {
    if let Some(end) = s.rfind("</think>") {
        &s[end + "</think>".len()..]
    } else {
        s
    }
}
