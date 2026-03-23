use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(Clone)]
pub struct Config {
    pub telegram_token: String,
    pub lm_studio_url: String,
    pub model_name: String,
    pub allowed_user_id: u64,
    pub workspace_dir: PathBuf,
    pub cli_timeout_secs: u64,
    pub morning_summary_cron: String,
    pub health_check_interval_mins: u64,
    pub copilot_model: Option<String>,
    pub heartbeat_cron: String,
    pub background_copilot_model: Option<String>,
    pub consolidation_threshold: usize,
    pub max_iters_conversation: usize,
    pub max_iters_heartbeat: usize,
    pub max_iters_consolidation: usize,
    pub max_iters_agenda: usize,
    pub context_compress_threshold: usize,
    pub background_lm_model: Option<String>,
    pub session_gap_hours: u64,
    pub session_summary_min_messages: usize,
    pub history_max_stored: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let telegram_token = std::env::var("TELEGRAM_TOKEN")
            .context("TELEGRAM_TOKEN must be set")?;

        let lm_studio_url = std::env::var("LM_STUDIO_URL")
            .context("LM_STUDIO_URL must be set")?;

        let model_name = std::env::var("MODEL_NAME")
            .context("MODEL_NAME must be set")?;

        let allowed_user_id = std::env::var("ALLOWED_USER_ID")
            .context("ALLOWED_USER_ID must be set")?
            .parse::<u64>()
            .context("ALLOWED_USER_ID must be a valid integer (your Telegram user ID)")?;

        let workspace_dir = PathBuf::from(
            std::env::var("WORKSPACE_DIR").unwrap_or_else(|_| "./workspace".into()),
        );

        let cli_timeout_secs = std::env::var("CLI_TIMEOUT_SECS")
            .unwrap_or_else(|_| "120".into())
            .parse::<u64>()
            .context("CLI_TIMEOUT_SECS must be a valid integer")?;

        let morning_summary_cron = std::env::var("MORNING_SUMMARY_CRON")
            .unwrap_or_else(|_| "0 0 8 * * *".into());

        let copilot_model = std::env::var("COPILOT_MODEL").ok();
        let heartbeat_cron = std::env::var("HEARTBEAT_CRON")
            .unwrap_or_else(|_| "0 0 * * * *".into());
        let background_copilot_model = std::env::var("BACKGROUND_COPILOT_MODEL").ok();
        let consolidation_threshold = std::env::var("CONSOLIDATION_THRESHOLD")
            .unwrap_or_else(|_| "15".into())
            .parse::<usize>()
            .context("CONSOLIDATION_THRESHOLD must be a valid integer")?;

        let max_iters_conversation = std::env::var("LOOP_ITERS_CONVERSATION")
            .unwrap_or_else(|_| "20".into())
            .parse::<usize>()
            .context("LOOP_ITERS_CONVERSATION must be a valid integer")?;

        let max_iters_heartbeat = std::env::var("LOOP_ITERS_HEARTBEAT")
            .unwrap_or_else(|_| "12".into())
            .parse::<usize>()
            .context("LOOP_ITERS_HEARTBEAT must be a valid integer")?;

        let max_iters_consolidation = std::env::var("LOOP_ITERS_CONSOLIDATION")
            .unwrap_or_else(|_| "10".into())
            .parse::<usize>()
            .context("LOOP_ITERS_CONSOLIDATION must be a valid integer")?;

        let max_iters_agenda = std::env::var("LOOP_ITERS_AGENDA")
            .unwrap_or_else(|_| "20".into())
            .parse::<usize>()
            .context("LOOP_ITERS_AGENDA must be a valid integer")?;

        let context_compress_threshold = std::env::var("LOOP_CONTEXT_THRESHOLD")
            .unwrap_or_else(|_| "30".into())
            .parse::<usize>()
            .context("LOOP_CONTEXT_THRESHOLD must be a valid integer")?;

        let background_lm_model = std::env::var("BACKGROUND_LM_MODEL").ok();

        let session_gap_hours = std::env::var("SESSION_GAP_HOURS")
            .unwrap_or_else(|_| "2".into())
            .parse::<u64>()
            .context("SESSION_GAP_HOURS must be a valid integer")?;

        let session_summary_min_messages = std::env::var("SESSION_SUMMARY_MIN_MESSAGES")
            .unwrap_or_else(|_| "6".into())
            .parse::<usize>()
            .context("SESSION_SUMMARY_MIN_MESSAGES must be a valid integer")?;

        let history_max_stored = std::env::var("HISTORY_MAX_STORED")
            .unwrap_or_else(|_| "500".into())
            .parse::<usize>()
            .context("HISTORY_MAX_STORED must be a valid integer")?;

        let health_check_interval_mins = std::env::var("HEALTH_CHECK_INTERVAL_MINS")
            .unwrap_or_else(|_| "30".into())
            .parse::<u64>()
            .context("HEALTH_CHECK_INTERVAL_MINS must be a valid integer")?;

        Ok(Self {
            telegram_token,
            lm_studio_url,
            model_name,
            allowed_user_id,
            workspace_dir,
            cli_timeout_secs,
            morning_summary_cron,
            health_check_interval_mins,
            copilot_model,
            heartbeat_cron,
            background_copilot_model,
            consolidation_threshold,
            max_iters_conversation,
            max_iters_heartbeat,
            max_iters_consolidation,
            max_iters_agenda,
            context_compress_threshold,
            background_lm_model,
            session_gap_hours,
            session_summary_min_messages,
            history_max_stored,
        })
    }
}
