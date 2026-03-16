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
        })
    }
}
