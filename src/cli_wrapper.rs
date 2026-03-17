use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::config::Config;

pub async fn run(config: &Config, task: &str) -> Result<String> {
    let mut cmd = Command::new("copilot");
    cmd.arg("-p")
        .arg(task)
        .arg("--allow-all")
        .arg("--silent")
        .arg("--no-ask-user");
    if let Some(model) = &config.copilot_model {
        cmd.arg("--model").arg(model);
    }
    cmd.current_dir(&config.workspace_dir);

    let timeout = Duration::from_secs(config.cli_timeout_secs);

    let output = tokio::time::timeout(timeout, cmd.output())
        .await
        .with_context(|| {
            format!(
                "copilot timed out after {} seconds",
                config.cli_timeout_secs
            )
        })?
        .context("failed to spawn copilot process")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        anyhow::bail!("copilot exited with an error:\n{}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
