use std::sync::Arc;

use anyhow::Result;
use teloxide::Bot;
use tracing_subscriber::EnvFilter;

struct LocalTimer;

impl tracing_subscriber::fmt::time::FormatTime for LocalTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))
    }
}

mod agent_session;
mod agenda;
mod bot;
mod cli_wrapper;
mod config;
mod core_memory;
mod episodic;
mod lm;
mod memory;
mod orchestrator;
mod schedule_store;
mod scheduler;
mod tools;

use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_timer(LocalTimer)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Arc::new(Config::from_env()?);

    tokio::fs::create_dir_all("data/conversations").await?;
    tokio::fs::create_dir_all("data").await?;
    tokio::fs::create_dir_all(&config.workspace_dir).await?;

    // Seed core memory from personality.md on first run.
    if !tokio::fs::try_exists("data/core_memory.json").await.unwrap_or(false) {
        let identity = tokio::fs::read_to_string("personality.md")
            .await
            .unwrap_or_else(|_| "I am AnieBot, a helpful and opinionated AI assistant.".to_string());
        let cm = core_memory::CoreMemory { identity, ..Default::default() };
        core_memory::save(&cm).await?;
        tracing::info!("Core memory seeded from personality.md");
    }

    tracing::info!("AnieBot starting...");

    let bot = Bot::new(config.telegram_token.clone());
    let scheduler = scheduler::start(bot.clone(), config.clone()).await;
    bot::run(bot, config, scheduler).await;

    Ok(())
}
