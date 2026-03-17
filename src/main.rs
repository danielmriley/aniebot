use std::sync::Arc;

use anyhow::Result;
use teloxide::Bot;
use tracing_subscriber::EnvFilter;

mod bot;
mod cli_wrapper;
mod config;
mod memory;
mod orchestrator;
mod schedule_store;
mod scheduler;

use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = Arc::new(Config::from_env()?);

    tokio::fs::create_dir_all("data/conversations").await?;
    tokio::fs::create_dir_all("data").await?;
    tokio::fs::create_dir_all(&config.workspace_dir).await?;

    tracing::info!("AnieBot starting...");

    let bot = Bot::new(config.telegram_token.clone());
    let scheduler = scheduler::start(bot.clone(), config.clone()).await;
    bot::run(bot, config, scheduler).await;

    Ok(())
}
