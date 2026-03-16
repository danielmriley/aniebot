use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use teloxide::{prelude::*, types::ChatId};
use tokio_cron_scheduler::{Job, JobScheduler};

use crate::config::Config;
use crate::orchestrator;

pub async fn start(bot: Bot, config: Arc<Config>) {
    let sched = match JobScheduler::new().await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to create scheduler: {}", e);
            return;
        }
    };

    add_morning_summary_job(&sched, bot.clone(), config.clone()).await;
    add_health_check_job(&sched, bot.clone(), config.clone()).await;

    if let Err(e) = sched.start().await {
        tracing::error!("Failed to start scheduler: {}", e);
    }

    // Keep this future alive so the spawned task doesn't drop the scheduler.
    std::future::pending::<()>().await;
}

async fn add_morning_summary_job(sched: &JobScheduler, bot: Bot, config: Arc<Config>) {
    let cron_expr = config.morning_summary_cron.clone();

    let job = Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
        let bot = bot.clone();
        let cfg = config.clone();
        Box::pin(async move {
            tracing::info!("Sending morning summary");
            let chat_id = cfg.allowed_user_id as i64;
            let reply = orchestrator::process_message(
                cfg.clone(),
                chat_id,
                "Give me a proactive morning summary.",
            )
            .await;
            if let Err(e) = bot.send_message(ChatId(chat_id), reply).await {
                tracing::error!("Failed to send morning summary: {}", e);
            }
        })
    });

    match job {
        Ok(j) => {
            if let Err(e) = sched.add(j).await {
                tracing::error!("Failed to add morning summary job: {}", e);
            }
        }
        Err(e) => tracing::error!("Failed to create morning summary job (bad cron?): {}", e),
    }
}

async fn add_health_check_job(sched: &JobScheduler, bot: Bot, config: Arc<Config>) {
    let interval = Duration::from_secs(config.health_check_interval_mins * 60);
    let lm_url = config.lm_studio_url.clone();
    let chat_id = config.allowed_user_id as i64;

    // true = currently healthy (optimistic initial state so we don't fire on startup)
    let is_healthy = Arc::new(AtomicBool::new(true));

    let job = Job::new_repeated_async(interval, move |_uuid, _lock| {
        let bot = bot.clone();
        let lm_url = lm_url.clone();
        let is_healthy = is_healthy.clone();
        Box::pin(async move {
            let reachable = ping_lm_studio(&lm_url).await;
            let was_healthy = is_healthy.swap(reachable, Ordering::Relaxed);

            match (was_healthy, reachable) {
                (true, false) => {
                    tracing::warn!("LM Studio is unreachable — notifying owner");
                    let _ = bot
                        .send_message(ChatId(chat_id), "⚠️ LM Studio is unreachable.")
                        .await;
                }
                (false, true) => {
                    tracing::info!("LM Studio has recovered — notifying owner");
                    let _ = bot
                        .send_message(ChatId(chat_id), "✅ LM Studio is back online.")
                        .await;
                }
                _ => {} // healthy→healthy or down→down: stay silent
            }
        })
    });

    match job {
        Ok(j) => {
            if let Err(e) = sched.add(j).await {
                tracing::error!("Failed to add health check job: {}", e);
            }
        }
        Err(e) => tracing::error!("Failed to create health check job: {}", e),
    }
}

/// Returns true if LM Studio responds with a 2xx status code.
async fn ping_lm_studio(lm_url: &str) -> bool {
    let url = format!("{}/v1/models", lm_url);
    match reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}
