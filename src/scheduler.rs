use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use teloxide::{prelude::*, types::ChatId};
use tokio::sync::{mpsc, oneshot};
use tokio_cron_scheduler::{Job, JobScheduler};
use uuid::Uuid;

use chrono::Utc;

use crate::config::Config;
use crate::core_memory;
use crate::orchestrator;
use crate::schedule_store::{self, ScheduleEntry};

// ---------------------------------------------------------------------------
// Public handle — only a channel sender so it is Send + Sync
// ---------------------------------------------------------------------------

/// Lightweight, cloneable handle to the live scheduler.
///
/// All mutable scheduler state lives inside the private `scheduler_worker`
/// task. Callers interact exclusively via channel messages, which are always
/// `Send` regardless of the internal scheduler implementation.
pub struct SchedulerHandle {
    command_tx: mpsc::Sender<ScheduleCommand>,
}

// Commands the background worker understands.
enum ScheduleCommand {
    Add {
        entry: ScheduleEntry,
        bot: Bot,
        config: Arc<Config>,
    },
    Remove {
        entry_id: String,
        reply_tx: oneshot::Sender<bool>,
    },
    AddInterestJob {
        interest: core_memory::Interest,
        bot: Bot,
        config: Arc<Config>,
    },
}

// ---------------------------------------------------------------------------
// Public entry-point
// ---------------------------------------------------------------------------

/// Spawn the scheduler background worker, register all built-in and persisted
/// jobs, and return a shared `Arc<SchedulerHandle>` that can be cloned freely.
pub async fn start(bot: Bot, config: Arc<Config>) -> Arc<SchedulerHandle> {
    let (tx, rx) = mpsc::channel(64);
    let handle = Arc::new(SchedulerHandle { command_tx: tx });
    tokio::spawn(scheduler_worker(rx, handle.clone(), bot, config));
    handle
}

// ---------------------------------------------------------------------------
// Public dynamic-job API
// ---------------------------------------------------------------------------

/// Send an "add job" command to the scheduler worker. The job will fire the
/// stored `entry.task` prompt through the orchestrator and reply to the owner.
pub async fn add_dynamic_job(
    handle: &SchedulerHandle,
    bot: Bot,
    config: Arc<Config>,
    entry: ScheduleEntry,
) -> anyhow::Result<()> {
    handle
        .command_tx
        .send(ScheduleCommand::Add { entry, bot, config })
        .await
        .map_err(|_| anyhow::anyhow!("Scheduler worker is not running"))?;
    Ok(())
}

/// Register a per-interest cron job. No-op if the interest has no check_cron.
pub async fn add_interest_job(
    handle: &SchedulerHandle,
    bot: Bot,
    config: Arc<Config>,
    interest: core_memory::Interest,
) -> anyhow::Result<()> {
    handle
        .command_tx
        .send(ScheduleCommand::AddInterestJob { interest, bot, config })
        .await
        .map_err(|_| anyhow::anyhow!("Scheduler worker is not running"))?;
    Ok(())
}

/// Send a "remove job" command and wait for a boolean confirmation.
/// Returns `true` if a job with the given stable entry ID was found and removed.
pub async fn remove_dynamic_job(
    handle: &SchedulerHandle,
    entry_id: &str,
) -> anyhow::Result<bool> {
    let (reply_tx, reply_rx) = oneshot::channel();
    handle
        .command_tx
        .send(ScheduleCommand::Remove {
            entry_id: entry_id.to_string(),
            reply_tx,
        })
        .await
        .map_err(|_| anyhow::anyhow!("Scheduler worker is not running"))?;
    Ok(reply_rx.await.unwrap_or(false))
}

// ---------------------------------------------------------------------------
// Background worker — owns the JobScheduler and id_map
// ---------------------------------------------------------------------------

async fn scheduler_worker(
    mut rx: mpsc::Receiver<ScheduleCommand>,
    handle: Arc<SchedulerHandle>,
    bot: Bot,
    config: Arc<Config>,
) {
    let sched = match JobScheduler::new().await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to create scheduler: {}", e);
            return;
        }
    };

    // Maps stable entry ID → scheduler-internal UUID (rebuilt on every start).
    let mut id_map: HashMap<String, Uuid> = HashMap::new();

    add_morning_summary_job(&sched, bot.clone(), config.clone()).await;
    add_health_check_job(&sched, bot.clone(), config.clone()).await;
    add_heartbeat_job(&sched, bot.clone(), config.clone(), handle.clone()).await;
    load_and_add_interest_jobs(&sched, &mut id_map, bot.clone(), config.clone(), handle.clone()).await;
    load_and_add_persisted_jobs(&sched, &mut id_map, bot.clone(), config.clone())
        .await;

    if let Err(e) = sched.start().await {
        tracing::error!("Failed to start scheduler: {}", e);
        return;
    }

    tracing::info!("Scheduler worker running");

    // Process commands until the sender side is dropped.
    while let Some(cmd) = rx.recv().await {
        match cmd {
            ScheduleCommand::Add { entry, bot, config } => {
                match create_and_register_job(&sched, bot, config, &entry).await {
                    Ok(uuid) => {
                        tracing::info!("Dynamic job '{}' registered ({})", entry.label, uuid);
                        id_map.insert(entry.id, uuid);
                    }
                    Err(e) => tracing::warn!("Failed to register dynamic job '{}': {}", entry.label, e),
                }
            }
            ScheduleCommand::Remove { entry_id, reply_tx } => {
                let found = if let Some(uuid) = id_map.remove(&entry_id) {
                    match sched.remove(&uuid).await {
                        Ok(_) => true,
                        Err(e) => {
                            tracing::warn!("Scheduler remove error for {}: {}", entry_id, e);
                            false
                        }
                    }
                } else {
                    false
                };
                let _ = reply_tx.send(found);
            }
            ScheduleCommand::AddInterestJob { interest, bot, config } => {
                create_and_register_interest_job(&sched, &mut id_map, bot, config, handle.clone(), interest).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in jobs
// ---------------------------------------------------------------------------

async fn add_morning_summary_job(
    sched: &JobScheduler,
    bot: Bot,
    config: Arc<Config>,
) {
    let cron_expr = config.morning_summary_cron.clone();

    let job = Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
        let bot = bot.clone();
        let cfg = config.clone();
        Box::pin(async move {
            tracing::info!("Sending morning summary");
            let chat_id = cfg.allowed_user_id as i64;
            let reply = orchestrator::execute_scheduled_task(
                cfg.clone(),
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

async fn add_heartbeat_job(
    sched: &JobScheduler,
    bot: Bot,
    config: Arc<Config>,
    handle: Arc<SchedulerHandle>,
) {
    let cron_expr = config.heartbeat_cron.clone();

    let job = Job::new_async(cron_expr.as_str(), move |_uuid, _lock| {
        let bot = bot.clone();
        let cfg = config.clone();
        let handle = handle.clone();
        Box::pin(async move {
            tracing::info!("Heartbeat firing");
            orchestrator::run_heartbeat(cfg, bot, handle).await;
        })
    });

    match job {
        Ok(j) => {
            if let Err(e) = sched.add(j).await {
                tracing::error!("Failed to add heartbeat job: {}", e);
            } else {
                tracing::info!("Heartbeat job registered (cron: {})", cron_expr);
            }
        }
        Err(e) => tracing::error!("Failed to create heartbeat job (bad cron '{}'?): {}", cron_expr, e),
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

// ---------------------------------------------------------------------------
// Dynamic-job helpers (used by the worker only)
// ---------------------------------------------------------------------------

/// Create and register a single dynamic job in the scheduler.
/// Returns the scheduler-internal UUID for storage in `id_map`.
async fn create_and_register_job(
    sched: &JobScheduler,
    bot: Bot,
    config: Arc<Config>,
    entry: &ScheduleEntry,
) -> anyhow::Result<Uuid> {
    let task = entry.task.clone();
    let chat_id = config.allowed_user_id as i64;

    let job = if let Some(fire_at) = entry.fire_once_at {
        let remaining = (fire_at - Utc::now())
            .to_std()
            .map_err(|_| anyhow::anyhow!("schedule_once target time is in the past"))?;
        let cleanup_id = entry.id.clone();
        Job::new_one_shot_async(remaining, move |_uuid, _lock| {
            let bot = bot.clone();
            let cfg = config.clone();
            let task = task.clone();
            let cleanup_id = cleanup_id.clone();
            Box::pin(async move {
                tracing::info!("One-shot job firing — task: {:?}", task);
                let reply = orchestrator::execute_scheduled_task(cfg.clone(), &task).await;
                if let Err(e) = bot.send_message(ChatId(chat_id), reply).await {
                    tracing::error!("Failed to send one-shot job reply: {}", e);
                }
                let _ = schedule_store::remove(&cleanup_id).await;
            })
        })?
    } else {
        Job::new_async(entry.cron.as_str(), move |_uuid, _lock| {
            let bot = bot.clone();
            let cfg = config.clone();
            let task = task.clone();
            Box::pin(async move {
                tracing::info!("Dynamic job firing — task: {:?}", task);
                let reply = orchestrator::execute_scheduled_task(cfg.clone(), &task).await;
                if let Err(e) = bot.send_message(ChatId(chat_id), reply).await {
                    tracing::error!("Failed to send dynamic job reply: {}", e);
                }
            })
        })?
    };

    let uuid = job.guid();
    sched.add(job).await?;
    Ok(uuid)
}

/// Load all persisted jobs from `data/schedules.json` at startup.
async fn load_and_add_persisted_jobs(
    sched: &JobScheduler,
    id_map: &mut HashMap<String, Uuid>,
    bot: Bot,
    config: Arc<Config>,
) {
    let entries = match schedule_store::load().await {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!("Could not load persisted schedules: {}", err);
            return;
        }
    };

    if entries.is_empty() {
        return;
    }

    tracing::info!("Loading {} persisted dynamic job(s)", entries.len());
    for entry in entries {
        if entry.fire_once_at.map_or(false, |t| t <= Utc::now()) {
            tracing::info!("One-shot job '{}' expired while offline — removing", entry.label);
            let _ = schedule_store::remove(&entry.id).await;
            continue;
        }
        let label = entry.label.clone();
        let id = entry.id.clone();
        match create_and_register_job(sched, bot.clone(), config.clone(), &entry)
            .await
        {
            Ok(uuid) => {
                id_map.insert(id, uuid);
            }
            Err(e) => tracing::warn!("Skipping persisted job '{}': {}", label, e),
        }
    }
}

/// At startup, register a dedicated cron job for every saved interest that has
/// a `check_cron` value. Interests without a cron are covered by the global heartbeat.
async fn load_and_add_interest_jobs(
    sched: &JobScheduler,
    id_map: &mut HashMap<String, Uuid>,
    bot: Bot,
    config: Arc<Config>,
    handle: Arc<SchedulerHandle>,
) {
    let core = match core_memory::load().await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Could not load core memory for interest jobs: {}", e);
            return;
        }
    };
    let interests_with_cron: Vec<_> = core.interests.into_iter().filter(|i| i.check_cron.is_some()).collect();
    if interests_with_cron.is_empty() {
        return;
    }
    tracing::info!("Registering {} interest job(s) from core memory", interests_with_cron.len());
    for interest in interests_with_cron {
        create_and_register_interest_job(sched, id_map, bot.clone(), config.clone(), handle.clone(), interest).await;
    }
}

/// Create and register a cron job for a single interest. No-op if `check_cron` is None.
async fn create_and_register_interest_job(
    sched: &JobScheduler,
    id_map: &mut HashMap<String, Uuid>,
    bot: Bot,
    config: Arc<Config>,
    handle: Arc<SchedulerHandle>,
    interest: core_memory::Interest,
) {
    let Some(ref raw_cron) = interest.check_cron else {
        return; // global heartbeat covers this interest
    };
    if raw_cron.trim().is_empty() {
        return; // stored as "" — treat as no dedicated cron
    }
    // Normalize: promote 5-field → 6-field Quartz (tokio-cron-scheduler requires 6 fields).
    let cron_owned;
    let cron = match raw_cron.split_whitespace().count() {
        5 => {
            cron_owned = format!("0 {}", raw_cron);
            tracing::info!("Interest '{}': auto-promoted cron to 6-field: {}", interest.topic, cron_owned);
            cron_owned.as_str()
        }
        _ => raw_cron.as_str(),
    };
    let interest_id = interest.id.clone();
    let topic = interest.topic.clone();
    let stable_id = interest.id.clone();

    let job = Job::new_async(cron, move |_uuid, _lock| {
        let bot = bot.clone();
        let cfg = config.clone();
        let handle = handle.clone();
        let id = interest_id.clone();
        Box::pin(async move {
            tracing::info!("Interest check firing for: {}", id);
            orchestrator::run_interest_check(cfg, bot, handle, id).await;
        })
    });

    match job {
        Ok(j) => {
            let uuid = j.guid();
            if let Err(e) = sched.add(j).await {
                tracing::error!("Failed to add interest job for '{}': {}", topic, e);
            } else {
                tracing::info!("Interest job registered for '{}' (cron: {})", topic, cron);
                id_map.insert(stable_id, uuid);
            }
        }
        Err(e) => tracing::error!("Failed to create interest job for '{}' (bad cron?): {}", topic, e),
    }
}

