use std::sync::Arc;
use std::time::Duration;

use teloxide::{dptree, prelude::*, types::ChatAction};

use crate::config::Config;
use crate::orchestrator;
use crate::scheduler::SchedulerHandle;

pub async fn run(bot: Bot, config: Arc<Config>, scheduler: Arc<SchedulerHandle>) {
    let handler = Update::filter_message().endpoint(handle_message);

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![config, scheduler])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

async fn handle_message(
    bot: Bot,
    msg: Message,
    config: Arc<Config>,
    scheduler: Arc<SchedulerHandle>,
) -> ResponseResult<()> {
    // Auth gate: silently drop messages from anyone other than the owner.
    if msg.from.as_ref().map(|u| u.id.0) != Some(config.allowed_user_id) {
        return Ok(());
    }

    let text = msg.text().unwrap_or("").to_string();
    if text.is_empty() {
        return Ok(());
    }

    // Show typing indicator immediately, then refresh every 4s for the duration.
    // Telegram drops it automatically after ~5s without a refresh.
    bot.send_chat_action(msg.chat.id, ChatAction::Typing).await?;
    let bot_typing = bot.clone();
    let typing_chat_id = msg.chat.id;
    let typing_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(4)).await;
            let _ = bot_typing.send_chat_action(typing_chat_id, ChatAction::Typing).await;
        }
    });

    let chat_id = msg.chat.id.0;
    let reply = orchestrator::process_message(config, bot.clone(), scheduler, chat_id, &text).await;

    typing_task.abort();
    bot.send_message(msg.chat.id, reply).await?;

    Ok(())
}
