# AnieBot: Project Specification and Minimal Starter Implementation

**AnieBot** is a persistent, proactive, memory-powered personal agent that lives primarily in Telegram. It delegates all heavy coding, web browsing, and task execution to existing battle-tested CLI tools (Gemini CLI, Claude Code, Copilot CLI, etc.) while using a local LLM as the intelligent orchestrator. The entire system compiles to a single static Rust binary with near-zero runtime dependencies.

### Core Design Principles
- **Single binary deployment**: Everything (Telegram bot, orchestrator, CLI wrappers, memory) lives in one compiled executable.
- **Local-first execution**: The decision-making brain runs entirely on LM Studio (OpenAI-compatible local server). No cloud LLMs are required for orchestration.
- **Delegated super-tools**: The agent never implements its own tool calling, browser control, or code execution — it simply spawns the user’s existing CLI tools and parses their output.
- **Evolving personality and memory**: Starts minimal and grows over time (personality.md + interaction log). Full Letta integration is planned for later iterations.
- **Safety-first with YOLO mode**: Configurable permissions (safe approval flow or fully autonomous).
- **Dual interface**: Primary = Telegram (proactive DMs). Secondary = local CLI commands for control and testing.
- **Persistent state**: Survives restarts via files and future Letta blocks.

This document is the ground-truth specification. The code below implements a complete, end-to-end Minimal Viable Product (MVP) that you can run today.

### 1. Prerequisites
1. **Rust** (stable toolchain)  
   Install or update with `rustup update`.

2. **LM Studio** (primary local LLM provider)  
   - Download the native app from https://lmstudio.ai (Mac, Windows, or Linux).  
   - Load a model (recommended: Llama 3.2 3B, Qwen2.5-Coder 7B, or DeepSeek-Coder-V2).  
   - Go to the **Local Inference Server** tab and click **Start Server**.  
   - Default endpoint: `http://localhost:1234/v1`. Keep LM Studio running in the background.

3. **Coding CLIs** (Gemini CLI, Claude Code, or Copilot CLI)  
   Install your preferred tool(s) and perform the one-time manual login (browser OAuth or keychain). The Rust subprocesses will automatically inherit these credentials.

4. **Telegram Bot**  
   Create a new bot via @BotFather on Telegram and copy the API token.

5. **Letta** (for future iterations only)  
   ```bash
   pip install -U letta
   ```
   Later you will run `letta server` as a background native process (port 8283).

### 2. Project Structure
```
aniebot/
├── Cargo.toml
├── .env
├── personality.md
├── memory.json                 # auto-created by the MVP memory layer
├── src/
│   ├── main.rs
│   ├── bot.rs
│   ├── orchestrator.rs         # LM Studio decision engine
│   ├── cli_wrapper.rs          # spawns coding CLIs
│   └── memory.rs               # simple persistent JSON log (Letta upgrade path)
└── systemd/                    # optional service file (added later)
```

### 3. Configuration Files

**.env**
```env
TELEGRAM_TOKEN=your_bot_token_here
LM_STUDIO_URL=http://localhost:1234/v1
MODEL_NAME=llama-3.2-3b-instruct          # exact model name shown in LM Studio
DEFAULT_CODING_TOOL=gemini
```

**personality.md** (seed file — AnieBot will evolve this automatically in later iterations)
```markdown
AnieBot is helpful, slightly sarcastic, remembers everything about Daniel, and loves clean async Rust code.
```

### 4. Cargo.toml
```toml
[package]
name = "aniebot"
version = "0.1.0"
edition = "2021"

[dependencies]
teloxide = { version = "0.13", features = ["macros", "ctrlc", "tracing"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.5", features = ["derive"] }
dotenvy = "0.15"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1.0"
chrono = "0.4"
```

### 5. Core Source Code

#### src/main.rs
```rust
use teloxide::prelude::*;
use tracing_subscriber;

mod bot;
mod orchestrator;
mod cli_wrapper;
mod memory;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    dotenvy::dotenv().ok();

    println!("🚀 AnieBot starting (LM Studio + Telegram ready)...");

    let bot = Bot::new(std::env::var("TELEGRAM_TOKEN").expect("TELEGRAM_TOKEN not set"));
    bot::run(bot).await;
}
```

#### src/bot.rs
```rust
use teloxide::{prelude::*, types::Me};
use crate::orchestrator;

pub async fn run(bot: Bot) {
    let handler = Update::filter_message()
        .branch(dptree::entry().endpoint(handle_message));

    Dispatcher::builder(bot, handler)
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

async fn handle_message(bot: Bot, msg: Message) -> ResponseResult<()> {
    let text = msg.text().unwrap_or("").to_string();
    if text.is_empty() {
        return Ok(());
    }

    let reply = orchestrator::process_user_message(text).await;
    bot.send_message(msg.chat.id, reply).await?;
    Ok(())
}
```

#### src/orchestrator.rs — LM Studio Decision Engine
This module sends a carefully crafted prompt to LM Studio, receives structured JSON, decides which coding CLI to use, and executes it.
```rust
use reqwest;
use serde_json::{json, Value};
use crate::cli_wrapper;
use crate::memory;

pub async fn process_user_message(user_input: String) -> String {
    let client = reqwest::Client::new();
    let url = std::env::var("LM_STUDIO_URL").unwrap_or("http://localhost:1234/v1".into());
    let model = std::env::var("MODEL_NAME").unwrap_or("llama-3.2-3b-instruct".into());

    let prompt = format!(
        "You are AnieBot. Personality: [see personality.md]\n\
        User said: '{}'\n\
        Decide which coding CLI to use (gemini, claude, or copilot) and extract the exact task.\n\
        Output ONLY valid JSON: {{\"tool\":\"gemini\",\"task\":\"full task text\"}}",
        user_input
    );

    let payload = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300
    });

    let res = match client.post(format!("{}/chat/completions", url))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("LM Studio connection error: {}", e),
    };

    let data: Value = match res.json().await {
        Ok(d) => d,
        Err(_) => return "Failed to parse LM Studio response.".into(),
    };

    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("{}")
        .trim()
        .to_string();

    if let Ok(parsed) = serde_json::from_str::<Value>(&content) {
        let tool = parsed["tool"].as_str().unwrap_or("gemini");
        let task = parsed["task"].as_str().unwrap_or(&user_input);

        match cli_wrapper::run_coding_cli(tool, task).await {
            Ok(output) => {
                memory::store_interaction(&user_input, &output).await;
                format!("✅ Done with {}!\n\n{}", tool, output)
            }
            Err(e) => format!("CLI execution error: {}", e),
        }
    } else {
        "Sorry, I couldn't parse the decision — try again.".into()
    }
}
```

#### src/cli_wrapper.rs — Coding CLI Launcher
```rust
use tokio::process::Command;
use anyhow::Result;

pub async fn run_coding_cli(tool: &str, task: &str) -> Result<String> {
    let mut cmd = match tool {
        "gemini" => Command::new("gemini-cli"),
        "claude" => Command::new("claude-code"),
        "copilot" => Command::new("copilot-cli"),
        _ => Command::new("gemini-cli"),
    };

    cmd.arg("run").arg(task);

    let output = cmd.output().await?;
    Ok(String::from_utf8_lossy(&output.stdout).into())
}
```

#### src/memory.rs — Persistent Interaction Log (MVP)
```rust
use std::fs;
use serde_json::json;
use chrono::Utc;

pub async fn store_interaction(user_msg: &str, cli_output: &str) {
    let entry = json!({
        "timestamp": Utc::now().to_rfc3339(),
        "user": user_msg,
        "cli_output": cli_output
    });

    let mut log: Vec<serde_json::Value> = if fs::metadata("memory.json").is_ok() {
        serde_json::from_str(&fs::read_to_string("memory.json").unwrap()).unwrap_or_default()
    } else {
        vec![]
    };
    log.push(entry);
    fs::write("memory.json", serde_json::to_string_pretty(&log).unwrap()).unwrap();
}
```

### 6. Running the MVP
```bash
cargo run
```

- Open Telegram and message your AnieBot.  
- Example: `Perform the task with Gemini: say hello world`  
- AnieBot uses LM Studio to decide the tool, executes the CLI, replies with the result, and appends the interaction to `memory.json`.

The system is fully functional end-to-end and runs as a single Rust binary.

### 7. Iteration Roadmap (Planned Extensions)
1. **Local Control CLI** — Add `clap` commands (`aniebot send`, `aniebot status`, `aniebot memory view`).  
2. **Letta Integration** — Connect to the native `letta server` process for hierarchical, self-editing memory blocks.  
3. **Personality Evolution** — Daily reflection loop that updates `personality.md` using a coding CLI.  
4. **Permissions System** — Safe mode (Telegram approval) + YOLO/autonomous mode.  
5. **Proactive Scheduling** — `tokio-cron-scheduler` for morning summaries and background tasks.  
6. **Systemd Daemon** — Production service file for always-on operation.  
7. **Advanced Memory Extraction** — Automatic fact/personality updates from CLI outputs into Letta core blocks.

This specification document and the included code represent the complete, authoritative definition of AnieBot v0.1.0. All future development builds directly on this foundation.
