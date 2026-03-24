# AnieBot Tool Reference

This document describes every tool available to the LLM. Tools are organized into functional groups. Each entry includes the tool name, which execution contexts it appears in, a description of what it does, and a parameter table.

**Contexts:** `U` = User conversation, `H` = Heartbeat (background), `C` = Consolidation (background), `I` = Interest evaluation

**Parallelism:** tools marked *parallel-safe* may be dispatched concurrently when an entire batch consists only of safe tools. All write tools dispatch sequentially.

---

## Session control

These tools terminate the agentic loop or send mid-loop messages to the user.

### `final_reply`
**Contexts:** U, H, C | **Parallel-safe:** no

Send a complete, polished reply to the user and end this session immediately. Use this when all work is done and results are ready to present.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | yes | The complete reply to send to the user. |

---

### `send_update`
**Contexts:** U, H | **Parallel-safe:** no

Send a progress message to the user now and keep working. Use this to keep the user informed during long tasks — for example, after completing one step and before starting the next. Must still call `final_reply` or `reflect(done=true)` when done.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | yes | The progress update to send. Be concise and informative. |

---

### `reflect`
**Contexts:** U, H, C, I | **Parallel-safe:** no

Record an intermediate observation about current progress. Call this after a series of tool calls to organize thinking. `done=false` continues the loop; `done=true` exits it and synthesizes a reply automatically. In background tasks (heartbeat, consolidation), `done=true` is the correct exit when there is nothing to report.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | string | yes | What has been done so far, and what was or wasn't found. |
| `done` | boolean | yes | Set `true` to exit the loop. A reply is synthesized from all work done. |
| `reply` | string | no | Optional draft reply. When `done=true` in a user conversation, passed to the synthesis step as a starting point. Ignored in background contexts. |

---

## File system

Tools for reading, writing, and navigating the file system. File paths are relative to the bot's working directory unless absolute.

### `read_file`
**Contexts:** U, H, C | **Parallel-safe:** yes

Read the contents of a file. Returns numbered lines. Use `offset` and `limit` to read large files in chunks.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative path to the file. |
| `offset` | integer | no | Line number to start reading from (1-based). |
| `limit` | integer | no | Maximum number of lines to return. |

---

### `write_file`
**Contexts:** U, H, C | **Parallel-safe:** no

Write content to a file, creating it or **overwriting it entirely**. Parent directories are created automatically. Use `edit_file` when you only need to change part of an existing file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative path to the file. |
| `content` | string | yes | Full content to write. |

---

### `edit_file`
**Contexts:** U, H, C | **Parallel-safe:** no

Make a targeted edit to a file by replacing an exact string. Read the file first to get the exact text to replace. Uses an atomic write (`.tmp` → rename) to avoid partial writes. Returns an error if `old_string` is not found; warns if multiple occurrences exist and `replace_all` is false.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative path to the file. |
| `old_string` | string | yes | Exact text to find and replace. Must appear verbatim in the file. |
| `new_string` | string | yes | Replacement text. |
| `replace_all` | boolean | no | If `true`, replace all occurrences instead of just the first (default `false`). |

---

### `list_dir`
**Contexts:** U, H | **Parallel-safe:** yes

List the contents of a directory. Subdirectory names are shown with a trailing `/`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative path to the directory. |

---

## Shell

### `shell_command`
**Contexts:** U, H | **Parallel-safe:** no

Run a shell command via `sh -c` and return combined stdout+stderr. Use for `git`, `grep`, `curl`, `python` scripts, builds, tests, etc. Output is capped at 16 KB. Long-running commands should use `timeout_secs` — default is the server-configured `SHELL_COMMAND_TIMEOUT_SECS` (default 30s).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cmd` | string | yes | Shell command to execute (passed to `sh -c`). |
| `cwd` | string | no | Working directory (defaults to the bot's working directory). |
| `timeout_secs` | integer | no | Seconds before the command is killed. Defaults to `SHELL_COMMAND_TIMEOUT_SECS`. Max 600. Use a higher value for builds or long-running scripts. |

---

## Web

### `fetch_page`
**Contexts:** U, H, C | **Parallel-safe:** yes

Fetch a web page. `mode='text'` returns a clean markdown rendering via Jina Reader — best for articles, documentation, or JS-rendered pages. `mode='raw'` returns the raw HTTP response body (up to 8 KB).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | yes | The URL to fetch (must be HTTP or HTTPS). |
| `mode` | enum: `raw`, `text` | no | `raw` = plain response body (8 KB cap); `text` = cleaned markdown via Jina Reader (32 KB cap). Defaults to `raw`. |

---

## Scheduling

Tools for creating, inspecting, updating, and deleting cron-driven and one-shot jobs. Scheduled jobs persist across restarts and are stored in `data/schedules.json`.

Cron expressions follow **6-field Quartz format** (`SEC MIN HRS DOM MON DOW`). 5-field standard cron is automatically promoted by prepending `"0 "`.

### `schedule_task`
**Contexts:** U | **Parallel-safe:** no

Create a recurring scheduled reminder or automated task.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cron` | string | yes | 5- or 6-field cron expression. Examples: `0 8 * * *` = daily at 08:00, `30 9 * * Mon-Fri` = weekdays at 09:30. |
| `label` | string | yes | Human-readable name for this job. |
| `task` | string | yes | Prompt that runs when the job fires. |

---

### `list_schedules`
**Contexts:** U | **Parallel-safe:** yes

List all currently scheduled jobs (ID, label, cron, task).

*No parameters.*

---

### `delete_schedule`
**Contexts:** U | **Parallel-safe:** no

Delete a scheduled job by its UUID. Removes it from the scheduler and from `data/schedules.json`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `schedule_id` | string | yes | UUID of the schedule to delete. |

---

### `update_schedule`
**Contexts:** U | **Parallel-safe:** no

Modify an existing scheduled job. Only fields provided are changed — omit any field to leave it as-is. The job ID is preserved; the scheduler is updated live.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `schedule_id` | string | yes | UUID of the schedule to update. |
| `label` | string | no | New human-readable name. |
| `cron` | string | no | New cron expression. Same format rules as `schedule_task`. |
| `task` | string | no | New prompt to run when the job fires. |

---

### `schedule_once`
**Contexts:** U | **Parallel-safe:** no

Run a task once at a future time, then auto-delete. Prefer `delay_seconds` for relative times ("in 5 minutes"); use `fire_at` only when the user specifies a specific clock time.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `delay_seconds` | integer | no* | Seconds from now until the job fires. Mutually exclusive with `fire_at`. |
| `fire_at` | string | no* | ISO 8601 UTC datetime. Only for specific clock times. Mutually exclusive with `delay_seconds`. |
| `label` | string | yes | Human-readable name. |
| `task` | string | yes | Prompt to execute when the job fires. |

*One of `delay_seconds` or `fire_at` is required.*

---

## Core memory

The bot's persistent identity and beliefs live in `data/core_memory.json`. These tools read and write it directly.

### `read_core_memory`
**Contexts:** U, H, C | **Parallel-safe:** yes

Read the current state of core memory (identity, beliefs, user profile, curiosity queue). Useful mid-session after updates to verify what was written.

*No parameters.*

---

### `update_core_memory`
**Contexts:** U, H, C | **Parallel-safe:** no

Update a section of core memory. Be conservative — only update when something meaningfully new has been established. In the consolidation context, the `identity` section is also exposed (use surgical edits only, not wholesale rewrites).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `section` | enum | yes | `beliefs` — evolving worldview; `user_profile` — facts about the user; `curiosity_queue` — topics to investigate; `identity` — self-description *(consolidation context only)*. |
| `content` | string | yes | New value. Plain text for `identity`/`user_profile`; JSON array string for `beliefs`/`curiosity_queue`. |

---

### `set_task`
**Contexts:** U, H, C | **Parallel-safe:** no

Record the current task or active project in core memory. Appears in every system prompt as a persistent reminder. Pass an empty string to clear once the task is complete.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | string | yes | Short description of the current task (1–2 sentences). Pass `""` to clear. |

---

## Episodic memory

A timestamped, searchable log of events. Stored in `data/episodic.jsonl`; entries with importance ≥ 4 are promoted to `data/archival.jsonl`.

### `remember`
**Contexts:** U, C | **Parallel-safe:** no

Log a notable moment, discovery, or event to episodic memory. Use during or after any exchange where something significant happened — a task completed, something surprising found, a meaningful decision made, a strong preference expressed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | yes | The observation or fact to remember. |
| `tags` | array of strings | yes | Topic tags for later retrieval, e.g. `["music", "preferences"]`. |
| `importance` | integer | yes | Importance 1–5 (5 = most important). Use 3 if unsure. |

---

### `recall`
**Contexts:** U, H, C | **Parallel-safe:** yes

Search episodic and archival memory by keyword or tags. Use before answering questions about past events or user preferences that may not be in the recent conversation. In heartbeat context, use this before sending a proactive message to check if the topic was covered recently.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Keyword or phrase to search for. |
| `tags` | array of strings | no | Optional tags to filter results. |

---

### `forget`
**Contexts:** U, H, C | **Parallel-safe:** no

Permanently delete an episodic memory entry by its UUID. Use when a stored note is incorrect, stale, or should no longer be recalled.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry_id` | string | yes | UUID of the episodic entry to delete, as returned by `recall`. |

---

### `list_episodic_recent`
**Contexts:** U, H, C | **Parallel-safe:** yes

Return the N most-recent episodic memory entries, ordered oldest-first. Useful for consolidation or reviewing recent activity.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `n` | integer | no | Number of recent entries to return (default 10, max 50). |

---

## Interests

Tracked topics the bot monitors proactively. Stored in `data/core_memory.json` under `active_interests`. Each interest has a health score (0–100) that decays during consolidation; interests that fall to 0 health are retired automatically.

### `add_interest`
**Contexts:** U, H, C, I | **Parallel-safe:** no

Register a topic the user explicitly asked to track. **Only call this when the user directly names a topic they want monitored** — never speculatively for related or inferred topics. The bot checks in automatically per cron schedule (or every heartbeat if no cron is given).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Short name for the interest, e.g. `"SpaceX launches"`. |
| `description` | string | yes | What specifically to look for or track. |
| `check_cron` | string | no | Optional 6-field Quartz cron for dedicated checks (e.g. `0 0 9 * * Mon` = Monday 09:00 UTC). Omit to rely on the hourly heartbeat. |

---

### `retire_interest`
**Contexts:** U, H, C, I | **Parallel-safe:** no

Remove a previously registered interest by its UUID. Use when the topic is no longer relevant.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `interest_id` | string | yes | UUID of the interest to retire, as shown in core memory. |

---

### `list_interests`
**Contexts:** U, I | **Parallel-safe:** yes

List all currently registered interests, including topic, description, health score, check schedule, and ID.

*No parameters.*

---

## Agenda

A persistent task list stored in `data/agenda.json`. Items appear in every system prompt until completed or cancelled. Distinct from `set_task` (which tracks a single active task) — the agenda is a queue of deferred work.

### `add_agenda_item`
**Contexts:** U, H, C | **Parallel-safe:** no

Queue a task to be done later. Use when there is work that cannot be finished now but needs to be remembered — it will appear in future system prompts until completed. Good for tasks deferred due to budget, scope, or ordering constraints.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | string | yes | What needs to be done. |
| `context` | string | no | Optional background helpful for completing this task later. |

---

### `list_agenda_items`
**Contexts:** U, H, C | **Parallel-safe:** yes

List all pending and in-progress agenda tasks.

*No parameters.*

---

### `update_agenda_item`
**Contexts:** U, H, C | **Parallel-safe:** no

Update the status or append a note to a pending agenda item. Use status `in_progress` when starting work on an item.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `item_id` | string | yes | UUID of the agenda item to update. |
| `status` | enum: `pending`, `in_progress` | no | New status. |
| `note` | string | no | Note to append to the item's context. |

---

### `cancel_agenda_item`
**Contexts:** U, H, C | **Parallel-safe:** no

Cancel a pending agenda task by its UUID when it is no longer relevant.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `item_id` | string | yes | UUID of the agenda item to cancel. |

---

### `complete_agenda_item`
**Contexts:** H, C | **Parallel-safe:** no

Mark an agenda task as completed and record what was accomplished.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `item_id` | string | yes | UUID of the agenda item to complete. |
| `result` | string | yes | Summary of what was accomplished. |

---

## Utility

### `get_current_time`
**Contexts:** U, H, C | **Parallel-safe:** yes

Get the current date and time in both local time and UTC. Use when the current timestamp is needed for scheduling calculations or time-sensitive decisions.

*No parameters.*

---

## Context summary table

| Tool | U | H | C | I |
|------|---|---|---|---|
| `final_reply` | ✓ | ✓ | ✓ | |
| `send_update` | ✓ | ✓ | | |
| `reflect` | ✓ | ✓ | ✓ | ✓ |
| `read_file` | ✓ | ✓ | ✓ | |
| `write_file` | ✓ | ✓ | ✓ | |
| `edit_file` | ✓ | ✓ | ✓ | |
| `list_dir` | ✓ | ✓ | | |
| `shell_command` | ✓ | ✓ | | |
| `fetch_page` | ✓ | ✓ | ✓ | |
| `schedule_task` | ✓ | | | |
| `list_schedules` | ✓ | | | |
| `delete_schedule` | ✓ | | | |
| `update_schedule` | ✓ | | | |
| `schedule_once` | ✓ | | | |
| `read_core_memory` | ✓ | ✓ | ✓ | |
| `update_core_memory` | ✓ | ✓ | ✓ | |
| `set_task` | ✓ | ✓ | ✓ | |
| `remember` | ✓ | | ✓ | |
| `recall` | ✓ | ✓ | ✓ | |
| `forget` | ✓ | ✓ | ✓ | |
| `list_episodic_recent` | ✓ | ✓ | ✓ | |
| `add_interest` | ✓ | ✓ | ✓ | ✓ |
| `retire_interest` | ✓ | ✓ | ✓ | ✓ |
| `list_interests` | ✓ | | | ✓ |
| `add_agenda_item` | ✓ | ✓ | ✓ | |
| `list_agenda_items` | ✓ | ✓ | ✓ | |
| `update_agenda_item` | ✓ | ✓ | ✓ | |
| `cancel_agenda_item` | ✓ | ✓ | ✓ | |
| `complete_agenda_item` | | ✓ | ✓ | |
| `get_current_time` | ✓ | ✓ | ✓ | |
