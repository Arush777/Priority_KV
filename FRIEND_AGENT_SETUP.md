# FRIEND_AGENT_SETUP — Priority_KV

**Repo:** https://github.com/Arush777/Priority_KV  
**Your agent id:** `friend`  
**Peer agent id:** `arush`  
**Telegram bot:** `@arush_ir_collab_bot`  
**Telegram group:** `Information_Retrieval agents`  
**Telegram chat id:** `-5470510083`  
**Canon plan:** `docs/PRIORITYKV_IMPLEMENTATION_PLAN.md`  
**Do not implement large features during setup** — bridge first, then CLAIM work.

Arush must send you the **bot token privately** (not via GitHub).

---

## Prompt for friend's Cursor agent (paste as-is)

```text
You are agent `friend` on https://github.com/Arush777/Priority_KV (PriorityKV-Agent).
Your peer is agent `arush`.

Read and follow:
- COLLAB.md
- scopes/PROJECT_SCOPE.md
- FRIEND_AGENT_SETUP.md
- AGENTS.md
- docs/PRIORITYKV_IMPLEMENTATION_PLAN.md  (canon research plan)

Rules for now:
1. Do NOT implement research/systems code until the human says setup is done.
2. Your role after setup: lead Workstream A / PriorityBench / eval angle from the plan.
3. Collaborate via Telegram collab_bridge (ticks/daemon).
4. Branches only under `agent/friend/...`. Never force-push main. Never commit secrets.
5. Commits use THIS machine's git user.name / user.email (the human owner).
6. Message grammar: CLAIM / DONE / BLOCKED / @agent:arush / PROPOSE_SCOPE.

Setup with the human (do this first):
- git clone git@github.com:Arush777/Priority_KV.git && cd Priority_KV
- cp .env.example .env
- AGENT_ID=friend
- SAME TELEGRAM_BOT_TOKEN as Arush (private share)
- TELEGRAM_CHAT_ID=-5470510083
- CURSOR_API_KEY for THIS Cursor account
- CURSOR_MODEL=auto
- REPO_ROOT=/absolute/path/to/Priority_KV
- GITHUB_REPO=Arush777/Priority_KV
- DRY_RUN=0
- python3 -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- pip install cursor-sdk
- If pip fails with "versions: none", use: python -m pip install cursor-sdk
  and verify `which python` / `which pip` point inside .venv
- python -m collab_bridge check
- python -m collab_bridge ping
- python -m collab_bridge tick   # one ack only
- start: python -m collab_bridge daemon   (or hourly cron)

When bridge ping appears in Telegram, post:
[agent:friend] online — CLAIM S1 after humans confirm setup complete.
Do not start coding the plan until humans say so.
```

---

## Friend human setup (step by step)

### 0. Prerequisites from Arush
- GitHub write invite to `Arush777/Priority_KV` accepted
- Added to Telegram group `Information_Retrieval agents`
- Received `TELEGRAM_BOT_TOKEN` in a **private** message

### 1. Git identity (your name on commits)

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

### 2. Clone

```bash
git clone git@github.com:Arush777/Priority_KV.git
cd Priority_KV
```

### 3. `.env`

```bash
cp .env.example .env
```

Set at least:

```bash
AGENT_ID=friend
TELEGRAM_BOT_TOKEN=<from Arush privately>
TELEGRAM_CHAT_ID=-5470510083
CURSOR_API_KEY=<your Cursor API key>
CURSOR_MODEL=auto
REPO_ROOT=/absolute/path/to/Priority_KV
GITHUB_REPO=Arush777/Priority_KV
DRY_RUN=0
```

Cursor API key: https://cursor.com/dashboard/integrations

### 4. Install + smoke

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install cursor-sdk
python -m collab_bridge check
python -m collab_bridge ping
```

You should see `[agent:friend] bridge online` in Telegram.

### 5. Keep the bridge alive

```bash
# recommended while testing
tmux new -s pk-bridge
source .venv/bin/activate
python -m collab_bridge daemon
# Ctrl-b then d to detach
```

Or hourly cron via `./scripts/install_cron.sh`.

### 6. Paste the Cursor agent prompt

Open Cursor on the `Priority_KV` checkout and paste the prompt block above.

### 7. Announce in Telegram

```text
[human] friend joined. agent friend setup starting.
```

After ping succeeds:

```text
[agent:friend] online — waiting for humans before CLAIM S1
```

---

## Emergency controls (either human)

| Message | Effect |
|---------|--------|
| `STOP_BRIDGE` / `HALT_AGENTS` | Pause |
| `RESUME_BRIDGE` | Resume |
| `ACK_SCOPE` | Accept scope proposal |
