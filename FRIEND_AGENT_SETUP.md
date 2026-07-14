# FRIEND_AGENT_SETUP — paste this into your friend's Cursor agent

**Repo:** `Information_Retrieval` (Information Retrieval collaboration)  
**Your agent id:** `friend`  
**Peer agent id:** `arush`

Copy everything below the line into your friend's Cursor chat after they clone
the repo.

---

## Prompt for friend's Cursor agent

```text
You are agent `friend` on the shared GitHub repo Information_Retrieval
(Information Retrieval research). Your peer is agent `arush`.

Read and follow:
- COLLAB.md
- scopes/PROJECT_SCOPE.md
- FRIEND_AGENT_SETUP.md
- AGENTS.md

Your job:
1. Lead the concrete IR research idea (datasets, methods, eval metrics).
2. Collaborate via the Telegram collab_bridge (hourly ticks).
3. Implement experiments on THIS machine under branches `agent/friend/...`.
4. Post clear CLAIM / DONE / BLOCKED / ASK messages; tag `@agent:arush` when
   you need infra or scaffolding help.
5. Never push secrets. Never force-push main. Use our human's git author.

Setup checklist (do with the human if not done):
- Clone Information_Retrieval
- Copy .env.example → .env
- Set AGENT_ID=friend
- Set the SAME TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as Arush
- Set CURSOR_API_KEY for THIS Cursor account
- Set REPO_ROOT to this checkout
- Set GITHUB_REPO correctly
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- pip install cursor-sdk
- python -m collab_bridge check
- python -m collab_bridge ping
- Start daemon or cron: python -m collab_bridge daemon

After bridge is live: propose the IR idea as PROPOSE_SCOPE / CLAIM S1, then
break work into T-tasks and invite @agent:arush to take complementary pieces.
```

## Friend human setup (step by step)

### 1. Git identity (commits under *their* name)

```bash
git config --global user.name "Friend Name"
git config --global user.email "friend@email-used-on-github.com"
```

### 2. Clone the shared repo

```bash
git clone git@github.com:Arush777/Priority_KV.git
cd Information_Retrieval
```

(Use the real OWNER/url Arush shares.)

### 3. Telegram (same group as Arush)

If Arush already created the bot + group:

- Get `TELEGRAM_BOT_TOKEN` from Arush **via a private channel** (not GitHub).
- Get `TELEGRAM_CHAT_ID` (same negative id).
- Confirm the friend human is in the Telegram group.

If not yet created, see README § Telegram setup.

### 4. Configure `.env`

```bash
cp .env.example .env
# edit .env:
# AGENT_ID=friend
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
# CURSOR_API_KEY=...   # friend's Cursor API key
# REPO_ROOT=/absolute/path/to/Information_Retrieval
# GITHUB_REPO=Arush777/Priority_KV
# DRY_RUN=0
```

Cursor API key: [Cursor Dashboard → Integrations / API Keys](https://cursor.com/dashboard/integrations)

### 5. Install + smoke test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install cursor-sdk
python -m collab_bridge check
python -m collab_bridge ping
python -m collab_bridge tick    # one live collaboration tick
```

### 6. Keep the loop running

Foreground:

```bash
python -m collab_bridge daemon
```

Or cron hourly:

```bash
crontab -e
# example:
0 * * * * cd /path/to/Information_Retrieval && . .venv/bin/activate && python -m collab_bridge tick >> state/cron_friend.log 2>&1
```

Also: `./scripts/install_cron.sh` (edit paths first).

### 7. Tell agent `arush` you are online

In Telegram:

```text
[human] friend agent joining. @agent:arush ack when your bridge is live.
```

Then both daemons chat each hour: idea from friend → scaffolding from arush →
shared PRs.

## Emergency controls (either human)

- `STOP_BRIDGE` or `HALT_AGENTS` in the group → agents pause
- `RESUME_BRIDGE` → continue
- `ACK_SCOPE` → accept a pending scope proposal
