# Priority_KV

GitHub: https://github.com/Arush777/Priority_KV

**Research plan (canon):** [`docs/PRIORITYKV_IMPLEMENTATION_PLAN.md`](docs/PRIORITYKV_IMPLEMENTATION_PLAN.md)  
**Agent collab protocol:** [`COLLAB.md`](COLLAB.md)  
**Friend onboarding:** [`FRIEND_AGENT_SETUP.md`](FRIEND_AGENT_SETUP.md)

Telegram group: `Information_Retrieval agents` (shared bridge; project is Priority_KV).

## What’s included

- `collab_bridge` — poll Telegram → run Cursor SDK agent → post status
- Protocol in `COLLAB.md` (CLAIM/DONE/ASK/STOP, branch rules)
- Friend onboarding packet: `FRIEND_AGENT_SETUP.md`
- Living scope: `scopes/PROJECT_SCOPE.md`

## You (Arush) — first-time setup

### 0. Git author = your name

```bash
git config --global user.name "Arush ..."
git config --global user.email "your-github-email@..."
```

### 1. Create the GitHub repo

`gh` is not installed on this cluster login. From a machine with GitHub access:

```bash
# repo already exists as Arush777/Priority_KV; from this checkout:
cd /u/arushh/Arush/Priority_KV
git push -u origin main
```

Add your friend as a collaborator with write access.

### 2. Telegram bot + group

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → save token.
2. Create a Telegram group; add you, your friend, and the bot.
3. BotFather → `/setprivacy` → **Disable** (so bot sees group messages). Remove + re-add bot if needed.
4. Post any message in the group, then:

```bash
export TELEGRAM_BOT_TOKEN='...'
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates" | python3 -m json.tool | head -80
```

Copy the group `chat.id` (negative number) → `TELEGRAM_CHAT_ID`.

### 3. Cursor API key

Create a key at [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations).

### 4. Configure this checkout

```bash
cd /u/arushh/Arush/Priority_KV
cp .env.example .env
```

Edit `.env`:

```bash
AGENT_ID=arush
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
CURSOR_API_KEY=...
REPO_ROOT=/u/arushh/Arush/Priority_KV
GITHUB_REPO=Arush777/Priority_KV
DRY_RUN=0
TICK_INTERVAL_SEC=3600
```

### 5. Install + verify

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install cursor-sdk
python -m collab_bridge check
python -m collab_bridge ping
```

You should see a message in Telegram: `[agent:arush] bridge online`.

### 6. Run the loop

```bash
# foreground (survives only while this process lives)
python -m collab_bridge daemon

# or hourly cron
./scripts/install_cron.sh
```

### 7. Onboard friend

Send them:

1. GitHub invite to `Arush777/Priority_KV`
2. Bot token + chat id **privately**
3. File `FRIEND_AGENT_SETUP.md` (they paste the prompt into *their* Cursor)

When both bridges ping, agents collaborate hourly on PriorityKV scope + code.

## Commands

| Command | Meaning |
|---------|---------|
| `python -m collab_bridge check` | Validate config / bot |
| `python -m collab_bridge ping` | Post online |
| `python -m collab_bridge tick` | One collaboration cycle |
| `python -m collab_bridge daemon` | Loop forever |
| `python -m collab_bridge resume` | Clear local pause |

## Human controls in Telegram

- `STOP_BRIDGE` / `HALT_AGENTS` — pause
- `RESUME_BRIDGE` — resume
- `ACK_SCOPE` — accept proposed scope change

## Architecture

```text
┌──────────────────┐         Telegram group         ┌──────────────────┐
│ agent:arush      │◄────── shared messages ───────►│ agent:friend     │
│ collab_bridge    │                                │ collab_bridge    │
│ + Cursor SDK     │                                │ + Cursor SDK     │
│ local cwd=repo   │──────── GitHub PRs/branches ───│ local cwd=repo   │
└──────────────────┘                                └──────────────────┘
```

## Safety

See `COLLAB.md`. Defaults: branch namespace `agent/<id>/`, stop keywords, optional
dry-run, scope ACK gate, no secret commits.
