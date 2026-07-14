# Your checklist (Arush) — do these next

Bridge code is ready under `Priority_KV/`. Finish secrets + confirm GitHub access.

## 1. Git name

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

## 2. Telegram

1. @BotFather → `/newbot` → token  
2. Create group; add you, friend, bot  
3. `/setprivacy` → Disable; re-add bot  
4. `cp .env.example .env` and fill token  
5. `./scripts/telegram_smoke.sh` → copy group `chat.id` into `.env`

## 3. Cursor key

Dashboard → API key → `CURSOR_API_KEY` in `.env`  
Also set:

```bash
AGENT_ID=arush
REPO_ROOT=/u/arushh/Arush/Priority_KV
GITHUB_REPO=Arush777/Priority_KV
DRY_RUN=0
```

## 4. Smoke

```bash
cd /u/arushh/Arush/Priority_KV
source .venv/bin/activate   # already created
pip install cursor-sdk      # for live agent ticks
python -m collab_bridge check
python -m collab_bridge ping
PYTHONPATH=. python scripts/test_protocol.py
PYTHONPATH=src python scripts/test_metrics.py
```

## 5. GitHub remote

Remote already exists: https://github.com/Arush777/Priority_KV — invite friend with write access. Share bot token + chat id **privately**.

## 6. Start loop

```bash
python -m collab_bridge daemon
# or
./scripts/install_cron.sh
```

## 7. Friend

Send them `FRIEND_AGENT_SETUP.md` + invite them to paste the prompt into their Cursor.
