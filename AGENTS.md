# AGENTS.md — Information_Retrieval

## What this repo is

Collaborative **Information Retrieval** research prototype. Two Cursor agents
(`arush`, `friend`) coordinate over Telegram via `collab_bridge` and share code
on GitHub.

## Layout

| Path | Purpose |
|------|---------|
| `collab_bridge/` | Telegram poller + Cursor SDK hourly runner |
| `scopes/PROJECT_SCOPE.md` | Living project scope |
| `COLLAB.md` | Multi-agent protocol (required reading) |
| `FRIEND_AGENT_SETUP.md` | Onboarding packet for peer agent |
| `docs/` | Design notes / IR notes |
| `state/` | Local bridge state (gitignored) |
| `scripts/` | Cron helpers, setup checks |

## Commands

```bash
python -m collab_bridge check
python -m collab_bridge ping
python -m collab_bridge tick
python -m collab_bridge daemon
```

## Hard rules

1. Follow `COLLAB.md` message grammar.
2. Branches only under `agent/<AGENT_ID>/`.
3. Never commit `.env` or tokens.
4. Scope edits require `ACK_SCOPE` when `REQUIRE_SCOPE_ACK=1`.
5. Do not run jobs on the partner's cluster account.
6. Prefer small PRs; humans may be out of the loop for routine work but
   STOP_BRIDGE always wins.
