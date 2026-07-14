# COLLAB — Priority_KV multi-agent protocol

Two Cursor agents (`arush` and `friend`) collaborate on PriorityKV research
with humans mostly out of the loop. Coordination bus: **Telegram group**.
Code source of truth: **GitHub** (`Arush777/Priority_KV`).

## Roles

| Agent ID | Account | Primary responsibility |
|----------|---------|------------------------|
| `arush` | Arush cluster login | Infra, bridge, implementation support |
| `friend` | Friend cluster/login | IR research idea leadership, experiments |

Either agent may claim implementation tasks. Cluster jobs stay on **that agent's
human account**.

## Message grammar (Telegram)

Every agent post is prefixed `[agent:<id>]` by the bridge.

| Pattern | Meaning |
|---------|---------|
| `@agent:arush ...` / `@agent:friend ...` | Directed request |
| `CLAIM T12` / `CLAIM S1` | Take ownership of task / workstream |
| `DONE T12` | Finished |
| `BLOCKED T12 <reason>` | Needs peer/human input |
| `PROPOSE_SCOPE <text>` | Propose scope change (writes wait for ACK) |
| `ACK_SCOPE` | Human (preferred) or peer accepts scope change |
| `STOP_BRIDGE` / `HALT_AGENTS` | Emergency pause (human) |
| `RESUME_BRIDGE` | Resume after pause (human) |

Task IDs: `T1`, `T2`, … for tasks; `S0`, `S1`, … for workstreams in
`scopes/PROJECT_SCOPE.md`.

## Hourly tick behavior

Each side runs `python -m collab_bridge daemon` (or cron `tick`).

1. Poll Telegram for new messages in the shared group.
2. Honor STOP / RESUME.
3. Ignore own prior `[agent:me]` posts.
4. Invoke Cursor SDK local agent on this checkout with protocol prompt.
5. Agent codes/pushes under `agent/<id>/...` branches and writes
   `state/last_status_<id>.txt`.
6. Bridge posts that status back to Telegram for the peer.

If the group is quiet, the tick is a **heartbeat**: pick a small unowned
improvement or ask the peer a concrete IR question.

## Git rules

- Branch namespace: `agent/<AGENT_ID>/...`
- Commits use the **machine owner's** `git config` identity (never invent a
  Cursor identity).
- Open PRs into `main`; do not force-push; never push `.env` or API keys.
- Prefer ≤ `MAX_COMMITS_PER_TICK` commits per tick.
- Review peer PRs when asked via `@agent:...`.

## Workload split

1. Post `CLAIM` before large work.
2. Do not touch files a peer claimed in the last 24h unless they `DONE` /
   `BLOCKED` or tagged you.
3. Prefer complementary streams: e.g. friend owns retrieval idea + eval design;
   arush owns scaffolding, scripts, bridge, reproducibility.
4. When stuck >1 tick, `BLOCKED` with a clear ask.

## Scope changes

- Edit `scopes/PROJECT_SCOPE.md` only after `ACK_SCOPE` if
  `REQUIRE_SCOPE_ACK=1`.
- `PROPOSE_SCOPE` must describe what is added/removed and why.

## Safety rails

- STOP keywords pause the bridge locally and notify Telegram.
- DRY_RUN mode posts plans without Cursor SDK.
- Agents must not spend paid external APIs without explicit human OK in chat.
- No destructive git (`reset --hard`, force-push main).

## Minimum human setup (once)

See `README.md` (you) and `FRIEND_AGENT_SETUP.md` (friend + their agent).
After Telegram + `.env` + cron/daemon are up, humans can leave the chat
except for ACK_SCOPE / STOP / spending approvals.
