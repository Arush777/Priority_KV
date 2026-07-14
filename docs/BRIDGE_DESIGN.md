# Design — collab bridge

## Goals

1. Two Cursor agents chat and split work without humans in the room.
2. Hourly wake cycle is durable (cron/daemon), not “IDE left open”.
3. GitHub remains source of truth; Telegram is the negotiation channel.

## Non-goals

- Native Cursor peer sockets
- Running jobs on the other person's cluster account
- Auto-merge to main without PR discipline

## Tick sequence

1. Acquire per-agent lock under `state/`
2. `getUpdates` with stored offset
3. Parse STOP/RESUME and peer/human messages
4. Build protocol prompt + transcript
5. Cursor SDK `Agent.create` + `send` against local checkout
6. Read `state/last_status_<id>.txt` and `sendMessage`
7. Persist offset + tick history

## Failure modes

| Failure | Handling |
|---------|----------|
| Missing SDK | Status posts install instructions |
| Overlapping ticks | File lock skip |
| Echo storms | Skip own `[agent:me]` messages |
| Human emergency | STOP_BRIDGE pauses |
| Scope creep | PROPOSE_SCOPE + ACK gate |

## Extending

- Swap Telegram for GitHub Issues comments (same protocol parser)
- Add webhook mode instead of getUpdates polling
- Add spend ledger before paid LLM calls outside Cursor
