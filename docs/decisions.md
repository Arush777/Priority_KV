# Decisions log (append-only)

Agents and humans append one line per durable decision.
Format: `YYYY-MM-DD | who | decision`

## Open

- 2026-07-14 | friend→arush | PriorityBench storage: committed JSONL under `data/prioritybench/{calibration,validation,test}/` + generator, OR generator+seeds only with JSONL gitignored?
- 2026-07-14 | arush→friend | Start W1 templates with tool-schema category first? Which Qwen3-8B chat-template version to pin for S2 page tagging?
- 2026-07-14 | friend | PROPOSE_SCOPE: mark S1 active (scaffold pushed); needs ACK_SCOPE / merge of S1 PR

## Decided

- 2026-07-14 | both | Shared Telegram bot + group is the agent↔agent channel; GitHub is long-term memory
- 2026-07-14 | both | Sticky memory: Telegram ring (last N) + docs/collab_memory.md + docs/decisions.md + Agent.resume per side
- 2026-07-14 | friend | CLAIM S1 (PriorityBench Workstream A); branch `agent/friend/s1-prioritybench-scaffold`
- 2026-07-14 | arush | CLAIM S0 hygiene/CI; branch `agent/arush/s0-hygiene-ci-smoke`
