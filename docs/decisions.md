# Decisions log (append-only)

Agents and humans append one line per durable decision.
Format: `YYYY-MM-DD | who | decision`

## Open

- 2026-07-14 | friend | PROPOSE_SCOPE: mark S1 active (scaffold + W1 templates landed); needs ACK_SCOPE / merge of S1 PR
- 2026-07-14 | human | open/merge PRs (`gh` missing on some logins)

## Decided

- 2026-07-14 | both | Shared Telegram bot + group is the agent↔agent channel; GitHub is long-term memory
- 2026-07-14 | both | Sticky memory: Telegram ring (last N) + docs/collab_memory.md + docs/decisions.md + Agent.resume per side
- 2026-07-14 | friend | CLAIM S1 (PriorityBench Workstream A); branch `agent/friend/s1-prioritybench-scaffold`
- 2026-07-14 | arush | CLAIM S0 hygiene/CI; branch `agent/arush/s0-hygiene-ci-smoke`
- 2026-07-14 | friend | PriorityBench storage: commit generator+seeds+small fixtures under `data/prioritybench/fixtures/`; generated JSONL under `data/prioritybench/{calibration,validation,test}/` is gitignored (plan §3.2 publish list)
- 2026-07-14 | friend | W1 starts with `tool_schema` category first; `instruction_supersession` + `multi_turn_state` follow after ≥40 tool_schema examples exist
- 2026-07-14 | friend | Pin Qwen3-8B for S2 page tagging/eval: HF `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`; `apply_chat_template(..., enable_thinking=False)` for agent-trace replay
