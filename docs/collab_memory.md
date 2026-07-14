# Shared collab memory (Priority_KV)

Updated by each agent tick. Read this every tick.
Per-agent detail also lives in `state/collab_memory_<id>.md` (local, gitignored).

## Current picture

- Project: Priority_KV / PriorityKV-Agent
- Bridges: arush + friend
- Sticky memory enabled: ring + resume + decisions.md
- Friend: S1 W1 tool_schema templates + generator on `agent/friend/s1-prioritybench-scaffold`
- Arush: S0 hygiene/CI branch (`agent/arush/s0-hygiene-ci-smoke`)
- Pins: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, `enable_thinking=False`

## Open asks

- Human: open/merge PRs (`gh` missing on some logins); ACK_SCOPE to mark S1 active
- Arush: please ACK storage+Qwen pin decisions; wire S2 page tagging to `prioritybench.pins`

## Recent tick notes

- 2026-07-14 friend: locked JSONL storage (fixtures committed, splits gitignored); W1 starts tool_schema; shipped 4 templates + 40-ex pilot generator
