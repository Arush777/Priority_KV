# PriorityBench-A (`src/prioritybench`)

Workstream **S1** scaffolding for the agent-reliability benchmark in
`docs/PRIORITYKV_IMPLEMENTATION_PLAN.md` §3.

| Module | Role |
|--------|------|
| `schema.py` | Example contract, categories, splits, context strata |
| `scoring.py` | Deterministic scorers (JSON-schema subset, regex, exact slots) |

## Categories (80 each → 240)

1. `tool_schema` — tool-call / JSON validity under long context  
2. `instruction_supersession` — follow *latest* constraint  
3. `multi_turn_state` — verbatim reuse of early-turn IDs / paths  

## Run tests

```bash
PYTHONPATH=src python scripts/test_prioritybench_scoring.py
```

## Not in this PR

- Template engine / example generator (W1)  
- Dataset JSONL under `data/` (await peer ack on layout)  
- Model eval harness / vLLM runners (S2 / systems)  
