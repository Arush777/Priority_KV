# PriorityBench-A (`src/prioritybench`)

Workstream **S1** for the agent-reliability benchmark in
`docs/PRIORITYKV_IMPLEMENTATION_PLAN.md` §3.

| Module | Role |
|--------|------|
| `schema.py` | Example contract, categories, splits, context strata |
| `scoring.py` | Deterministic scorers (JSON-schema subset, regex, exact slots) |
| `templates/` | W1 template engine (starts with `tool_schema`) |
| `generate.py` | Seeded generator → JSONL splits |
| `pins.py` | Locked Qwen3-8B HF revision + `enable_thinking=False` |

## Categories (80 each → 240)

1. `tool_schema` — tool-call / JSON validity under long context  
2. `instruction_supersession` — follow *latest* constraint  
3. `multi_turn_state` — verbatim reuse of early-turn IDs / paths  

## Storage (locked)

- **Committed:** generator, seeds, templates, scorers, small fixtures under
  `data/prioritybench/fixtures/`
- **Gitignored:** generated splits
  `data/prioritybench/{calibration,validation,test}/`

## Generate W1 pilot (40 tool_schema examples)

```bash
PYTHONPATH=src python scripts/generate_prioritybench.py --n 40 \
  --fixture data/prioritybench/fixtures/tool_schema_smoke.jsonl
```

## Run tests

```bash
PYTHONPATH=src python scripts/test_prioritybench_scoring.py
PYTHONPATH=src python scripts/test_prioritybench_generate.py
```
