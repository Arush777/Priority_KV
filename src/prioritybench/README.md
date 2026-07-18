# PriorityBench-A (`src/prioritybench`)

Custom **agent-reliability** benchmark for PriorityKV (not LongBench).

**Locked dataset docs:** [`docs/DATASET.md`](../../docs/DATASET.md)  
**Manifest:** `data/prioritybench/manifests/w3_lock.json` (n=240, audit PASS)

| Module | Role |
|--------|------|
| `schema.py` | Example contract, categories, splits, context strata |
| `scoring.py` | Deterministic scorers (JSON-schema subset, regex, exact slots) |
| `templates/` | `tool_schema` · `instruction_supersession` · `multi_turn_state` |
| `generate.py` | Seeded generator → JSONL splits |
| `pins.py` | Locked Qwen3-8B HF revision + `enable_thinking=False` |

## Categories (80 each → 240)

1. **`tool_schema`** — emit a valid tool call after long filler (early schema planted).  
2. **`instruction_supersession`** — obey the *latest* constraint when an earlier one conflicts.  
3. **`multi_turn_state`** — reuse early-turn order ID / file path / user pref verbatim.

Every example: plant state → long filler → final ask → programmatic 0/1 score.

## Storage

- **Committed:** generator, seeds, templates, scorers, fixtures, **manifest**  
- **Gitignored:** `data/prioritybench/{calibration,validation,test}/` JSONL (rebuild locally)

```bash
PYTHONPATH=src uv run python scripts/mk_bench.py --mode w3_lock
PYTHONPATH=src uv run python scripts/audit_bench.py
```
