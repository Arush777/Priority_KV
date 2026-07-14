# Failure atlas (W3–W4) — scaffolding

**Gate G2 (end W4):** proceed with PriorityKV only if (a) uniform compression
drops ≥5 points on ≥1 PriorityBench-A category while guardrails move <1 point,
OR (b) oracle structure-aware allocation beats uniform INT4 by ≥3 points at
equal bytes. See `docs/IMPLEMENTATION_PLAN.md` §3.3.

## Status

| Baseline | Pilot | Status |
|---|---|---|
| FullKV (S0) | w2 / w2b 8k+16k | green on **leaky v1** templates |
| FP8 (S1) | w2b + w2c 16k | delta=0 (v1 too easy) |
| Uniform INT4 (Q2) | w2c / w2d 16k | soft/broken path → still ~1.0; not decisive |
| DropKeep ~64× | stress_dropkeep_16k | **HIT:** full=1.0 drop=0.0 all cats — first G2 signal |
| SnapKV (Q3) | — | scaffold / optional KVPress |

**2026-07-15:** `run_stress.py` → full=1.000 drop=0.000 @ ~63.8×. Agent reliability collapses under aggressive eviction.

## How to append rows

After a quality pilot, normalize into atlas CSV/JSONL:

```bash
python scripts/atlas_collect.py \
  --pilot $PRIORITYKV_SCRATCH/runs/w2c_pb_quality/w2c_pb_quality_16k_r1.json \
  --out $PRIORITYKV_SCRATCH/runs/atlas/rows.jsonl
```

Each row: `example_id, category, context_length, method, score, delta_vs_fullkv`.

## Headline figure F1 (draft)

X = context length · Y = category score · series = FullKV / FP8 / INT4 / SnapKV.
Expected shape once INT4 lands: flat FullKV+FP8, collapsing INT4 on tool_schema
and/or multi_turn_state.

## Oracle structure check (CPU)

`tests/test_locked_structure.py` — same budget, structural TOOL retention ≥
uniform TOOL demotion. Not yet the page-perturbation KL atlas (W4).
