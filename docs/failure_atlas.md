# Failure atlas (W3–W4)

**Gate G2 (end W4):** proceed with PriorityKV only if (a) uniform compression
drops ≥5 points on ≥1 PriorityBench-A category while guardrails move <1 point,
OR (b) oracle structure-aware allocation beats uniform by ≥3 points at
equal bytes. See `docs/IMPLEMENTATION_PLAN.md` §3.3.

## Status

| Baseline | Pilot | Status |
|---|---|---|
| FullKV (S0) | w2 / w2b 8k+16k | green |
| FP8 (S1) | w2b + w2c 16k | delta≈0 ≤16k (not the stress) |
| Uniform INT4 (Q2) | `w3_int4_assert_r4` | **real path GREEN** · modes=`hf_cache_implementation_quantized` · n=6 · int4=1.000 (soft @8k — path a still needs harder stress) |
| DropKeep ~64× | stress_dropkeep_16k | **HIT:** full=1.0 drop=0.0 |
| Structure @25% token | stress_structured_25 | **HIT:** structure=1.0 vs uniform/random=0 |
| Structure @25% page | w3_structured_paged_r1 | **HIT:** structure=0.643 vs uniform=0.000 |
| Buried adversarial | stress_structured_25_buried | **scoped:** structure=0.43 |
| Structure denser | w4 `0.15` / `0.35` page | **GREEN:** structure=0.643 vs uniform=0.000 at both (random 0.071→0.429) |
| SnapKV (Q3) | `w4_snapkv_quality_r1` | import OK; matched-byte quality job enqueued |
| Guardrails | guardrails_w4_r2 | **PASS** gate Δ=0 (`ruler_vt`+`scbench_choice`) |
| FlashInfer CUDA | flashinfer_multicall | **DEFERRED** W5–6 (CPU LSE parity ✅) |

**W3 closed · W4 G2 path (b) closed (2026-07-15):** denser keep curves + guardrails + formal G2. Atlas rows: `docs/atlas_w4_structure_rows.jsonl`.

## How to append rows

```bash
python scripts/atlas_collect.py \
  --pilot $PRIORITYKV_SCRATCH/runs/stress_structured/w3_structured_paged_r1.json \
  --out $PRIORITYKV_SCRATCH/runs/atlas/rows.jsonl
```

Each row: `example_id, category, context_length, method, score, delta_vs_fullkv`.

## Headline figure F1 (draft)

X = context length · Y = category score · series = FullKV / FP8 / INT4 / structure / DropKeep.
Path-(b) story: structure retains tool schemas where uniform keep collapses.
