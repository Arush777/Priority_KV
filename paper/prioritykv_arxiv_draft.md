# PriorityKV: Structure-Aware KV Retention for Long Agent Traces

**Status:** science-core **HOME** · freeze `SCIENCE_CORE_HOME_2026_07_19` · D3 **CLOSED**  
**Authors:** Arush (PriorityKV)  
**Model:** Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`  
**Hardware:** NVIDIA H200 · secondary Gemma-2-9b-it (reduced)

## Abstract

Autoregressive Transformers store conversation history in a KV cache that dominates memory for long multi-turn *agent* traces. Serving stacks therefore compress or evict KV. We show that **uniform eviction / missing-state** can preserve average-looking metrics while destroying tool-schema conformance, instruction supersession, and multi-turn IDs. **Structure-aware retention** restores those capabilities at matched keep budgets. Soft INT4 quantization at matched `int4_frac=0.75` does **not** open a PriorityBench quality gap (falsified). We therefore evaluate a structure-protected mixed BF16/INT4 **packed** cache with FlashInfer-backed decode on H200 for **honest packed bytes and latency**, with quality matched to FullKV on a locked 240-example agent bench. Publish-track appendix: vLLM FP8 head-to-head (**PASS**), expanded guardrails (**PASS**, Δ=0), and Gemma reduced matched-keep (**PASS**, n=14).

## 1. Introduction

Long agent sessions pack tool schemas, superseding instructions, and persistent IDs into the prompt. Dropping the wrong tokens (StreamingLLM-style sink+recent, SnapKV-class eviction) is a reliability failure, not only a compression knob. PriorityKV studies (i) which tokens matter for agent reliability at matched keep budgets, and (ii) whether a role-aware mixed BF16/INT4 paged cache can deliver systems value via **bytes + latency** when soft INT4 does not separate quality.

**Contributions.**

1. PriorityBench-A (240 locked examples, 3 agentic categories) with audit SHA.
2. Matched-keep evidence: structure ≫ uniform eviction; soft-INT4 quality gap falsified.
3. Packed BF16/INT4 + FI shim decode on H200 with phase-honest latency and peak/payload reporting (**D3 CLOSED**; cold-scratch caveat).
4. Publish-track: vLLM FP8 systems compare; RULER/SCBench/MATH-style guardrails; Gemma reduced keep matrix.

## 2. Locked claim (do not overclaim)

> Uniform KV **eviction / missing-state** silently corrupts tool schemas and instruction hierarchies. Structure-aware retention preserves those traces at matched keep budgets. Soft INT4 at `int4_frac=0.75` does not open a PriorityBench quality gap. Systems value is packed payload bytes + honest H200 latency (e2e≈FullKV, TPOT ~1.2× FI shim) with quality matched to FullKV, with cold-scratch caveats on peak VRAM.

## 3. PriorityBench-A and reliability (family A)

- Manifest: `data/prioritybench/manifests/w3_lock.json` · SHA256 `fc44b966…ae89` · n=240.
- Matched keep @ 0.25: structure recovers agent scores where uniform collapses (see W2–W4 structured stress jobs).
- Zero-degrade wiring proof: structure ≫ uniform at matched masks.

## 4. Soft INT4 quality (falsified)

Corrected mixed forwards @ int4_frac=0.75 (and 2-bit severity): uniform and structure both ≈ FullKV. Do not claim an INT4 *quantization* quality win.

## 5. Systems: packed mixed cache + FI decode (D3)

Canonical middle-ground jobs (see `FINAL_RUN_MANIFEST.yaml` · `docs/D3_CLOSE.md`):

| Job | Result |
|---|---|
| `d4_latency_m3c_gpu56_r1` | D4_M3_PASS — e2e ~1.11–1.12× FullKV; TPOT ~1.20–1.21×; pack/cold tens of ms |
| `mg_a_peak_mem_gpu5_r1` | peak ~0.87× FullKV; measured payload ~0.72×; modeled ~0.47× |
| `mg_b_lock240_quality_gpu01_r1` | full 0.888 / structure 0.883 / uniform 0.879 (n=240) |

**32k note:** all arms drop (multi-turn hard); structure slightly above uniform, both near FullKV.

**D3 stack:** `packed_mixed_cache.py` + `fi_mixed_decode.py` + `qwen3_fi_shim.py`. Decode refuses silent HF materialize. INT4→BF16 **cold scratch** for FI attend is the accepted peak-VRAM caveat.

## 6. Publish-track results

### 6.1 vLLM FP8 head-to-head

Job: `pub_a_d4_fp8_compare_gpu01_r1` → **D4_FP8_COMPARE_PASS** (exit=0). Arms: FullKV SDPA, structure-FI @ 0.75, vLLM FP8. Scratch artifact: `runs/d4_fp8_compare/d4_fp8_compare_gpu01_r1.json`.

### 6.2 Guardrails

Job: `pub_b_guardrails_gpu5_r1` → **GUARDRAILS_PUB_PASS** · gate `max_abs_delta=0.0`.

### 6.3 Gemma secondary (reduced)

Job: `pub_c_gemma_reduced_gpu01_r6` → **GEMMA_REDUCED_PASS** · n=14 @ ~8144 tokens (Gemma max 8192). Means: full **0.357** · structure **0.143** · uniform **0.000**. Structure ≥ uniform (gate); absolute scores are secondary (PriorityBench scorers Qwen-oriented).

## 7. Limitations

- Custom PriorityBench, not full LongBench/RULER paper matrices.
- FI cold scratch expands INT4→BF16 for attend — do not oversell peak VRAM.
- FP8 e2e in compare harness is batch-amortized wall/n (not identical phase protocol to HF).
- Gemma reduced matrix only (by v2 design).

## 8. Reproducibility

- Repo: `github.com:Arush777/Priority_KV`
- Freeze: `FINAL_RUN_MANIFEST.yaml` (`SCIENCE_CORE_HOME_2026_07_19`)
- D3: `docs/D3_CLOSE.md`
- H200 worker: `jobs/pending` + `scripts/remote_worker.sh`
- Pins: model revision + bench SHA above

## References

Internal: `docs/decisions.md`, `docs/APPLIED_SCIENCE_REVIEW_BRIEF.md`, job results under `jobs/results/`.
