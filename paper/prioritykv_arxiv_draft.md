# PriorityKV: Structure-Aware KV Retention for Long Agent Traces

**Status:** draft tech report (arXiv-bound) · freeze `G4_MIDDLE_GROUND_2026_07_19` + publish-track jobs pending  
**Authors:** Arush (PriorityKV)  
**Model:** Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`  
**Hardware:** NVIDIA H200

## Abstract

Autoregressive Transformers store conversation history in a KV cache that dominates memory for long multi-turn *agent* traces. Serving stacks therefore compress or evict KV. We show that **uniform eviction / missing-state** can preserve average-looking metrics while destroying tool-schema conformance, instruction supersession, and multi-turn IDs. **Structure-aware retention** restores those capabilities at matched keep budgets. Soft INT4 quantization at matched `int4_frac=0.75` does **not** open a PriorityBench quality gap (falsified). We therefore evaluate a structure-protected mixed BF16/INT4 packed cache with FlashInfer-backed decode on H200 for **honest packed bytes and latency**, with quality matched to FullKV on a locked 240-example agent bench. A vLLM FP8 head-to-head, expanded guardrails, and optional Gemma secondary are included in the publish-track appendix as they complete.

## 1. Introduction

Long agent sessions pack tool schemas, superseding instructions, and persistent IDs into the prompt. Dropping the wrong tokens (StreamingLLM-style sink+recent, SnapKV-class eviction) is a reliability failure, not only a compression knob. PriorityKV studies (i) which tokens matter for agent reliability at matched keep budgets, and (ii) whether a role-aware mixed BF16/INT4 paged cache can deliver systems value via **bytes + latency** when soft INT4 does not separate quality.

**Contributions.**

1. PriorityBench-A (240 locked examples, 3 agentic categories) with audit SHA.
2. Matched-keep evidence: structure ≫ uniform eviction; soft-INT4 quality gap falsified.
3. Packed BF16/INT4 + FI shim decode on H200 with phase-honest latency and peak/payload reporting.
4. Publish-track comparisons: vLLM FP8 systems table; RULER/SCBench/MATH-style guardrails; optional Gemma reduced keep matrix.

## 2. Locked claim (do not overclaim)

> Uniform KV **eviction / missing-state** silently corrupts tool schemas and instruction hierarchies. Structure-aware retention preserves those traces at matched keep budgets. Soft INT4 at `int4_frac=0.75` does not open a PriorityBench quality gap. Systems value is packed payload bytes + honest H200 latency (e2e≈FullKV, TPOT ~1.2× FI shim) with quality matched to FullKV, with cold-scratch caveats on peak VRAM.

## 3. PriorityBench-A and reliability (family A)

- Manifest: `data/prioritybench/manifests/w3_lock.json` · SHA256 `fc44b966…ae89` · n=240.
- Matched keep @ 0.25: structure recovers agent scores where uniform collapses (see W2–W4 structured stress jobs).
- Zero-degrade wiring proof: structure ≫ uniform at matched masks.

## 4. Soft INT4 quality (falsified)

Corrected mixed forwards @ int4_frac=0.75 (and 2-bit severity): uniform and structure both ≈ FullKV. Do not claim an INT4 *quantization* quality win.

## 5. Systems: packed mixed cache + FI decode

Canonical middle-ground jobs (see `FINAL_RUN_MANIFEST.yaml`):

| Job | Result |
|---|---|
| `d4_latency_m3c_gpu56_r1` | D4_M3_PASS — e2e ~1.11–1.12× FullKV; TPOT ~1.20–1.21×; pack/cold tens of ms |
| `mg_a_peak_mem_gpu5_r1` | peak ~0.87× FullKV; measured payload ~0.72×; modeled ~0.47× |
| `mg_b_lock240_quality_gpu01_r1` | full 0.888 / structure 0.883 / uniform 0.879 (n=240) |

**32k note:** all arms drop (multi-turn hard); structure slightly above uniform, both near FullKV.

## 6. Publish-track additions (fill when jobs return)

### 6.1 vLLM FP8 head-to-head

Job: `pub_a_d4_fp8_compare_gpu01_r1`. Arms: FullKV SDPA, structure-FI @ 0.75, vLLM FP8. Report quality + e2e/TPOT + modeled byte ratios. If structure does not beat FP8 on latency, use **reliability-at-parity** reframe.

### 6.2 Guardrails

Job: `pub_b_guardrails_gpu5_r1`. FullKV vs structure-mixed on local RULER/SCBench-style probes + MATH-500 subsample if available. Gate: |Δ| ≤ 1pt on gate tasks.

### 6.3 Gemma secondary

Job: `pub_c_gemma_reduced_gpu5_r1`. Reduced matched-keep structure vs uniform, or `SKIP_NO_GEMMA`.

## 7. Limitations

- Custom PriorityBench, not full LongBench/RULER paper matrices.
- FI cold scratch expands INT4→BF16 for attend — do not oversell peak VRAM.
- FP8 e2e in compare harness is batch-amortized wall/n (not identical phase protocol to HF).
- Single primary model (Qwen3-8B) unless Gemma lands.

## 8. Reproducibility

- Repo: `github.com:Arush777/Priority_KV`
- Freeze: `FINAL_RUN_MANIFEST.yaml`
- H200 worker: `jobs/pending` + `scripts/remote_worker.sh`
- Pins: model revision + bench SHA above

## References

Internal: `docs/decisions.md`, `docs/APPLIED_SCIENCE_REVIEW_BRIEF.md`, job results under `jobs/results/`.
