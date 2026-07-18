# PriorityKV — what this project is and what we shipped

**Freeze:** `SCIENCE_CORE_HOME_2026_07_19` · tip documents this close  
**Model:** Qwen3-8B @ `b968826d9c46…` · H200 (`dgre2`)  
**D3:** CLOSED — [`docs/D3_CLOSE.md`](docs/D3_CLOSE.md)

## Point of the project

Long agent chats stuff **tool schemas, superseding instructions, and IDs** into the KV cache.
Serving stacks **compress or evict** that cache. If you drop the wrong tokens, the model can
look fine on average metrics while **silently breaking agent behavior**.

**PriorityKV** shows:

1. **Uniform eviction** destroys agent reliability at matched keep budgets.
2. **Structure-aware keep** (protect system/tool/constraint/sink/recent) restores it.
3. Soft **INT4 quantization alone does not** open a PriorityBench quality gap at `int4_frac=0.75`
   (that hypothesis was **falsified**).
4. So systems value is a **packed BF16/INT4 cache + FlashInfer decode**: real **payload bytes**
   + honest **latency**, with quality matched to FullKV on a locked agent bench — not a fake
   INT4 accuracy win.

## What we actually built

| Layer | Artifact |
|---|---|
| Bench | PriorityBench-A · 240 locked examples · 3 agent categories · audit SHA |
| Reliability | Structure ≫ uniform matched-keep (token + page); buried-state scoped claim |
| Mixed cache | Role planner · true packed INT4 pages · FI LSE multicall · FI decode shim |
| Systems metrics | Pack/cold/e2e/TPOT · peak + payload (cold-scratch caveat) |
| Publish appendix | FP8 head-to-head · guardrails · Gemma reduced secondary |

## Canonical metrics (cite these)

### Lock-240 quality @ int4_frac=0.75 (packed FI) — `mg_b_lock240_quality_gpu01_r1`

| Arm | Mean score (n=240) |
|---|---|
| FullKV | **0.888** |
| Structure-mixed | **0.883** |
| Uniform-mixed | **0.879** |

By length: 8k/16k all **1.0**; 32k drops for all (~0.62–0.65). Soft-INT4 does **not** separate quality.

### Latency — `d4_latency_m3c_gpu56_r1` (`D4_M3_PASS`)

Structure-FI vs FullKV (order of magnitude): e2e ~**1.11–1.12×** · TPOT ~**1.20–1.21×** · pack/cold tens of ms.

### Peak / payload — `mg_a_peak_mem_gpu5_r1` (`MG_PEAK_MEM_PASS`)

| Metric | vs FullKV |
|---|---|
| Peak CUDA | ~**0.87×** |
| Measured packed payload | ~**0.72×** |
| Modeled compression | ~**0.47×** |

**Caveat:** FI cold scratch expands INT4→BF16 for attend — do **not** claim peak ≪ FullKV.

### Matched-keep reliability (earlier decisive runs)

| Setting | Uniform | Structure |
|---|---|---|
| Token keep_frac=0.25 | **0.000** | **1.000** |
| Page keep @0.25 | **0.000** | **0.643** |
| Buried gold (token) | 0.000 | **0.429** (scoped — not oracle) |

### Publish track

| Job | Decision | Note |
|---|---|---|
| `pub_a_d4_fp8_compare_gpu01_r1` | **D4_FP8_COMPARE_PASS** | FullKV vs FP8 vs structure-FI |
| `pub_b_guardrails_gpu5_r1` | **GUARDRAILS_PUB_PASS** | gate Δ=**0.0** |
| `pub_c_gemma_reduced_gpu01_r6` | **GEMMA_REDUCED_PASS** | n=14 · full **0.36** / structure **0.14** / uniform **0.00** |

## What we are *not* claiming

- Soft INT4 accuracy win on PriorityBench  
- Peak VRAM collapse (cold scratch)  
- Full LongBench/RULER paper matrices  
- Gemma = Qwen lock-240 absolute scores (reduced secondary only)

## Still open (DeepMind packaging, not GPU science)

arXiv submit · blog go-live · upstream PR · outreach · interview prep.

## Source of truth

- Dataset (tasks): [`docs/DATASET.md`](docs/DATASET.md)  
- Freeze: [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml)  
- Next chat: [`docs/NEXT_CHAT_HANDOFF.md`](docs/NEXT_CHAT_HANDOFF.md)  
- Decisions log: [`docs/decisions.md`](docs/decisions.md)  
- Paper draft: [`paper/prioritykv_arxiv_draft.md`](paper/prioritykv_arxiv_draft.md)  
- Job status: `jobs/status/<id>.json` · done YAMLs: `jobs/done/`
