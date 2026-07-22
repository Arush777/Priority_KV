# PriorityKV: Results

**Freeze:** `SCIENCE_CORE_HOME_2026_07_19`
**Authors:** Arush Sharma (IIT (ISM) Dhanbad) · Anupam Rawat (IIT Bombay)
**Model:** Qwen3-8B @ `b968826d9c46…` · H200 (`dgre2`)  

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
| Secondary check | Gemma reduced stress slice |

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

### Secondary model check

| Job | Decision | Note |
|---|---|---|
| `pub_c_gemma_reduced_gpu01_r6` | **GEMMA_REDUCED_PASS** | n=14 · full **0.36** / structure **0.14** / uniform **0.00** |

## Credibility track (P0–P3) — post-freeze H200

**External-audit response + checklist:** [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

### P0 — structure vs uniform/random (Qwen, n=120)

| Arm | Pooled mean |
|---|---|
| structure | **0.933** |
| uniform | **~0.008** |
| random | **~0.008** |

Jobs: `p0_w5_s{0,1,2}_kf25_token_*`.

**Placement controls (s0):** mid-context → structure=full=**0.975** (Δ0); buried → structure **0.675** < full **0.900**. Jobs: `p0a_…_middle_…`, `p0b_…_buried_…`. **Do not claim structure > FullKV.**

### P1 — structure vs attention eviction (Qwen, n=120)

| Arm | Pooled mean |
|---|---|
| structure | **0.933** (112/120) |
| SnapKV / Pyramid / hybrid | **0.900** (108/120) |
| H2O | **0.683** = (0.725 chunked s0 + 0.625 s1 + 0.700 s2) / 3 |

McNemar structure vs SnapKV: b=4, c=0, exact two-sided **p=0.125** — [`jobs/results/p1_structure_vs_snapkv_mcnemar.json`](jobs/results/p1_structure_vs_snapkv_mcnemar.json).  
Phrase as **matches or slightly exceeds** SnapKV-class; hybrid **equals** SnapKV (no complementarity).

### P2 — streamed cold attend (smoke)

Job `p2_fi_stream_cold_16k_gpu1_r1`: exit=0. Log peak_gib ≈ **36.4** (structure/uniform).
`summary.json` reconstructed from `log_full.txt` (smoke hygiene). Not a systems result; cite frozen D4/MG for latency/peak.

### P3 — Llama-3.1-8B

| Setting | Result |
|---|---|
| kf=0.25 n=120 | structure+attn arms **1.000** (easy-task ceiling) |
| Retention audit s0 | gold in sink+recent **0.0**; structure gold kept **1.0**; uniform **0.0** → **not** a port artifact |
| kf=0.05 s0 | SnapKV **1.0** > structure **0.875** |
| kf=0.05 s1 | SnapKV **1.0** > structure **0.900** (replicate) |

CPU artifacts: `jobs/results/audit_retention_{qwen,llama}_s0_kf25_summary.json`.

## What we are *not* claiming

- Soft INT4 accuracy win on PriorityBench  
- Peak VRAM collapse (cold scratch)  
- Full LongBench/RULER paper matrices  
- Gemma = Qwen lock-240 absolute scores (reduced secondary only)
- Structure beats FullKV  
- Statistically significant structure≫SnapKV on Qwen  
- Universal Llama transfer / hybrid complementarity

## External evaluation — `EXTERNAL_BFCL_PRAJNA_V1`

**Separate freeze. Does not modify any number above.** Benchmarks we did not author:
BFCL V3 multi-turn (Gorilla `cd9429cc`, official `multi_turn_checker`, unmodified)
and public τ-bench trajectories (`AgentSuite/tau-bench-trajectories` `382e57d1`).
Qwen3-8B, 25% keep, L40S/sm_89. All non-FullKV arms are kvpress presses over an
identical full prefill, so arms differ only in *which* KV entries survive.

### BFCL V3 multi-turn — n=141 paired conversations

| Arm | Qwen3-8B (n=140) | Llama-3.1-8B (n=143) |
|---|---:|---:|
| FullKV | **0.193** | **0.077** |
| SnapKV (attention) | **0.136** | **0.084** |
| ADAPT (ours) | **0.129** | — |
| Structure | **0.000** | **0.000** |
| Uniform | 0.000 | 0.000 |
| Random (corrected) | 0.000 | 0.000 |

Paired completeness 0.933 / 0.953 · exclusions all `MODEL_CONTEXT_LIMIT` ·
**0 matched-budget violations**. The result replicates across two architectures.

| Comparison | exact McNemar | Δ | 95% CI |
|---|---:|---:|---|
| FullKV vs Structure | **1.5e-08** | +0.191 | [+0.128, +0.262] |
| FullKV vs SnapKV | **0.152** (n.s.) | +0.057 | [−0.007, +0.128] |
| Structure vs SnapKV | **3.8e-06** | −0.135 | [−0.191, −0.078] |

**Structure-aware retention does not transfer to BFCL.** SnapKV is statistically
indistinguishable from FullKV at a 4× budget; structure is significantly worse
than both. This does not contradict PriorityBench-A — it bounds it. Llama-3.1-8B
reproduces every sign: FullKV vs SnapKV **p=1.0**, FullKV vs structure
**p=9.8e-04**, structure vs SnapKV **p=4.9e-04**.

### ADAPT — structure as a budget-relative prior

`alpha = min(1, keep_budget / protected_mass)`, blending rank-normalised structure
and attention scores. Alpha uses only quantities known from the prompt: no tuning,
no fitting. The formula was frozen in the config *before* any ADAPT result existed.

Measured alpha on BFCL: **mean 0.267** (min 0.250, max 0.401) over 833 generation
steps, against **~0.25 predicted** from the 98.8% protected fraction — a prediction
made in advance and confirmed.

| Comparison | exact McNemar | Δ | 95% CI |
|---|---:|---:|---|
| ADAPT vs SnapKV | **1.000** (n.s.) | −0.007 | [−0.057, +0.071] |
| ADAPT vs FullKV | 0.108 (n.s.) | −0.064 | [−0.136, +0.007] |
| ADAPT vs Structure | **7.6e-06** | +0.129 | [+0.079, +0.186] |

**ADAPT ties SnapKV and is indistinguishable from FullKV, on a workload where the
structure policy it generalises scores exactly zero.** It does *not* beat SnapKV —
the claim is "never worse, and it recovers attention-level behaviour automatically
from a measurement rather than a hand-chosen policy." At alpha=1 it provably
selects exactly what the structure arm selects, so it subsumes the frozen policy.

### The boundary condition (why)

Structure can only express a preference while protected mass stays *under* the
keep budget. Measured at `keep_frac=0.25`:

| Workload | Protected tokens | Oversubscribed | Structure |
|---|---:|---:|---:|
| PriorityBench-A | **6.1%** | 0% | **0.933** |
| τ-bench | 79.5% | 99% | retention-only |
| BFCL | **98.8%** | 100% | **0.000** |

PriorityBench-A is **94.9% filler** — exactly the regime where "protect structure,
drop filler" wins. A BFCL system prompt *is* 32 JSON tool schemas, so ~98% of
tokens carry the protected `TOOL` role and the policy has nothing to discard;
it degenerates to index order. See `scripts/analyze_protected_fraction.py`.

### τ-bench gold-span retention — 4,856 trajectories, 828k spans

Generation-free, CPU-only. **Mechanistic evidence, not task success.**

| Span class | n | Structure | Uniform |
|---|---:|---:|---:|
| **explicit policy** | 82,971 | **0.820** | 0.001 |
| tool name | 37,161 | 0.128 | 0.222 |
| tool-call argument | 47,441 | 0.131 | 0.293 |
| reused identifier | 276,237 | 0.055 | 0.315 |
| reused tool result | 383,364 | 0.069 | 0.140 |
| correction | 1,682 | 0.064 | 0.392 |

Structure retains durable policy constraints ~680× better; recency wins on
recently-referenced values. Same boundary, independent measurement.

### Defects found in the frozen core

1. **The published `random` baseline is byte-identical to `uniform`.**
   `select_random` sets `recent = budget - sink_tokens`, so the forced block fills
   the budget and the RNG branch never executes. The `~0.008` random column above
   is therefore *not* an independent control. Frozen code left untouched; this
   namespace uses a corrected `select_random_external`.
2. **Reasoning blocks were discarded.** A `<think>…</think>` prefix was passed
   whole to the official decoder, so correct tool calls decoded to nothing.
   Fixing it moved FullKV 0.000 → 0.105.
3. **Arms compared two different mechanisms** (prompt deletion vs KV eviction).
   Rebuilt as kvpress presses throughout.

### Not claimed from the external evaluation

- Structure beats SnapKV on any external benchmark — the opposite is measured
- τ-bench task success (the audit is retention-only; no simulator, no generation)
- Cross-model generality (Qwen3-8B only at time of writing)
- Any revision to the frozen PriorityBench-A numbers above

Artifacts: `configs/external_bfcl_prajna_v1.yaml` (incl. all deviations),
`$PRAJNA_ROOT/results/external_bfcl_prajna_v1/summaries/`.

## Source of truth

- Evidence track: [`docs/EVIDENCE.md`](docs/EVIDENCE.md)  
- Dataset (tasks): [`docs/DATASET.md`](docs/DATASET.md)  
- Freeze: [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml)  
- Manuscript: [`paper/prioritykv_manuscript.md`](paper/prioritykv_manuscript.md)
- Reproduction guide: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- Job manifests and bundles: [`jobs/`](jobs/)
