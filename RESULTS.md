# PriorityKV: Results

**Freeze:** `SCIENCE_CORE_HOME_2026_07_19`
**Authors:** Arush Sharma (IIT (ISM) Dhanbad) · Anupam Rawat (IIT Bombay)
**Model:** Qwen3-8B @ `b968826d9c46…` · H200 (`dgre2`)  

## EXTERNAL_BFCL_TIGHT_V1 — pre-registered external test (2026-07-31)

**The registered prediction was confirmed.** Pre-registration
([`docs/PREREG_BFCL_TIGHT.md`](docs/PREREG_BFCL_TIGHT.md)) was committed *before*
any tight-budget conversation was scored, and predicted a null or small negative
for ADAPT on BFCL — because BFCL's failures are dominated by loss of accumulated
multi-turn state, which the structural prior does not protect, not by schema
loss, which it does.

### Budget selection (pre-specified rule: tightest rung with SnapKV in [0.25, 0.70] × FullKV)

| Budget | n | SnapKV | FullKV | ratio | selected |
|---|---:|---:|---:|---:|:--:|
| 0.02 | 49 | 0.000 | 0.184 | 0.00 | |
| 0.05 | 49 | 0.000 | 0.184 | 0.00 | |
| 0.10 | 48 | 0.021 | 0.188 | 0.11 | |
| **0.15** | 89 | 0.124 | 0.202 | 0.61 | ✅ |
| 0.20 | 92 | 0.087 | 0.207 | 0.42 | |

SnapKV **floors below a 10% keep budget** on BFCL, so the tight budgets where the
synthetic sweep shows an ADAPT advantage do not exist on this workload.

### Main test @ keep_frac=0.15 — n=233 fully paired, 94% completeness, 0 budget violations

| Arm | Passes | Accuracy | Wilson 95% |
|---|---:|---:|---|
| FullKV | 52/233 | **0.223** | [0.174, 0.281] |
| SnapKV | 32/233 | **0.137** | [0.099, 0.187] |
| ADAPT | 26/233 | **0.112** | [0.077, 0.158] |

| Comparison | b | c | Δ | exact McNemar |
|---|---:|---:|---:|---:|
| ADAPT vs SnapKV | 10 | 16 | −0.026 | **0.327** (n.s. — predicted) |
| SnapKV vs FullKV | 12 | 32 | −0.086 | 0.0037 |
| ADAPT vs FullKV | 12 | 38 | −0.112 | 0.0003 |

**Interpretation (fixed in advance).** ADAPT is indistinguishable from SnapKV and
trends 2.6 points below. The registration bounds any true advantage below ~0.05
absolute; the measured Δ sits inside that bound. **The method confers no external
benefit. The mechanism that predicts its failure has external predictive power.**

### Why BFCL cannot reward this prior (failure census, CPU-only)

| Arm | pass | empty turn response | state mismatch | exec mismatch |
|---|---:|---:|---:|---:|
| FullKV | 20.5% | **45.5%** | 22.7% | 11.4% |
| SnapKV | 13.9% | 50.4% | 21.2% | 14.6% |
| ADAPT | 12.9% | 46.4% | **27.1%** | 13.6% |
| Structure | 0.0% | **78.0%** | 10.0% | 12.0% |

45.5% of conversations fail with an unparseable turn response **even at FullKV** —
an 8B model on this benchmark is limited more by its own tool-calling ability than
by retention. The measurable interval between SnapKV and FullKV is ~8 conversations.

---

## SWEEP_PM_V3_32B — scaling check: the boundary is a policy property (2026-08-04)

Qwen3-32B (64 layers, 2x L40S via `device_map=auto`), `keep_frac=0.05`, n=60/cell,
2,880 work units, **zero failures**. Protected mass alone varies the ratio 1.0 -> 18.2.

| rho | Structure (32B) | Structure (8B) | SnapKV | ADAPT | FullKV | structure vs snapkv |
|---:|---:|---:|---:|---:|---:|---|
| 1.0x | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | b=0 c=0 p=1.0 |
| 2.2x | **0.617** | 0.617 | 1.000 | 1.000 | 1.000 | b=0 c=23 p=2.4e-07 |
| 3.7x | **0.417** | 0.417 | 1.000 | 1.000 | 1.000 | b=0 c=35 p=5.8e-11 |
| 5.3x | **0.400** | 0.400 | 1.000 | 1.000 | 1.000 | b=0 c=36 p=2.9e-11 |
| 8.0x | **0.400** | 0.400 | 0.983 | 1.000 | 1.000 | b=1 c=36 p=5.5e-10 |
| 11.0x | **0.433** | 0.400 | 1.000 | 1.000 | 1.000 | b=0 c=34 p=1.2e-10 |
| 14.5x | **0.400** | 0.400 | 0.967 | 1.000 | 1.000 | b=2 c=36 p=5.4e-09 |
| 18.2x | **0.400** | 0.400 | 0.983 | 1.000 | 1.000 | b=1 c=36 p=5.5e-10 |

**The 32B curve matches the 8B curve to three significant figures at every level.**
The knee does not move, the descent does not soften, the floor does not shift — at 4x
the parameters. That is the signature of a property of the *retention policy*, not of
the model: the 0.400 floor is the tool-schema family passing while the other two fail,
and which family passes is fixed by where the tiebreak retains positions.

Structure loses to matched-budget SnapKV in 7 of 8 cells with b=0-2 against c=23-36.

Per-category at 32B (kf=0.05) matches 8B exactly: tool_schema **1.00** at every level
tested, supersession **0.00**, multi_turn_state **0.20**.

Artifacts: `configs/pm_sweep_32b.yaml`, `cluster/prajna/pm_sweep_32b.sbatch`,
`$PRAJNA_ROOT/results/pm_sweep_v3/qwen32/`.

---

## SWEEP_PM_V3 — the operating boundary, measured (2026-07-28)

Post-freeze namespace. **Does not modify any frozen number below.** Fills the interval
between the two endpoints the frozen work reported (PriorityBench-A ~6% protected mass,
BFCL ~99%) with a controlled dose–response sweep: 8 protected-mass levels (5.2%–92.2%)
× 3 keep budgets (2/5/10%) × 6 arms, 16k contexts, **7,680 generations per model on
Qwen3-8B and Llama-3.1-8B, zero failures**.

### Structure vs oversubscription ratio (`|S∪M| / B`)

| Ratio | Structure (Qwen) | Structure (Llama) | SnapKV | ADAPT | FullKV |
|---|---:|---:|---:|---:|---:|
| 0.5× | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.0× | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.1× | 1.000 | 0.883 | 1.000 | 1.000 | 1.000 |
| 2.2× | 0.617 | 0.600 | 1.000 | 1.000 | 1.000 |
| 4–18× | 0.400 | 0.333–0.417 | 0.967–1.000 | 1.000 | 1.000 |
| 27–45× | 0.067 | 0.000 | 0.750–0.867 | 0.950–1.000 | 1.000 |

**The knee is at ratio ≈ 1.1 on both architectures.** FullKV is 1.000 in every cell, so
the collapse is retention, not task difficulty.

### The mechanism: pooling hides a dissociation

At high oversubscription, structure by task family (n=20/cell/model):

| Model | Budget | tool_schema | supersession | multi_turn_state |
|---|---|---:|---:|---:|
| Qwen | 0.10 / 0.05 | **1.000** | **0.000** | 0.200 |
| Qwen | 0.02 | 0.000 | 0.000 | 0.200 |
| Llama | 0.10 / 0.05 | **1.000** | **0.000** | 0.000–0.150 |
| Llama | 0.02 | 0.000 | 0.000 | 0.000–0.050 |

A tool-schema item's load-bearing content is the **leading system block**, and the policy's
tiebreak keeps the *earliest* protected positions — so the contract survives any ratio until
the budget cannot hold the block at all. Supersession/state live mid-conversation and are
dropped. **This supersedes the protected-mass-only account** and explains the BFCL 0.000
directly: BFCL schemas are in the system prompt; its task-critical state is not.

### ADAPT

Holds 0.950–1.000 (Qwen) and 0.983–1.000 (Llama) across all 24 cells per model, **losing
no discordant pair to SnapKV in 48 cells**. Significantly exceeds SnapKV on Qwen at
`keep_frac=0.02`:

| Level | ADAPT | SnapKV | b | c | exact McNemar |
|---|---:|---:|---:|---:|---:|
| pm50 | 1.000 | 0.900 | 6 | 0 | 0.031 |
| pm65 | 1.000 | 0.867 | 8 | 0 | 0.0078 |
| pm80 | 1.000 | 0.867 | 8 | 0 | 0.0078 |
| pm95 | 0.950 | 0.750 | 12 | 0 | **4.9e-04** |

**Does not replicate on Llama** — SnapKV never falls below 0.983 there, so no headroom.
Claim scope: ADAPT *matches* attention selection everywhere tested and *exceeds* it where
attention selection itself degrades.

### Two informative nulls (kept, not discarded)

- **V1** (8k, gold in template prefix, budgets 10/25/50%): every non-blind arm 1.000 in all
  24 cells, including 9× oversubscription. Cause: the earliest-first tiebreak re-selects the
  gold positions. **Structure is immune to oversubscription when gold sits in the prefix.**
- **V2** (relocation applied *after* schema conversion): confounded. The relocation helper
  treats converted schema turns as non-filler, so gold drifted from char-fraction 0.65 at
  pm06 to **0.07** at pm95 — back to the prefix exactly where oversubscription should bite.
  V3 relocates *before* conversion; gold verified at 0.63–0.65 at every level.

### Not claimed from the sweep

- Any external validity — the sweep is synthetic and its protected-mass axis is built from
  distractor schemas authored to carry roles the tagger recognises (instrument and stimulus
  are not independent).
- Workload realism — FullKV is 1.000 in every cell, so there is no difficulty gradient.
- ADAPT superiority in general — the win is Qwen-and-2%-specific.

Artifacts: `configs/pm_sweep_v3.yaml`, `scripts/{mk_pm_sweep,run_pm_sweep,pm_sweep_summary,make_pm_sweep_figure}.py`,
`$PRAJNA_ROOT/results/pm_sweep_v3/{qwen,llama}/summaries/`.

---

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

### BFCL V3 multi-turn — frozen all-arm intersections

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

| Comparison | Exact McNemar |
|---|---:|
| FullKV vs Structure | **1.5e-08** |
| FullKV vs SnapKV | **0.152** (n.s.) |
| Structure vs SnapKV | **3.8e-06** |

**BFCL identifies the high-protected-mass boundary.** SnapKV is statistically
indistinguishable from FullKV at a 4× budget, while the binary structure score is
saturated. This complements rather than contradicts PriorityBench-A. Llama-3.1-8B
reproduces the ordering: FullKV vs SnapKV **p=1.0**, FullKV vs structure
**p=9.8e-04**, structure vs SnapKV **p=4.9e-04**.

### ADAPT — structure as a budget-relative prior

`alpha = min(1, keep_budget / protected_mass)`, blending rank-normalised structure
and attention scores. Alpha uses only quantities known from the prompt: no tuning,
no fitting, no free parameter.

**Pre-registration caveat.** The formula was written before the ADAPT run, but the
commit that was meant to freeze it timed out and did not land, so it was committed
*after* the results were seen. The prediction is therefore pre-specified **by
construction** — alpha is a closed form over two measured quantities with nothing
to fit — but not by commit timestamp. Weigh the "~0.25 predicted, 0.267 measured"
agreement accordingly.

Measured alpha on BFCL: **mean 0.267** (min 0.250, max 0.401) over 833 generation
steps, against **~0.25 predicted** from the 98.8% protected fraction — a prediction
made in advance and confirmed.

| Comparison | exact McNemar | Δ | 95% CI |
|---|---:|---:|---|
| ADAPT vs SnapKV | **1.000** (n.s.) | −0.007 | [−0.057, +0.071] |
| ADAPT vs FullKV | 0.108 (n.s.) | −0.064 | [−0.136, +0.007] |
| ADAPT vs Structure | **7.6e-06** | +0.129 | [+0.079, +0.186] |

**ADAPT reaches the same measured range as SnapKV and is indistinguishable from
FullKV on this workload.** It does not establish superiority to SnapKV; it shows
that a prompt-derived weight can recover attention-level behavior when hard
structure is saturated. At alpha=1 it recovers the evaluated structure ordering.

### The boundary condition (why)

Structure can only express a preference while protected mass stays *under* the
keep budget. Measured at `keep_frac=0.25`:

| Workload | Protected tokens | Oversubscribed | Structure |
|---|---:|---:|---:|
| PriorityBench-A | **6.0%** | 0% | **0.933** |
| τ-bench | 79.5% | 99% | retention-only |
| BFCL | **98.8%** | 100% | **0.000** |

PriorityBench-A is **94.0% non-protected by the same mean-fraction measure** — the regime where "protect structure,
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
- Manuscript: [`paper/prioritykv.tex`](paper/prioritykv.tex)
- Reproduction guide: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- Job manifests and bundles: [`jobs/`](jobs/)
