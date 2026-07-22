# Evidence track & external-audit response

Status for collaborators, agents, and external reviewers. Numbers from
`jobs/results/*/summary.json` on `main` unless noted. Hardware: NVIDIA H200
(`dgre2`). Primary model: Qwen3-8B (`b968826d…`). Transfer: Llama-3.1-8B-Instruct
(`0e9e39f…`).

## Audit checklist (what was raised → status)

| # | Issue | Status | Evidence |
|---|---|---|---|
| 1 | structure > FullKV may be leakage / “less is more” | **Partially addressed (GPU)** | Middle: structure=full=**0.975** (Δ**0**). Buried: structure **0.675** < full **0.900**. Do **not** lead with structure>FullKV. |
| 2 | “beats SnapKV” overclaims 4/120 | **Addressed (stats + wording)** | 112 vs 108; McNemar exact two-sided **p=0.125** (`jobs/results/p1_structure_vs_snapkv_mcnemar.json`). Not significant. |
| 3 | Llama kf25 “ceiling” may be port artifact (gold in sink/recent) | **Falsified as port artifact (CPU)** | `scripts/audit_retention.py` on Llama s0 kf25: mean `gold_in_sink_recent_frac`=**0.0**; uniform gold kept **0.0**; structure gold kept **1.0**. Verdict: `GOLD_MOSTLY_EVICTABLE`. Same pattern on Qwen (sink+recent ≈**0.01**). P3 all-1.0 among structure+attn arms is an **easy-task ceiling**, not gold already in the always-kept window. (P3 did not run a uniform arm.) |
| 4 | kf05 SnapKV>structure one slice | **Addressed (GPU)** | s1 replicate: SnapKV **1.0** > structure **0.900**. |
| 5 | H2O number mismatch | **Addressed (docs)** | Canonical **0.683** = (0.725+0.625+0.700)/3. Never cite obsolete s0 0.600. |
| 6 | P2 is smoke | **Addressed (docs)** | exit=0; peak ≈**36.4 GiB** in log (not 59). |
| 7 | Hybrid ≤ SnapKV kills complementarity | **Addressed (docs)** | hybrid=SnapKV=**0.900** at Qwen kf25. At Llama kf05, hybrid collapses below both parents (s0 **0.575**, s1 **0.525** vs SnapKV/structure ≥0.875) — unclaimed; noted as a negative complementarity datapoint. |

### Retention preflight (CPU)

| Audit | n | mean gold in sink+recent | structure gold kept | uniform gold kept | Verdict |
|---|---|---|---|---|---|
| Qwen s0 kf25 | 40 | **0.010** | **1.000** | **0.010** | `GOLD_MOSTLY_EVICTABLE` |
| Qwen s1 kf25 | 40 | **0.013** | **1.000** | **0.013** | `GOLD_MOSTLY_EVICTABLE` |
| Qwen s2 kf25 | 40 | **0.010** | **1.000** | **0.010** | `GOLD_MOSTLY_EVICTABLE` |
| Llama s0 kf25 | 40 | **0.000** | **1.000** | **0.000** | `GOLD_MOSTLY_EVICTABLE` |
| Llama s1 kf25 | 40 | **0.000** | **1.000** | **0.000** | `GOLD_MOSTLY_EVICTABLE` |
| Llama s2 kf25 | 40 | **0.000** | **1.000** | **0.000** | `GOLD_MOSTLY_EVICTABLE` |

Artifacts: `jobs/results/audit_retention_{qwen,llama}_s{0,1,2}_kf25_summary.json`,
`scripts/audit_retention.py`.

### Placement controls (GPU) — mid / buried

| Slice | Middle (structure / full / uniform) | Buried (structure / full / uniform) |
|---|---|---|
| s0 | **0.975 / 0.975 / 0.025** (`p0a_…`) | **0.675 / 0.900 / 0.000** (`p0b_…`) |
| s1 | **0.950 / 0.950 / 0.050** (`p0c_…`) | **0.650 / 0.875 / 0.000** (`p0d_…`) |
| s2 | **0.950 / 0.950 / 0.000** (`p0e_…`) | **0.675 / 0.900 / 0.000** (`p0f_…`) |

Pattern holds across **3/3** slices: middle ties FullKV; buried structure loses to FullKV; uniform ~0. Paper tex is intentionally untouched (stale vs evidence; packaging debt acknowledged).

## Honest claim (paste this)

> On PriorityBench-A (synthetic agent traces), structure-aware retention at a 25% keep budget far exceeds position-blind eviction on Qwen (P0 n=120: structure **0.933** vs uniform/random **~0.008**). Mid-context relocation ties FullKV on s0/s1/s2 (e.g. s0 both **0.975**); burying state hurts structure on all three slices (s0 **0.675** vs FullKV **0.900**; s1 **0.650** vs **0.875**; s2 **0.675** vs **0.900**) while uniform/random stay ~0 — so we do **not** claim structure beats FullKV. A CPU gold-span audit shows gold is **not** concentrated in sink+recent on Qwen or Llama (≈0–1% of gold tokens); structure keep retains essentially all gold tokens while uniform retains ≈0–1%, so the Qwen blind-eviction gap is retention-real, not a labeling leak into the always-kept window. Versus SnapKV-class selection on Qwen (n=120): structure **0.933** vs SnapKV/Pyramid/hybrid **0.900** (112/120 vs 108/120; McNemar **p=0.125**, not significant) — we claim only that it **matches or slightly exceeds** SnapKV-class methods while decisively beating position-only baselines. Hybrid did **not** beat SnapKV. Llama-3.1 at kf=0.25 is saturated among structure+attention arms (all **1.0**); the gold audit rules out “gold already in sink/recent” as the explanation — the task is too easy once any competent keep runs. At kf=0.05 SnapKV outperforms structure on **two** slices (s0: 1.0 vs 0.875; s1: 1.0 vs 0.900). P2 streamed-cold is a **smoke test** (~36 GiB peak in log), not a systems result.

## What we ran

| Phase | Jobs | Outcome |
|---|---|---|
| **P0** plain | `p0_w5_s{0,1,2}_kf25_token_*` | structure **0.933** vs uniform/random **~0.008** (n=120) |
| **P0 middle** | `p0a_w5_s0_kf25_token_middle_gpu1_r1` | structure=full=**0.975**; uniform/random **0.025** |
| **P0 buried** | `p0b_w5_s0_kf25_token_buried_gpu1_r1` | structure **0.675** < full **0.900**; uniform/random **0** |
| **P1** | `p1_attn_baselines_s{0,1,2}_*` + `p1_h2o_chunked_s0_*` | structure 0.933 / SnapKV·Pyr·hyb 0.900 / H2O **0.683**; McNemar p=0.125 |
| **P2** | `p2_fi_stream_cold_16k_gpu1_r1` | smoke exit=0; peak_gib≈36.4; `summary.json` reconstructed from log |
| **P3** kf25 | `p3_llama31_attn_s{0,1,2}_kf25_*` | structure+attn arms **1.000** (n=120) |
| **P3** kf05 | `p3_llama31_attn_s0_kf05_*`, `…_s1_kf05_*` | SnapKV > structure; hybrid **0.575 / 0.525** (worse than both parents) |
| **Retention audit** | `audit_retention_{qwen,llama}_s{0,1,2}_kf25` | gold outside sink+recent; structure keeps gold, uniform does not |
| **P0 mid/buried s1/s2** | `p0{c,d,e,f}_…` | mid ties FullKV; buried structure < FullKV (3/3) |

## Strength table (revised)

| Pillar | Strength | Why |
|---|---|---|
| Structure ≫ uniform/random | **Strong** | P0 n=120; gold audit shows gold is evictable and structure keeps it |
| Structure > FullKV | **Not claimed** | Mid: Δ0; buried: structure loses to FullKV |
| Structure vs SnapKV (Qwen) | **Weak–moderate** | Directional 3/3; **not** significant (p=0.125) |
| Hybrid complementarity | **Falsified** | hybrid = SnapKV |
| Llama @ kf25 | **Easy-task ceiling** (not port artifact) | all structure+attn arms 1.0; gold not in sink+recent |
| Llama @ kf05 | **Honest negative** | SnapKV > structure on s0+s1 |
| P2 systems | **Smoke only** | no FullKV/TPOT frontier |

## External evaluation — `EXTERNAL_BFCL_PRAJNA_V1`

Separate freeze; nothing below modifies a frozen claim. Full detail in
[`../RESULTS.md`](../RESULTS.md).

| # | Claim under test | Verdict | Evidence |
|---|---|---|---|
| E1 | Hard structural retention transfers without checking protected mass | **Boundary identified** | On the frozen all-arm intersection of BFCL V3 multi-turn ($n=140$), structure records **0.000** vs FullKV **0.193** (exact McNemar **p=1.5e-08**) after protected mass exceeds the budget. Uniform and corrected-random also record 0.000. |
| E2 | Hard structure matches SnapKV externally | **Not supported in the high-mass regime** | SnapKV records **0.136** vs structure **0.000**, **p=3.8e-06** on the same $n=140$ intersection. |
| E3 | Attention-based eviction preserves agent capability at 4× | **Supported within measured uncertainty** | FullKV 0.193 vs SnapKV 0.136, **p=0.152 (n.s.)**; the paired interval spans zero. |
| E4 | Structure's advantage is bounded by protected fraction | **Supported** | PriorityBench-A has 6.0% mean protected mass and scores 0.933; BFCL has 98.8% protected mass and the hard structure arm scores 0.000. BFCL is 100% oversubscribed at kf=0.25. |
| E5 | Structure preserves durable constraints better than recency | **Supported (retention only)** | τ-bench, 828k spans: explicit-policy any-retained **0.820** vs uniform **0.001**; loses on reused identifiers (0.055 vs 0.315). Not task success. |
| E7 | The boundary is specific to Qwen | **Not supported** | Llama-3.1-8B, $n=143$: FullKV 0.077 vs SnapKV 0.084 (**p=1.0**), structure 0.000 (**p=9.8e-04** vs FullKV). The same ordering appears on a second architecture. |
| E8 | ADAPT beats SnapKV | **Not established** | ADAPT 0.129 vs SnapKV 0.136, **p=1.0**, Δ −0.007 CI [−0.057, +0.071]. ADAPT reaches the same measured range as SnapKV and is indistinguishable from FullKV ($p=0.108$), while exceeding structure/uniform/random (**p=7.6e-06**). |
| E9 | ADAPT's alpha is fitted to task outcomes | **No** | The formula was selected before the recorded ADAPT run and committed afterwards. Protected mass predicts approximately 0.25; measured mean $\alpha$ is **0.267** over 833 steps. |
| E6 | The frozen `random` arm is an independent control | **No; duplicate control** | `select_random` is byte-identical to `select_uniform` at every tested context length because the RNG branch never executes. The two 1/120 columns are reported once. |

### Negative results and corrections

- **`random` ≡ `uniform` in the frozen core** (E6). Affects `RESULTS.md` P0. Frozen
  code deliberately untouched; corrected only in the external namespace.
- **Thinking-disabled Qwen3 scores 0.000 on BFCL** for *every* arm including
  FullKV, so the benchmark cannot discriminate. Thinking is enabled externally and
  the reasoning block is stripped before decoding, as the official handler does.
- **`hybrid_press` failed for a diagnosable reason.** Its hard union force-protects
  structural positions, which swallows the entire budget whenever protected mass
  exceeds it — consistent with the recorded Llama kf05 collapse below both parents.
- **FullKV capability ceiling is real**: 16 conversations exceeded Qwen3-8B's
  40,960-token context and were excluded with `MODEL_CONTEXT_LIMIT`, concentrated on
  the arms that actually make working tool calls (their conversations grow).

### Scope limits

- Qwen3-8B only; L40S/sm_89 (Prajna has no H100 — see `DEV_NO_H100`).
- `keep_frac=0.25` only; no budget sweep yet.
- BFCL V3 has four categories, not five — there is no `composite` split in V3.
- The τ audit excludes SnapKV: it needs realised attention and cannot run
  generation-free on CPU.

## Not claimed

- Structure beats FullKV or SnapKV in general  
- Significant structure≫SnapKV on Qwen  
- Universal cross-model transfer to Llama  
- Soft INT4 quality win; peak VRAM collapse; LongBench/RULER matrices  

## Pointers

- [`../RESULTS.md`](../RESULTS.md) · [`../README.md`](../README.md) · [`../FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml)  
- McNemar: [`../jobs/results/p1_structure_vs_snapkv_mcnemar.json`](../jobs/results/p1_structure_vs_snapkv_mcnemar.json)  
- Retention audits: `jobs/results/audit_retention_{qwen,llama}_s{0,1,2}_kf25_summary.json`
