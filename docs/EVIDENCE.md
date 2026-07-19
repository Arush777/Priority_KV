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
| s1 | queued `p0c_…_middle` | queued `p0d_…_buried` |
| s2 | queued `p0e_…_middle` | queued `p0f_…_buried` |

**No further claim-blocking work** beyond landing s1/s2 controls. Paper tex is intentionally untouched (stale vs evidence; packaging debt acknowledged).

## Honest claim (paste this)

> On PriorityBench-A (synthetic agent traces), structure-aware retention at a 25% keep budget far exceeds position-blind eviction on Qwen (P0 n=120: structure **0.933** vs uniform/random **~0.008**). Mid-context relocation (s0) ties FullKV (both **0.975**); burying state hurts structure (**0.675** vs FullKV **0.900**) while uniform/random stay ~0 — so we do **not** claim structure beats FullKV. A CPU gold-span audit shows gold is **not** concentrated in sink+recent on Qwen or Llama (≈0–1% of gold tokens); structure keep retains essentially all gold tokens while uniform retains ≈0–1%, so the Qwen blind-eviction gap is retention-real, not a labeling leak into the always-kept window. Versus SnapKV-class selection on Qwen (n=120): structure **0.933** vs SnapKV/Pyramid/hybrid **0.900** (112/120 vs 108/120; McNemar **p=0.125**, not significant) — we claim only that it **matches or slightly exceeds** SnapKV-class methods while decisively beating position-only baselines. Hybrid did **not** beat SnapKV. Llama-3.1 at kf=0.25 is saturated among structure+attention arms (all **1.0**); the gold audit rules out “gold already in sink/recent” as the explanation — the task is too easy once any competent keep runs. At kf=0.05 SnapKV outperforms structure on **two** slices (s0: 1.0 vs 0.875; s1: 1.0 vs 0.900). P2 streamed-cold is a **smoke test** (~36 GiB peak in log), not a systems result.

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
| **P0 mid/buried s1/s2** | `p0{c,d,e,f}_…` | extending s0 controls (queued) |

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

## Not claimed

- Structure beats FullKV or SnapKV in general  
- Significant structure≫SnapKV on Qwen  
- Universal cross-model transfer to Llama  
- Soft INT4 quality win; peak VRAM collapse; LongBench/RULER matrices  

## Pointers

- [`../RESULTS.md`](../RESULTS.md) · [`../README.md`](../README.md) · [`../FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml)  
- McNemar: [`../jobs/results/p1_structure_vs_snapkv_mcnemar.json`](../jobs/results/p1_structure_vs_snapkv_mcnemar.json)  
- Retention audits: `jobs/results/audit_retention_{qwen,llama}_s{0,1,2}_kf25_summary.json`
