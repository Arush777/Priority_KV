# Evidence track & external-audit response

Status for collaborators, agents, and external reviewers. Numbers from
`jobs/results/*/summary.json` on `main` unless noted. Hardware: NVIDIA H200
(`dgre2`). Primary model: Qwen3-8B (`b968826d…`). Transfer: Llama-3.1-8B-Instruct
(`0e9e39f…`).

## Audit checklist (what was raised → status)

| # | Issue | Status | Evidence |
|---|---|---|---|
| 1 | structure > FullKV may be leakage / “less is more” | **Partially addressed (GPU)** | Middle control: structure=full=**0.975** (Δ**0**). Buried: structure **0.675** < full **0.900**. Do **not** lead with structure>FullKV. |
| 2 | “beats SnapKV” overclaims 4/120 | **Addressed (stats + wording)** | 112 vs 108 passes; McNemar exact two-sided **p=0.125** (`jobs/results/p1_structure_vs_snapkv_mcnemar.json`). Not significant. |
| 3 | Llama kf25 “ceiling” may be port artifact | **Open (CPU)** | Gold-in-kept-region audit not run yet. Prefer: “non-discriminative / saturated probe” until audit. |
| 4 | kf05 SnapKV>structure one slice | **Addressed (GPU)** | s1 replicate: SnapKV **1.0** > structure **0.900** (`p3_llama31_attn_s1_kf05_gpu1_r1`). |
| 5 | H2O number mismatch | **Addressed (docs)** | Canonical **0.683** = (0.725 chunked s0 + 0.625 s1 + 0.700 s2) / 3 ≈ **82/120**. Never cite obsolete s0 0.600. Chunked reimplementation caveat remains. |
| 6 | P2 is smoke | **Addressed (docs)** | exit=0 only; peak in log ~**36.4 GiB** (not 59). Frozen D4/MG peak-latency numbers unchanged. |
| 7 | Hybrid ≤ SnapKV kills complementarity | **Addressed (docs)** | hybrid=SnapKV=**0.900**; no complementarity claim. |

**Still open (CPU, not claim-blocking for the softened text below):** retention preflight script; Llama gold-span vs sink/recent audit. Council: **STOP further GPU** until those are done if you want a harder Llama claim.

## Honest claim (paste this)

> On PriorityBench-A (synthetic agent traces), structure-aware retention at a 25% keep budget far exceeds position-blind eviction on Qwen (P0 n=120: structure **0.933** vs uniform/random **~0.008**). Mid-context relocation (s0) ties FullKV (both **0.975**); burying state hurts structure (**0.675** vs FullKV **0.900**) while uniform/random stay ~0 — so we do **not** claim structure beats FullKV. Versus SnapKV-class selection on Qwen (n=120): structure **0.933** vs SnapKV/Pyramid/hybrid **0.900** (112/120 vs 108/120; McNemar **p=0.125**, not significant) — we claim only that it **matches or slightly exceeds** SnapKV-class methods while decisively beating position-only baselines. Hybrid did **not** beat SnapKV. Llama-3.1 at kf=0.25 is saturated (all arms 1.0); at kf=0.05 SnapKV outperforms structure on **two** slices (s0: 1.0 vs 0.875; s1: 1.0 vs 0.900). P2 streamed-cold is a **smoke test** (~36 GiB peak in log), not a systems result.

## What we ran

| Phase | Jobs | Outcome |
|---|---|---|
| **P0** plain | `p0_w5_s{0,1,2}_kf25_token_*` | structure **0.933** vs uniform/random **~0.008** (n=120) |
| **P0 middle** | `p0a_w5_s0_kf25_token_middle_gpu1_r1` | structure=full=**0.975**; uniform/random **0.025** |
| **P0 buried** | `p0b_w5_s0_kf25_token_buried_gpu1_r1` | structure **0.675** < full **0.900**; uniform/random **0** |
| **P1** | `p1_attn_baselines_s{0,1,2}_*` + `p1_h2o_chunked_s0_*` | structure 0.933 / SnapKV·Pyr·hyb 0.900 / H2O **0.683**; McNemar p=0.125 |
| **P2** | `p2_fi_stream_cold_16k_gpu1_r1` | smoke exit=0; peak_gib≈36.4 in log |
| **P3** kf25 | `p3_llama31_attn_s{0,1,2}_kf25_*` | all arms **1.000** (n=120) |
| **P3** kf05 | `p3_llama31_attn_s0_kf05_*`, `…_s1_kf05_*` | SnapKV > structure on both slices |

## Strength table (revised)

| Pillar | Strength | Why |
|---|---|---|
| Structure ≫ uniform/random | **Strong** | P0 n=120; survives middle & buried as gap vs blind |
| Structure > FullKV | **Not claimed** | Mid: Δ0; buried: structure loses to FullKV |
| Structure vs SnapKV (Qwen) | **Weak–moderate** | Directional 3/3 slices; **not** significant (p=0.125) |
| Hybrid complementarity | **Falsified** | hybrid = SnapKV |
| Llama transfer @ kf25 | **Non-discriminative** until gold audit | all arms ceiling |
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
