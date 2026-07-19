# Evidence track (P0–P3)

Status for collaborators and agents. All numbers are from `jobs/results/*/summary.json`
on `main` unless noted. Hardware: NVIDIA H200 (`dgre2`). Primary model: Qwen3-8B
(`b968826d…`). Transfer model: Llama-3.1-8B-Instruct (`0e9e39f…`).

## Claim (scoped)

**Strong on Qwen:** at matched keep budgets, structure-aware KV retention beats
role-blind keep (uniform/random) and beats common attention eviction baselines
(SnapKV / PyramidKV / hybrid); H2O is weaker still.

**Honest on Llama:** the same P1 protocol at `keep_frac=0.25` is **ceiling-saturated**
(all arms 1.0). That is a completed transfer *run*, not a positive transfer *win*.
At `keep_frac=0.05` (slice 0), SnapKV **beats** structure (1.0 vs 0.875).

## What we ran

| Phase | Question | Jobs (canonical) | Outcome |
|---|---|---|---|
| **P0** | Structure vs uniform/random @ kf=0.25, n=120 | `p0_w5_s{0,1,2}_kf25_token_*` | Structure **0.933** vs uniform/random **~0.008** |
| **P1** | Structure vs SnapKV/H2O/Pyramid/hybrid @ kf=0.25, n=120 (Qwen) | `p1_attn_baselines_s{0,1,2}_kf25_*` + `p1_h2o_chunked_s0_*` | Structure **0.933** > SnapKV/Pyramid/hybrid **0.900** > H2O **~0.68** |
| **P2** | Streamed cold attend (no full BF16 cold scratch) | `p2_fi_stream_cold_16k_gpu1_r1` | exit=0; FI path runnable (~59 GiB peak) |
| **P3** | Same P1 protocol on Llama, n=120 @ kf=0.25 | `p3_llama31_attn_s{0,1,2}_kf25_gpu1_r1` | All arms **1.000** (saturated / `SNAPKV_MATCHES`) |

Budget probes on Llama s0 (not the n=120 claim): kf10 still ceiling for structure/SnapKV;
kf05 separates against structure (`p3_llama31_attn_s0_kf05_gpu1_r1`).

## How strong is the idea?

| Pillar | Strength | Why |
|---|---|---|
| Structure ≫ uniform/random | **Strong** | P0 n=120, three slices, near-zero role-blind scores |
| Structure > SnapKV (Qwen) | **Moderate–strong** | Consistent on all three P1 slices; gap is real but not huge (~3 pp) |
| Attention baselines ranking | **Useful** | SnapKV ≈ Pyramid ≈ hybrid; H2O clearly worse (needs chunked impl) |
| Systems / FI stream | **Supporting** | P2 proves the cold path; not the main science claim |
| Cross-model transfer (Llama) | **Weak / negative at kf25** | Task too easy at 25% keep; tighter budget flips against structure |

**One-line for Claude:** Lead with Qwen P0+P1 (structure beats uniform and SnapKV at matched budgets); report Llama P3 as an honest ceiling/negative transfer at kf25, with kf05 showing SnapKV≥structure — revise any universal-transfer wording.

## Not claimed

- Structure always beats SnapKV on every model/budget
- Soft INT4 quality win (already falsified on lock-240)
- Peak VRAM collapse under FI cold scratch
- LongBench / RULER matrices

## Pointers

- Metrics narrative: [`../RESULTS.md`](../RESULTS.md)
- Frozen science core: [`../FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml)
- H200 queue: [`H200_QUEUE.md`](H200_QUEUE.md)
