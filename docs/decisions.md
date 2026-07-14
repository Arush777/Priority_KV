# Decisions log (solo)

Append-only. Newest at bottom.

## 2026-07-14 — Dual-machine ops

- **Decided:** Solo ownership (no A/B split). All workstreams owned by Arush.
- **Decided:** Cursor agents develop on CCC/login checkout and push to `Arush777/Priority_KV`.
- **Decided:** H200 is human-operated only: `git pull` + `uv` + GPU runs. No agents on H200.
- **Decided:** Env manager is `uv` with `pyproject.toml` + lockfile; GPU extras via `uv sync --extra gpu`.
- **Decided:** Primary model pin remains `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`.

## 2026-07-14 — Shared H200 etiquette

- **Decided:** Hard cap of **2 GPUs** on the shared 8× H200 host. Default `CUDA_VISIBLE_DEVICES=6,7` (override only if busy).
- **Decided:** Operator-facing scripts are bland (`scripts/sync.sh`, `scripts/check.sh`); no project/model slogans in shell banners.

## 2026-07-14 — Git identity + W1 start

- **Decided:** All commits authored/committed as `Arush777 <153831754+Arush777@users.noreply.github.com>` (never CCC/IBM host identity).
- **Decided:** W1 FullKV compare CLI is `scripts/cmp_gen.py`; results under `$PRIORITYKV_SCRATCH/runs/`.

## 2026-07-14 — W1 byte model freeze (pre)

- **Measured (Qwen3-8B GQA 36L×8H×128d):** all-INT4 realized floor ≈ **29.7%** of FullKV BF16 (payload+scales+zp+page table).
- **Implication:** budget **25% is infeasible** without eviction (matches plan). Budget **30%** leaves almost no BF16 headroom (~0.4% of tokens ≈ 144 toks @ 32K); protected pages must be tiny or we treat 50% as the primary quality operating point and 30% as a stress budget.
- **Geom pin:** `QWEN3_8B_KV = (36, 8, 128)` in `src/prioritykv/byte_model.py`; table in `configs/w1_byte_budget.json`.
- **W1 PriorityBench pilot:** 40 `tool_schema` examples, 8 templates, seeds in `data/prioritybench/manifests/w1_pilot.json`.

## 2026-07-14 — W1 FP8 baseline smoke

- **Decided:** W1 S1 smoke uses vLLM `kv_cache_dtype=fp8` + `calculate_kv_scales=True` (on-the-fly). Dataset oneshot via `scripts/prep_fp8.py` (llmcompressor) is optional follow-up, not blocking G0.
- **CLI:** `scripts/cmp_fp8.py`; results under `$PRIORITYKV_SCRATCH/runs/w1_fp8/`.
