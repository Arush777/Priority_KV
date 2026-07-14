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

## 2026-07-14 — W1 closed / W2 start

- **W1 FP8 smoke:** `cmp_fp8.py` → exact=0.850 tok=0.926 pass=1 (S1 runnable on H200). Defer `prep_fp8.py`/llmcompressor to S1 freeze.
- **W2 start:** page manager + structural tagging + protected invariants (CPU reference). Grow PriorityBench with instruction_supersession templates.

## 2026-07-14 — W2 H200 confirm

- **Page smoke (H200):** `check_pages.py` → seq_len=25282, pages=1581, bf16=274 / int4=25008, within_budget=true, invariants_ok=true.
- **W2 pilot:** `mk_bench.py --mode w2` → n=120 (cal 56 / val 27 / test 37); manifest `data/prioritybench/manifests/w2_pilot.json`.

## 2026-07-14 — W2 quality pilot harness

- **Decided:** First agent-reliability compare is `scripts/run_pilot.py` (15× cal/8k: 10 tool + 5 supersession; FullKV vs FP8; deterministic scorers).

## 2026-07-14 — W2 pilot result + supersession fix

- **8k pilot (r1):** full=0.800 fp8=0.800 delta=0; tool_schema 1.0/1.0; supersession 0.40/0.40. Failures were format_flip scorer (looked for format *names* in prose), not FP8.
- **Fix:** format_flip now requires explicit `[[FMT:...]]` tags (rev 2). Re-run 8k then 16k pilots.
- **SnapKV:** scaffold only (`scripts/snap_status.py`); not runnable yet.

## 2026-07-14 — 8k pilot rev2 + 16k OOM-length fix

- **8k r2:** full=1.000 fp8=1.000 delta=0; both categories 1.0 — tag fix worked; FP8 still no harm at 8k.
- **16k r1 failed:** chat-templated prompts exceeded max_model_len 20480. Bumped to 32768 + defensive trim.

## 2026-07-14 — 16k pilot r2

- **16k r2:** full=0.933 fp8=0.933 delta=0; tool_schema 1.0/1.0; supersession 0.80/0.80.
- **Read:** longer context slightly hurts supersession on FullKV already; FP8 still tracks FullKV (no extra agent damage yet). INT4 / 32k / harder templates are the next stress.

## 2026-07-14 — language_flip case fix

- **16k miss:** `...language_flip...s20271049` scored 0 because output had `Bravo` vs required `bravo` (case-sensitive). Flags now include `IGNORECASE`.

## 2026-07-14 — choose multi_turn next

- **Decided:** Highest-signal next step is `multi_turn_state` (exact ID/path recall), not 32k FP8 (likely still delta≈0) or SnapKV wiring yet. Target ~145 with w2b pilot + 16k 3-category quality run.

## 2026-07-14 — w2b 16k 3-cat pilot

- **Result:** full=1.000 fp8=1.000 delta=0; tool_schema / instruction_supersession / multi_turn_state all 1.0/1.0 at 16k (n=15).
- **Read:** FP8 KV is too gentle to surface PriorityBench failures on these templates at ≤16k. Next stress must be **stronger compression (uniform INT4)** or **32k + harder adversarial templates**, not another FP8 smoke.
