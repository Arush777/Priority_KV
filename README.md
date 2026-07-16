# PriorityKV-Agent

Structure-protected mixed-precision KV cache (BF16/INT4) for long multi-turn agent traces.
Primary hardware: NVIDIA H200 (`dgre2`).

**Headline claim:** uniform KV compression silently breaks tool schemas / instruction hierarchies in long agent traces even when average accuracy looks fine; PriorityKV keeps structure-critical pages so agent reliability holds at matched byte/keep budgets.

Plan: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) (v2.1 execution overlay) · Decisions: [`docs/decisions.md`](docs/decisions.md) · H200 ops: [`docs/H200_SETUP.md`](docs/H200_SETUP.md)

**Status (2026-07-16):** G0–G3 closed · mid-context: structure/P2 **0.688** vs
FixedHot **0.125** · FlashInfer LSE + packed-page multicall **PARITY_PASS** ·
true packed BF16/INT4 storage wired (`storage=packed`) · soft INT4/2-bit quality
gap **falsified** (both 1.000 @0.75) · **missing: FI decode without materialize +
D4 TTFT/TPOT + paper/PR** · full handoff: [`docs/HANDOFF.md`](docs/HANDOFF.md)


---

## Dual-machine workflow (read this first)

| Where | Role |
|---|---|
| **Agent machine** (Cursor / IBM CCC) | Write code, CPU tests, push; optionally enqueue `jobs/pending/*.yaml`; `./scripts/fetch_results.sh` |
| **H200** (`anupam@dgre2` / `169.38.10.80`) | Direct GPU machine — `git pull`, `uv`/`sync.sh --cuda`, run `python scripts/*.py` on GPUs 6,7. No coding agents installed on the box. |

Agents never run *on* the H200. The host is **directly available via SSH** (not a batch queue). `jobs/` + `remote_worker.sh` are optional bookkeeping. **Deps = uv only** — never `pip install` into `.venv`.

```bash
# H200 — one-time bootstrap + worker
./scripts/sync.sh --cuda          # ≡ uv sync --extra gpu --extra dev
export CUDA_VISIBLE_DEVICES=6,7
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
mkdir -p "$PRIORITYKV_SCRATCH/logs"
tmux new -s pkworker './scripts/remote_worker.sh'

# Agent machine — after push, pull artifacts
./scripts/fetch_results.sh        # → scratch_mirror/{runs,logs}/
```

Job queue docs: [`jobs/README.md`](jobs/README.md) · H200 detail: [`docs/H200_SETUP.md`](docs/H200_SETUP.md)
---

## Results so far (H200, Qwen3-8B)

PriorityBench agent-reliability scores (1.0 = programmatic pass). Full evidence in `docs/decisions.md`.

### Uniform compression is too gentle / wrong stress

| Run | Setting | FullKV | FP8 / DropKeep / notes |
|---|---|---|---|
| W2 FP8 @ 16k (3-cat) | vLLM FP8 KV | 1.000 | FP8 **1.000** (δ≈0 ≤16k — not the stress) |
| DropKeep kill | sink+recent ~64× | 1.000 | drop **0.000** (first clear info-loss) |

### Structure at matched keep budget (G2 path b)

Matched `keep_frac=0.25`, prompt-level then page-level. Arms: uniform / structure / random / keep_all.

| Run | Granularity | Full | Uniform | Structure | Random | Keep-all |
|---|---|---|---|---|---|---|
| `stress_structured_25_r1` | token | 1.000 | **0.000** | **1.000** | 0.000 | 1.000 |
| `stress_structured_25_buried_r1` | token + buried gold | 1.000 | 0.000 | **0.429** | 0.000 | 1.000 |
| `w3_structured_paged_r1` | **page** (16 tok) @0.25 | 1.000 | **0.000** | **0.643** | 0.286 | 1.000 |
| `w4_structured_paged_015_r1` | page @0.15 | 1.000 | **0.000** | **0.643** | 0.071 | 1.000 |
| `w4_structured_paged_035_r1` | page @0.35 | 1.000 | **0.000** | **0.643** | 0.429 | 1.000 |

Buried run: structure drops (tool_schema still 1.0; supersession/multi_turn → 0) — scopes the claim to role/length-separable state, not an oracle leak.

### W3 lock

| Item | Value |
|---|---|
| Manifest | `data/prioritybench/manifests/w3_lock.json` |
| Size | 240 examples · 80/category · splits cal/val/test |
| SHA256 | `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89` |
| Audit | [`docs/audit_w3.md`](docs/audit_w3.md) PASS · W2d 145 IDs preserved |

### Open — Q2 uniform INT4

**CLOSED (H200 `w3_int4_assert_r4`):** real mode `hf_cache_implementation_quantized`, n=6 scored, int4_mean=1.000, `allow_fake_fallback: false`. JIT fix = force `-std=c++20` via `prioritykv.cxx20_cuda_ext` in the pilot process (see `docs/decisions.md`).

Older handoff detail remains in [`docs/HANDOFF_W3_INT4.md`](docs/HANDOFF_W3_INT4.md).

---

## Repo layout

```
src/prioritybench/   # PriorityBench-A generator + scorers
src/prioritykv/      # page manager, INT4 path, keep policies, mixed-cache ref
scripts/             # pilots, worker, fetch_results, audit
jobs/                # pending/done/failed queue for H200 worker
tests/               # CPU unit tests
configs/             # frozen run YAMLs (w3_structured_paged, w3_int4_assert, …)
data/prioritybench/  # manifests tracked; JSONL splits gitignored (rebuild with mk_bench)
docs/                # plan, decisions, H200 setup, handoff
```
---

## Quick start

**Agent machine**

```bash
cd /u/arushh/Arush/Priority_KV   # or your clone
./scripts/sync.sh
uv run pytest -q
```

**H200**

```bash
git clone git@github.com:Arush777/Priority_KV.git   # first time
cd /data/anupam/scratch/Priority_KV                   # typical path
git pull
./scripts/sync.sh --cuda
# edit .env: REPO_ROOT, PRIORITYKV_SCRATCH, HF_*, CUDA_VISIBLE_DEVICES=6,7
mkdir -p "$PRIORITYKV_SCRATCH/logs"
tmux new -s pkworker './scripts/remote_worker.sh'   # poll jobs/pending

# Or manual: uv run python scripts/mk_bench.py --mode w3_lock
```
---

## Collaborator / Cursor handoff

1. **[`docs/HANDOFF.md`](docs/HANDOFF.md)** — ground-up status (start here)
2. [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`docs/decisions.md`](docs/decisions.md)
3. Historical INT4 trail: [`docs/HANDOFF_W3_INT4.md`](docs/HANDOFF_W3_INT4.md)

---

## Checklist

- [x] Repo pushes; H200 `git pull`
- [x] CPU smoke / pytest green for W3 refs
- [x] H200: `uv sync --extra gpu` · Qwen3-8B pinned rev on scratch
- [x] W2 closed (FP8 flat; DropKeep kill; structure HIT; buried scope)
- [x] W3 lock + audit SHA256
- [x] W3 page-level structure pilot (`structure=0.643`)
- [x] Q2 real quanto INT4 (`w3_int4_assert`) — GREEN on H200 (`hf_cache_implementation_quantized`)
- [x] Q3 SnapKV ≤4-day attempt or keep DropKeep (loud) — attempt scripted; lock DropKeep if import fails
- [x] W3 15% dual audit (`docs/audit_w3_dual.md`)
- [x] W4 guardrails H200 + G2 formal close (path b)
- [x] W4 confirmatory denser structure sweeps (0.15 / 0.35)
- [x] FlashInfer CUDA deferred (CPU LSE oracle) · atlas fold for denser sweeps
- [x] SnapKV matched-byte → LOCK_Q_DROPKEEP (generate incompatible)
- [ ] W5 P2 structure_risk H200 pilot (`w5_p2_structure_risk_r1`)
- [x] W5 P2 HIT (structure_risk=1.000 vs structure=0.643)
- [x] W5 Q6 FixedHot ablation (`w5_q6_fixedhot_r1`) — fixed_hot=1.000 (ties P2 unburied)
- [x] W6 FlashInfer probe (`w6_flashinfer_probe_r1`) — IMPORT_OK_CUDA_TOUCH v0.6.13
- [x] W5 buried FixedHot vs P2 (`w5_p2_buried_r1`) — FixedHot=P2=1.000; Q7=0.429
- [x] W6 FlashInfer LSE parity — r1 illegal head_dim=32; **r3 native merge PASS @128**
- [x] Harder FixedHot discriminator — mid-context FixedHot 0.125 vs structure/P2 0.688
- [x] Mixed BF16/INT4 quality-forward scaffold (split-prefill corrected; fake quant)
- [x] True packed mixed BF16/INT4 storage path (`packed_mixed_cache` + `storage=packed`)
- [x] FlashInfer packed-page multicall parity (`w6i` / `w6j`, dtype-coalesced)
- [ ] FlashInfer decode without full KV materialize
- [ ] H200 D4 TTFT/TPOT/throughput/Nsight
- [ ] Paper / PR / Gemma / outreach (D5–D9)
