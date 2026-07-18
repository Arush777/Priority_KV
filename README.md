# PriorityKV-Agent

Structure-protected mixed-precision KV cache (BF16/INT4) for long multi-turn agent traces.
Primary hardware: NVIDIA H200 (`dgre2`).

**One-line point:** Uniform KV *eviction* silently breaks agent tool/instruction/state
reliability; structure-aware retention fixes that at matched budgets. Soft INT4 does
**not** open a PriorityBench quality gap — systems value is **packed bytes + honest latency**.

**Science core: HOME** (`SCIENCE_CORE_HOME_2026_07_19`) · D3 **CLOSED**  
→ Start here: [`RESULTS.md`](RESULTS.md) · freeze: [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml) · D3: [`docs/D3_CLOSE.md`](docs/D3_CLOSE.md)

Plan: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · Decisions: [`docs/decisions.md`](docs/decisions.md) · H200: [`docs/H200_SETUP.md`](docs/H200_SETUP.md) · Paper draft: [`paper/prioritykv_arxiv_draft.md`](paper/prioritykv_arxiv_draft.md)

### Headline numbers (Qwen3-8B, H200)

| Evidence | Result |
|---|---|
| Lock-240 packed quality | full **0.888** · structure **0.883** · uniform **0.879** |
| Matched keep_frac=0.25 (token) | uniform **0.0** · structure **1.0** |
| Latency (structure-FI) | e2e ~**1.12×** FullKV · TPOT ~**1.2×** |
| Payload / peak | payload ~**0.72×** · peak ~**0.87×** (cold-scratch caveat) |
| Guardrails | Δ=**0** · FP8 compare **PASS** · Gemma reduced **PASS** |

---

## Dual-machine workflow (read this first)

| Where | Role |
|---|---|
| **Agent machine** (Cursor / IBM CCC) | Write code, CPU tests, push; enqueue `jobs/pending/*.yaml` |
| **H200** (`anupam@dgre2`) | `git pull`, `uv`/`sync.sh --cuda`, `pkworker` · **max 2 GPUs** · no coding agents on box |

```bash
# H200
./scripts/sync.sh --cuda
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
tmux new -s pkworker './scripts/remote_worker.sh'

# Agent — pull job status
./scripts/pull_job.sh --watch <job_id>
```

Job queue: [`jobs/README.md`](jobs/README.md) · H200 detail: [`docs/H200_SETUP.md`](docs/H200_SETUP.md)

---

## Results archive

Full tables and history: [`RESULTS.md`](RESULTS.md) · narrative log: [`docs/decisions.md`](docs/decisions.md).

Canonical job IDs live in [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml). Failed experiment jobs under `jobs/failed/` are kept for audit (e.g. early Gemma attempts) — they are **not** the claim.

---

## Dev quickstart (agent box / CPU)

```bash
git clone git@github.com:Arush777/Priority_KV.git
cd Priority_KV
./scripts/sync.sh
./scripts/check.sh
```

Git author for pushes: `Arush777 <153831754+Arush777@users.noreply.github.com>`.

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

1. **[`RESULTS.md`](RESULTS.md)** — point of project + metrics (start here)
2. [`docs/HANDOFF.md`](docs/HANDOFF.md) · [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml)
3. [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`docs/decisions.md`](docs/decisions.md)

---

## Checklist (science core)

- [x] PriorityBench-A lock-240 + audit
- [x] Structure ≫ uniform matched-keep (token + page); buried scoped
- [x] Soft INT4 quality gap falsified @ 0.75
- [x] Packed BF16/INT4 + FI decode (D3 CLOSED; cold-scratch caveat)
- [x] Lock-240 / latency M3c / peak-mem canonical jobs
- [x] Publish GPU: FP8 compare · guardrails · Gemma reduced
- [ ] arXiv submit · blog · upstream PR · outreach (DeepMind packaging)
