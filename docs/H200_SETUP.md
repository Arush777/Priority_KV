# H200 setup (human-operated box)

Develop + push from the agent machine. On this host: pull, sync, run.

## Two-GPU rule

Shared box has 8× H200. We only use **two**. Default in `.env`:

```bash
CUDA_VISIBLE_DEVICES=6,7
```

Change only if 6/7 are busy. All run scripts load this via `scripts/_env.sh`.

## Commands to run here (keep them bland)

```bash
cd /data/anupam/scratch/Priority_KV   # your checkout
git pull origin main

# CPU deps + unit checks
./scripts/sync.sh

# CUDA deps (torch stack). Still uses only CUDA_VISIBLE_DEVICES from .env
./scripts/sync.sh --cuda

# Re-check units anytime
./scripts/check.sh
```

Verify devices (shows only the two you exported):

```bash
source .venv/bin/activate
set -a && source .env && set +a
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Expect `2` and two H200 names.

## .env

```bash
REPO_ROOT=/data/anupam/scratch/Priority_KV
PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
HF_TOKEN=...                 # real token before model download
HF_HOME=/data/anupam/scratch/prioritykv/hf_cache
CUDA_VISIBLE_DEVICES=6,7
```

```bash
mkdir -p /data/anupam/scratch/prioritykv/{models,datasets,runs,hf_cache}
```

## Later updates

```bash
cd /data/anupam/scratch/Priority_KV
git fetch origin && git reset --hard origin/main   # after force-pushes
# or: git pull origin main
./scripts/sync.sh --cuda    # only if lockfile/deps changed
```

## W1 — FullKV backend compare (G0)

Uses GPUs from `CUDA_VISIBLE_DEVICES` (default 6,7). Runs Transformers then vLLM greedy decode on 20 prompts.

```bash
cd /data/anupam/scratch/Priority_KV
git fetch origin && git reset --hard origin/main
source .venv/bin/activate
set -a && source .env && set +a

python scripts/cmp_gen.py
```

Success line looks like:

```text
n=20 exact=0.xxx tok=0.xxx pass=1 out=/data/anupam/scratch/prioritykv/runs/w1_fullkv/...
```

`pass=1` means gate G0 green. `pass=0` → paste the json path here.

## W1 — byte table + PriorityBench pilot (CPU OK)

These are CPU-only (safe on the H200 login shell; no big GPU use):

```bash
cd /data/anupam/scratch/Priority_KV
git pull origin main   # or fetch + reset --hard if needed
source .venv/bin/activate

python scripts/bytes_report.py
python scripts/mk_bench.py --n 40 \
  --out-dir "$PRIORITYKV_SCRATCH/datasets/prioritybench"
```

Expect `bytes_report` to show all-INT4 floor ≈0.297 and 30% budget with tiny `maxBF16`.
Expect `mk_bench` → `n=40 ... manifest=.../w1_pilot.json`.

## W1 — FP8 KV smoke (GPU)

FullKV vs FP8 KV inside vLLM (on-the-fly scales). Uses 2 GPUs from `.env`.

```bash
cd /data/anupam/scratch/Priority_KV
git pull origin main
source .venv/bin/activate
set -a && source .env && set +a

python scripts/cmp_fp8.py
```

Expect ~10–20 min. Success:
```text
n=20 exact=… tok=… pass=1 out=.../runs/w1_fp8/...
```

Optional later (heavier oneshot calib; needs `uv pip install llmcompressor`):
```bash
python scripts/prep_fp8.py
```

## W2 — page manager smoke (CPU)

```bash
cd /data/anupam/scratch/Priority_KV
git pull origin main
source .venv/bin/activate

./scripts/check.sh
python scripts/check_pages.py
```

Expect `check_pages` JSON with `"invariants_ok":true`.

Do not commit `.env`. Do not run agents on this host.
