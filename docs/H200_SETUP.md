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
git pull origin main
./scripts/sync.sh --cuda    # only if lockfile/deps changed
```

Do not commit `.env`. Do not run agents on this host.
