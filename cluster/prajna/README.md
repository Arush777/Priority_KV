# Running EXTERNAL_BFCL_PRAJNA_V1 on Prajna

Prajna (IIT Bombay) is a Slurm cluster. **All GPU work goes through `sbatch`**;
use `squeue`/`sacct` to watch it. Never run CUDA on a login node.

Everything below was discovered on the cluster on 2026-07-21, not assumed. Where
it contradicts [`docs/PRAJNA_BFCL_HANDOFF.md`](../../docs/PRAJNA_BFCL_HANDOFF.md),
the cluster wins and the deviation is recorded in
[`configs/external_bfcl_prajna_v1.yaml`](../../configs/external_bfcl_prajna_v1.yaml).

## 1. Site facts

### There is no H100

The handoff assumes 72 H100 GPU-hours at compute capability 9.0. **Prajna has no
H100 and no sm_90 device of any kind.**

| Partition | Required QOS | GPU | Cap | Max GPU/user | Jobs/user | Max wall |
|---|---|---|---|---:|---:|---|
| `dgx` | `dgx` | 8/node, A100-class | 8.0 | 4 | 4 | 6 d |
| `a40` | `a40` | A40 48 GB | 8.6 | 2 | 3 | 4 d |
| `l40` *(default)* | `l40` | **L40S 46 GB** | 8.9 | 4 | 4 | 2 d |
| `interactive` | `interactive` | mixed | — | 8 | 2 | 4 h |
| `debug` | `debug` | A40 | 8.6 | — | — | 30 min |

We run on **`l40` (L40S, sm_89)**: `dgx` was 100% allocated (all 72 GPUs) during
provisioning while `l40` scheduled in under a minute. Qwen3-8B in bf16 is ~16 GB,
leaving ~30 GB for KV on a 46 GB card.

**A partition's QOS is mandatory.** Submitting without `--qos` fails with
`Invalid qos specification`. No account string is needed for this association.

### Compute nodes have no internet

DNS resolution fails outright on compute nodes (`curl https://huggingface.co` →
`Could not resolve host`). Login nodes do have egress. **Every model, dataset,
and wheel must be staged from a login node and referenced by local path.** Jobs
set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` and call `require_local` so a
missing artefact fails immediately instead of hanging on a download.

### Only `$HOME` is writable

`/lustre-scratch`, `/lustre-flash`, and `/scratch` all reject user writes, so
`PRAJNA_ROOT` lives under `$HOME`. Home is Lustre-backed with no enforced quota
(`lfs quota` reports limit 0) and ~613 TB free, but first access to a freshly
written file can take minutes — that is metadata latency, not a hang.

## 2. One-time staging (login node, has internet)

```bash
cp cluster/prajna/config.example.env cluster/prajna/config.env   # edit, chmod 600
source cluster/prajna/config.env
mkdir -p "$PRAJNA_ROOT"/{scratch/models,datasets,results,logs,uv-cache,hf-cache}

# Pinned model (~16 GB)
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B',
                  revision='b968826d9c46dd6066d109eabc6255188de91218',
                  local_dir='$PKV_MODEL_DIR', max_workers=8)"

# Official BFCL: data, stateful API classes, and scorer must come from ONE commit
git clone https://github.com/ShishirPatil/gorilla.git "$PKV_GORILLA_ROOT"
git -C "$PKV_GORILLA_ROOT" checkout cd9429ccf3d4d04156affe883c495b3b047e6b64

# Public tau-bench trajectories (CPU audit only)
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('AgentSuite/tau-bench-trajectories', repo_type='dataset',
                  revision='382e57d1784b55c5155f4ef394ef48f1c747a287',
                  local_dir='$PKV_TAU_DIR', max_workers=8)"

uv sync --frozen --extra gpu --extra kvpress --extra dev --extra external
```

### Why the Gorilla repo and not the Hugging Face mirror

`gorilla-llm/Berkeley-Function-Calling-Leaderboard` on the Hub is **stale**: 23
of 200 `base` questions differ from the repo and `travel_booking.json` disagrees
with the API class the official checker executes. Data, checker, and API
implementations must come from one commit or scores are not comparable to the
leaderboard. The HF revision is recorded in the config but unused.

### BFCL V3 has four categories, not five

V3 multi-turn is `base`, `miss_param`, `miss_func`, `long_context` — 200
conversations each, **800 total**. There is no `composite` category (it arrives
in V4). The handoff's 5×200 split and its 150-composite quota do not exist, so
sampling is balanced 150/150/150/150 for `n=600` and 100 each for the `n=400`
floor.

## 3. Gates

```bash
source cluster/prajna/config.env

# P0/P1 prerequisites: CPU tests + manifest freeze (no GPU)
sbatch --parsable cluster/prajna/bootstrap_cpu.sbatch

# P1: single-GPU harness integrity. Do NOT proceed unless this passes.
sbatch --parsable cluster/prajna/smoke_gpu.sbatch

# P2: pilot — 20 tasks x 5 arms, to measure seconds/task-arm before committing
uv run python scripts/prepare_bfcl_external.py --n 20 \
  --out "$PRAJNA_ROOT/results/pilot_external_bfcl"
sbatch --parsable --array=0-3%2 cluster/prajna/bfcl_array.sbatch

# P3: primary paired table, one GPU per shard, never more than 2 at once
N_SHARDS=$(( ($(wc -l < "$PRAJNA_ROOT/results/external_bfcl_prajna_v1/manifest/work_items.jsonl") + 24) / 25 ))
sbatch --parsable --array=0-$((N_SHARDS-1))%2 cluster/prajna/bfcl_array.sbatch

# P5: CPU retention audit (no GPU, no budget cost)
sbatch --parsable cluster/prajna/tau_audit_cpu.sbatch
```

`%2` is not optional — it enforces the two-concurrent-GPU cap from the handoff
even though the `l40` QOS would allow four.

## 4. Monitoring and recovery

```bash
squeue -u "$USER"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -n 100 pkv_bfcl_${ARRAY_JOB_ID}_${TASK}.out
```

Track **allocated** GPU-hours from `sacct` (`AllocTRES`), not wall-clock in the
logs, and stop before 72.

On timeout or preemption: do not delete partial output. Resubmit the same array;
the runner skips only points that are present *and* valid, retries corrupt or
missing ones, and preserves the failure ledger. `--signal=B:TERM@180` gives the
runner three minutes to finish the current conversation and flush.

If a job is clearly wrong, `scancel` it promptly.

## 5. What must never happen

- No ad-hoc `pip install` — the environment comes from `uv.lock` only.
- No silent truncation: an over-context conversation is excluded with reason
  `MODEL_CONTEXT_LIMIT` and reported.
- No SnapKV fallback: if `kvpress.SnapKVPress` cannot run, the shard fails.
  DropKeep or any other heuristic must never be reported as SnapKV.
- No committing `config.env`, `HF_TOKEN`, model files, raw caches, or huge logs.
- No editing frozen configs or result IDs under `FINAL_RUN_MANIFEST.yaml`.
