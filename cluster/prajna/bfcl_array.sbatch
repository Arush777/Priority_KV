#!/bin/bash
#SBATCH --job-name=pkv_bfcl
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:30:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# Gate P3: primary paired table. One GPU per array task, one shard each.
# Submit with an explicit %2 concurrency cap:
#   sbatch --parsable --array=0-$((N-1))%2 cluster/prajna/bfcl_array.sbatch
# The runner owns point-level integrity: it traps SIGTERM, finishes the current
# conversation, flushes its atomic checkpoint, and writes a shard status file.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

require_local "$PKV_MODEL_DIR"    "pinned Qwen3-8B snapshot"
require_local "$PKV_GORILLA_ROOT" "pinned Gorilla/BFCL checkout"

cd "$REPO_ROOT"
srun uv run python scripts/run_bfcl_external.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-size "${PKV_SHARD_SIZE:-25}"

echo "ARRAY_TASK_DONE ${SLURM_ARRAY_TASK_ID:-0} $(date -Is)"
