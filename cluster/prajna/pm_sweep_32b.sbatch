#!/bin/bash
#SBATCH --job-name=pkv_pm32
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# Scaling check at 32B. Qwen3-32B is ~64 GB in bf16 and does not fit one 46 GB
# L40S, so each task takes 2 GPUs and relies on device_map=auto to shard the
# model. The l40 QOS permits 4 GPUs per job and 4 jobs per user, so up to four
# 2-GPU instances can run concurrently.
#
#   sbatch --array=0-3 cluster/prajna/pm_sweep_32b.sbatch
#
# PKV_PM_CONFIG selects the grid; PKV_PM_SHARD_SIZE the instances per shard.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

PM_CONFIG="${PKV_PM_CONFIG:-configs/pm_sweep_v3.yaml}"
require_local "$PRIORITYKV_SCRATCH/models/Qwen3-32B" "staged Qwen3-32B snapshot"
require_local "$REPO_ROOT/$PM_CONFIG" "sweep config"

cd "$REPO_ROOT"
srun uv run python scripts/run_pm_sweep.py \
  --config "$PM_CONFIG" \
  --model qwen32 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-size "${PKV_PM_SHARD_SIZE:-60}" \
  ${PKV_PM_LIMIT:+--limit-instances "$PKV_PM_LIMIT"}

echo "PM32_SHARD_DONE ${SLURM_ARRAY_TASK_ID:-0} $(date -Is)"
