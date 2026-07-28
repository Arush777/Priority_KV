#!/bin/bash
#SBATCH --job-name=pkv_pm
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# SWEEP_PM_V1: protected-mass dose-response sweep. One GPU per array task, one
# shard each. Partition/QOS are overridable at submit time, e.g.:
#   sbatch --array=0-15%4 cluster/prajna/pm_sweep_array.sbatch                  # l40
#   sbatch -p a40 --qos=a40 --array=16-23%2 cluster/prajna/pm_sweep_array.sbatch # a40
# Model selection: PKV_PM_MODEL=qwen|llama (default qwen).
# Config selection: PKV_PM_CONFIG=configs/pm_sweep_v2.yaml (default v1).
# The runner checkpoints per point and skips finished points on resume.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

MODEL_KEY="${PKV_PM_MODEL:-qwen}"
if [[ "$MODEL_KEY" == "llama" ]]; then
  require_local "$PRIORITYKV_SCRATCH/models/Llama-3.1-8B-Instruct" "pinned Llama snapshot"
else
  require_local "$PKV_MODEL_DIR" "pinned Qwen3-8B snapshot"
fi
PM_CONFIG="${PKV_PM_CONFIG:-configs/pm_sweep_v1.yaml}"
require_local "$REPO_ROOT/$PM_CONFIG" "sweep config"
# The manifest path lives in the config; check the one that config names.
PM_MANIFEST=$(grep -E '^manifest:' "$REPO_ROOT/$PM_CONFIG" | awk '{print $2}')
require_local "$REPO_ROOT/$PM_MANIFEST" "sweep manifest ($PM_MANIFEST)"

cd "$REPO_ROOT"
srun uv run python scripts/run_pm_sweep.py \
  --config "$PM_CONFIG" \
  --model "$MODEL_KEY" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-size "${PKV_PM_SHARD_SIZE:-20}" \
  ${PKV_PM_LIMIT:+--limit-instances "$PKV_PM_LIMIT"}

echo "ARRAY_TASK_DONE ${SLURM_ARRAY_TASK_ID:-0} $(date -Is)"
