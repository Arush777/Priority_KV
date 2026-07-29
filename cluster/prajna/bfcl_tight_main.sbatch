#!/bin/bash
#SBATCH --job-name=pkv_bfclm
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
# EXTERNAL_BFCL_TIGHT_V1 main test at the pre-registered budget (keep_frac=0.15,
# selected by the Section 3 rule from the pilot ladder). Arms: full, snapkv, adapt.
#   sbatch --array=0-11%4 cluster/prajna/bfcl_tight_main.sbatch
source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner
ROOT="$PRAJNA_ROOT/results/external_bfcl_tight_v1/main_kf015"
require_local "$PKV_MODEL_DIR" "pinned Qwen3-8B snapshot"
require_local "$ROOT/manifest/work_items.jsonl" "prepared main work items"
cd "$REPO_ROOT"
srun uv run python scripts/run_bfcl_external.py \
  --config configs/external_bfcl_tight_main.yaml \
  --out "$ROOT" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-size "${PKV_MAIN_SHARD:-75}"
echo "MAIN_SHARD_DONE ${SLURM_ARRAY_TASK_ID:-0} $(date -Is)"
