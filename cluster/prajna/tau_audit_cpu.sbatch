#!/bin/bash
#SBATCH --job-name=pkv_tau
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Gate P5: generation-free retention audit. Requests NO GPU, so it consumes no
# GPU-hour budget. Never invokes a user simulator, tool backend, or generation.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

require_local "$PKV_TAU_DIR"   "pinned tau-bench trajectories"
require_local "$PKV_MODEL_DIR" "pinned Qwen3-8B tokenizer"

cd "$REPO_ROOT"
uv run python scripts/audit_tau_retention.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  ${PKV_TAU_LIMIT:+--limit "$PKV_TAU_LIMIT"}

echo "TAU_AUDIT_DONE $(date -Is)"
