#!/bin/bash
#SBATCH --job-name=pkv_bfclp
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:30:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# EXTERNAL_BFCL_TIGHT_V1 budget-ladder pilot: SnapKV only, one keep_frac rung
# per job, on the first PKV_PILOT_N conversations of the frozen sample order.
# FullKV is budget-independent, so the existing EXTERNAL_BFCL_PRAJNA_V1 FullKV
# outcomes serve as the reference ceiling and are not recomputed.
#
#   PKV_PILOT_KF=010 sbatch cluster/prajna/bfcl_tight_pilot.sbatch
#
# The runner checkpoints per conversation and resumes without redoing work.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

KF="${PKV_PILOT_KF:?set PKV_PILOT_KF to 010, 005 or 002}"
N="${PKV_PILOT_N:-60}"
ROOT="$PRAJNA_ROOT/results/external_bfcl_tight_v1/pilot_kf$KF"

require_local "$PKV_MODEL_DIR"    "pinned Qwen3-8B snapshot"
require_local "$PKV_GORILLA_ROOT" "pinned Gorilla/BFCL checkout"
require_local "$ROOT/manifest/work_items.jsonl" "prepared pilot work items ($KF)"

cd "$REPO_ROOT"
srun uv run python scripts/run_bfcl_external.py \
  --config configs/external_bfcl_tight_pilot.yaml \
  --out "$ROOT" \
  --arms snapkv \
  --shard-index 0 \
  --shard-size "$N"

echo "PILOT_RUNG_DONE kf=$KF $(date -Is)"
