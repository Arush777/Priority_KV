#!/bin/bash
#SBATCH --job-name=pkv_bootstrap
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# CPU-only staging + manifest freeze. Requests no GPU.
# Everything it needs must already be on disk: submit only after the login-node
# staging step (see cluster/prajna/README.md) has completed.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

require_local "$PKV_MODEL_DIR"    "pinned Qwen3-8B snapshot"
require_local "$PKV_GORILLA_ROOT" "pinned Gorilla/BFCL checkout"
require_local "$PKV_TAU_DIR"      "pinned tau-bench trajectories"

cd "$REPO_ROOT"
echo "--- locked environment ---"
uv sync --frozen --extra gpu --extra kvpress --extra dev --extra external

echo "--- CPU tests ---"
uv run pytest tests/test_external_bfcl.py tests/test_tau_spans.py -q
uv run ruff check src/prioritykv/external scripts/prepare_bfcl_external.py \
  scripts/run_bfcl_external.py scripts/score_bfcl_external.py scripts/audit_tau_retention.py

echo "--- freeze manifest ---"
uv run python scripts/prepare_bfcl_external.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  --n "${PKV_N:-600}"

echo "BOOTSTRAP_DONE $(date -Is)"
