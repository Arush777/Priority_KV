#!/usr/bin/env bash
# Shared environment for every Prajna job. Sourced, never executed directly.
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CLUSTER_DIR/../.." && pwd)"

# Personal values live in config.env (git-ignored); the example is tracked.
if [[ -f "$CLUSTER_DIR/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$CLUSTER_DIR/config.env"
else
  # shellcheck disable=SC1091
  source "$CLUSTER_DIR/config.example.env"
fi

export PATH="$HOME/.local/bin:$PATH"

# Prajna ships no CUDA toolkit: compute nodes have no nvcc, spack has no cuda
# build, and /usr/local/cuda is an empty stub. FlashInfer JIT-compiles kernels,
# so point it at a user-installed toolchain if one exists. Absence is not fatal
# for the BFCL run, which uses transformers SDPA + kvpress and never calls
# FlashInfer; only the packed-parity check needs it.
if [[ -x "${PKV_CUDA_HOME:-}/bin/nvcc" ]]; then
  export CUDA_HOME="$PKV_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$PRAJNA_ROOT/logs"

# Compute nodes have no DNS or outbound network: every artefact must already be
# on disk. Fail early and loudly rather than mid-run on a download attempt.
require_local () {
  local path="$1" what="$2"
  if [[ ! -e "$path" ]]; then
    echo "FATAL: $what missing at $path" >&2
    echo "Stage it on the LOGIN node first (compute nodes have no internet)." >&2
    exit 2
  fi
}

banner () {
  echo "=================================================================="
  echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname)"
  echo "partition=${SLURM_JOB_PARTITION:-none} repo=$REPO_ROOT"
  echo "started=$(date -Is)"
  echo "=================================================================="
}
