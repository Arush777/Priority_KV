#!/usr/bin/env bash
# Load local .env if present (never printed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
# Cap to two devices unless already set in the environment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PATH="$HOME/.local/bin:${PATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"
