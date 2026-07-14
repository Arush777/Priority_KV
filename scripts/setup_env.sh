#!/usr/bin/env bash
# One-time (or after pull) environment bootstrap for Priority_KV.
# Safe on CPU machines. On H200, re-run with --gpu after CUDA is visible.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_GPU=0
if [[ "${1:-}" == "--gpu" ]]; then
  WITH_GPU=1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
# Prefer a home-local cache (some shared /tmp sandboxes are not writable).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

echo "==> uv version: $(uv --version)"

if [[ ! -f .env ]]; then
  cp .env.example .env
  # shellcheck disable=SC2016
  echo "REPO_ROOT=$ROOT" >> .env
  echo "Created .env — fill HF_TOKEN / PRIORITYKV_SCRATCH before GPU runs."
fi

if [[ "$WITH_GPU" -eq 1 ]]; then
  echo "==> uv sync --extra gpu --extra dev"
  uv sync --extra gpu --extra dev
else
  echo "==> uv sync --extra dev  (CPU; pass --gpu on H200 for torch/vLLM)"
  uv sync --extra dev
fi

chmod +x scripts/smoke_cpu.sh scripts/setup_env.sh
./scripts/smoke_cpu.sh

echo
echo "Done. Next on H200 GPU node:"
echo "  ./scripts/setup_env.sh --gpu"
echo "  huggingface-cli login   # or export HF_TOKEN=..."
echo "  # then download Qwen3-8B into \$PRIORITYKV_SCRATCH/models"
