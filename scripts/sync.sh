#!/usr/bin/env bash
# Dependency sync. Usage: ./scripts/sync.sh [--cuda] [--no-check]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
cd "$ROOT"

WITH_CUDA=0
NO_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --cuda) WITH_CUDA=1 ;;
    --no-check) NO_CHECK=1 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "REPO_ROOT=$ROOT" >> .env
fi

if [[ "$WITH_CUDA" -eq 1 ]]; then
  uv sync --extra gpu --extra kvpress --extra dev -q
else
  uv sync --extra dev -q
fi

chmod +x scripts/*.sh 2>/dev/null || true
if [[ "$NO_CHECK" -eq 0 ]]; then
  "$ROOT/scripts/check.sh"
fi
