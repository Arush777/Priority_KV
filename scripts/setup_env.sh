#!/usr/bin/env bash
# Thin wrapper kept for older notes. Prefer: ./scripts/sync.sh [--cuda]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == "--gpu" ]]; then
  exec "$ROOT/scripts/sync.sh" --cuda
fi
exec "$ROOT/scripts/sync.sh"
