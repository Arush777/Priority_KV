#!/usr/bin/env bash
# Thin wrapper. Prefer: ./scripts/check.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/check.sh"
