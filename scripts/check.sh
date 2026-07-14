#!/usr/bin/env bash
# Local unit checks (no device required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "missing .venv — run: ./scripts/sync.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pytest -q tests/test_byte_model.py
PYTHONPATH=src python scripts/test_prioritybench_scoring.py
PYTHONPATH=src python scripts/test_prioritybench_generate.py
echo ok
