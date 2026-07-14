#!/usr/bin/env bash
# CPU smoke test — safe on login node / agent machine / H200 (no GPU needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> priority-kv smoke (CPU)"
echo "    root: $ROOT"

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv missing. Run: uv sync --extra dev"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> pytest byte model"
pytest -q tests/test_byte_model.py

echo "==> PriorityBench scorers"
PYTHONPATH=src python scripts/test_prioritybench_scoring.py

echo "==> PriorityBench generator"
PYTHONPATH=src python scripts/test_prioritybench_generate.py

echo "OK — CPU smoke passed."
