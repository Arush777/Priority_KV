#!/usr/bin/env bash
# Install an hourly crontab entry for collab_bridge tick.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
LOG="${ROOT}/state/cron_${AGENT_ID:-agent}.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv python at $PYTHON — create .venv first" >&2
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a
source "$ROOT/.env"
set +a

AGENT_ID="${AGENT_ID:-agent}"
LOG="${ROOT}/state/cron_${AGENT_ID}.log"
mkdir -p "${ROOT}/state"

LINE="0 * * * * cd ${ROOT} && . ${ROOT}/.venv/bin/activate && python -m collab_bridge tick >> ${LOG} 2>&1"

(crontab -l 2>/dev/null | grep -v 'collab_bridge tick' || true; echo "$LINE") | crontab -
echo "Installed cron:"
echo "  $LINE"
echo "Logs: $LOG"
