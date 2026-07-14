#!/usr/bin/env bash
# Guided Telegram chat_id discovery + bridge smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "Set TELEGRAM_BOT_TOKEN in .env or the environment" >&2
  exit 1
fi

echo "== bot getMe =="
curl -sS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" | python3 -m json.tool

echo
echo "== recent getUpdates (look for chat.id of your group) =="
curl -sS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates" | python3 -m json.tool | head -120

if [[ -n "${TELEGRAM_CHAT_ID:-}" ]]; then
  echo
  echo "== send test message to TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID} =="
  curl -sS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "text=Information_Retrieval collab-bridge smoke test from $(hostname)" \
    | python3 -m json.tool
fi

echo
echo "Next: python -m collab_bridge check && python -m collab_bridge ping"
