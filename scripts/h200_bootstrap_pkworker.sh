#!/usr/bin/env bash
# ONE-TIME H200 bootstrap for agent-remote control (run on dgre2 while you have SSH).
# Usage on H200:
#   curl -fsSL …   OR copy this file after git pull, then:
#   bash scripts/h200_bootstrap_pkworker.sh
set -euo pipefail
cd /data/anupam/scratch/Priority_KV
echo "== stopping old pkworker (if any) =="
tmux kill-session -t pkworker 2>/dev/null || true
echo "== syncing rewritten main =="
git fetch origin
git reset --hard origin/main
echo "== head =="
git log -1 --oneline
echo "== starting pkworker =="
tmux new -d -s pkworker './scripts/remote_worker.sh'
sleep 2
tmux ls
echo "== last pane lines =="
tmux capture-pane -t pkworker -p | tail -25
echo "OK — leave this host; control from git on the agent box."
