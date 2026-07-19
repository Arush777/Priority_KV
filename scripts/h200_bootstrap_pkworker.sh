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
echo "== clearing local zombie jobs/running (not tracked by git) =="
mkdir -p jobs/running
# Stale claims survive reset --hard and block/confuse the worker.
rm -f jobs/running/*.yaml 2>/dev/null || true
echo "== head =="
git log -1 --oneline
echo "== starting pkworker =="
tmux new -d -s pkworker './scripts/remote_worker.sh'
sleep 3
tmux ls
echo "== last pane lines =="
tmux capture-pane -t pkworker -p | tail -40
echo "== queue dirs =="
ls jobs/pending jobs/running 2>/dev/null || true
echo "OK — leave this host; control from git on the agent box."
