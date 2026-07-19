#!/usr/bin/env bash
# ONE-TIME H200 bootstrap for agent-remote control (run on dgre2 while you have SSH).
# Starts up to two 1-GPU workers on disjoint empty GPUs (default 0 and 1).
# Usage on H200:
#   bash scripts/h200_bootstrap_pkworker.sh
# Override GPUs:
#   PKWORKER_GPUS="0 1" bash scripts/h200_bootstrap_pkworker.sh
set -euo pipefail
cd /data/anupam/scratch/Priority_KV
GPUS="${PKWORKER_GPUS:-0 1}"
echo "== stopping old pkworker sessions =="
tmux kill-session -t pkworker 2>/dev/null || true
for g in $GPUS; do
  tmux kill-session -t "pkworker${g}" 2>/dev/null || true
done
echo "== syncing rewritten main =="
git fetch origin
git reset --hard origin/main
echo "== clearing local zombie jobs/running (not tracked by git) =="
mkdir -p jobs/running
# Stale claims survive reset --hard and block/confuse the worker.
rm -f jobs/running/*.yaml 2>/dev/null || true
echo "== head =="
git log -1 --oneline
echo "== starting 1-GPU workers on: ${GPUS} =="
# Force line-buffered stderr/stdout so capture-pane shows progress immediately.
for g in $GPUS; do
  tmux new -d -s "pkworker${g}" \
    "stdbuf -oL -eL env REMOTE_WORKER_GPU_FILTER=${g} ./scripts/remote_worker.sh"
  echo "  started pkworker${g} (REMOTE_WORKER_GPU_FILTER=${g})"
done
sleep 5
tmux ls
SCRATCH="${PRIORITYKV_SCRATCH:-/data/anupam/scratch/prioritykv}"
for g in $GPUS; do
  echo "== pkworker${g} pane (tail) =="
  tmux capture-pane -t "pkworker${g}" -p | tail -20 || true
  HB="${SCRATCH}/pkworker_heartbeat_gpu${g}.txt"
  echo "== heartbeat gpu${g} =="
  if [[ -f "$HB" ]]; then
    tail -15 "$HB"
  else
    echo "(no heartbeat yet at $HB)"
  fi
done
echo "== queue dirs =="
ls jobs/pending jobs/running 2>/dev/null || true
echo "== worker processes =="
pgrep -af remote_worker || echo "(no remote_worker process — session may have died)"
echo "OK — leave this host; control from git on the agent box."
echo "Workers only claim jobs whose gpus: field matches their filter (0 / 1 / …)."
