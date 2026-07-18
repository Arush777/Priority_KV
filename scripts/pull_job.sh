#!/usr/bin/env bash
# Agent-box helper: pull latest worker status/results for a job (no H200 SSH).
# Usage:
#   ./scripts/pull_job.sh w8_fi_greedy_smoke_r1
#   ./scripts/pull_job.sh --watch w8_fi_greedy_smoke_r1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WATCH=0
JOB_ID=""
for arg in "$@"; do
  case "$arg" in
    --watch|-w) WATCH=1 ;;
    -h|--help)
      echo "Usage: $0 [--watch] <job_id>"
      exit 0
      ;;
    *)
      JOB_ID="$arg"
      ;;
  esac
done

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 [--watch] <job_id>" >&2
  exit 2
fi

show() {
  git fetch origin main --quiet
  git merge --ff-only origin/main >/dev/null 2>&1 || true
  echo "=== git tip ==="
  git log -1 --oneline
  echo
  if [[ -f "jobs/status/${JOB_ID}.json" ]]; then
    echo "=== jobs/status/${JOB_ID}.json ==="
    cat "jobs/status/${JOB_ID}.json"
    echo
  else
    echo "(no jobs/status/${JOB_ID}.json yet)"
  fi
  if [[ -d "jobs/results/${JOB_ID}" ]]; then
    echo "=== jobs/results/${JOB_ID}/ ==="
    ls -la "jobs/results/${JOB_ID}/"
    echo
    [[ -f "jobs/results/${JOB_ID}/summary.json" ]] && {
      echo "=== summary.json ==="
      cat "jobs/results/${JOB_ID}/summary.json"
      echo
    }
    [[ -f "jobs/results/${JOB_ID}/nvidia_smi.txt" ]] && {
      echo "=== nvidia_smi.txt (head) ==="
      head -40 "jobs/results/${JOB_ID}/nvidia_smi.txt"
      echo
    }
    [[ -f "jobs/results/${JOB_ID}/log_tail.txt" ]] && {
      echo "=== log_tail.txt (last 40 lines) ==="
      tail -40 "jobs/results/${JOB_ID}/log_tail.txt"
      echo
    }
  fi
  if [[ -f "jobs/done/${JOB_ID}.yaml" ]]; then
    echo "state=done"
    return 0
  fi
  if [[ -f "jobs/failed/${JOB_ID}.yaml" ]]; then
    echo "state=failed"
    return 1
  fi
  if [[ -f "jobs/pending/${JOB_ID}.yaml" ]]; then
    echo "state=pending"
    return 2
  fi
  echo "state=unknown"
  return 3
}

if [[ "$WATCH" -eq 1 ]]; then
  echo "watching ${JOB_ID} (Ctrl-C to stop)…"
  while true; do
    if show; then
      exit 0
    fi
    # failed → exit 1
    st=$?
    if [[ "$st" -eq 1 ]]; then
      exit 1
    fi
    sleep 30
  done
else
  show
fi
