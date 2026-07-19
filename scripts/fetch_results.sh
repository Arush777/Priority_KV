#!/usr/bin/env bash
# Pull H200 scratch runs + logs into local scratch_mirror/ for agent analysis.
# Usage:
#   ./scripts/fetch_results.sh
#   H200_HOST=anupam@169.38.10.80 ./scripts/fetch_results.sh
#   ./scripts/fetch_results.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

H200_HOST="${H200_HOST:-anupam@169.38.10.80}"
H200_SCRATCH="${H200_SCRATCH:-/data/anupam/scratch/prioritykv}"
LOCAL_MIRROR="${LOCAL_MIRROR:-$ROOT/scratch_mirror}"
RSYNC_FLAGS=(-avz --progress)

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "  H200_HOST=$H200_HOST"
      echo "  H200_SCRATCH=$H200_SCRATCH"
      echo "  LOCAL_MIRROR=$LOCAL_MIRROR"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

mkdir -p "$LOCAL_MIRROR/runs" "$LOCAL_MIRROR/logs"

echo "fetching ${H200_HOST}:${H200_SCRATCH}/runs/ → ${LOCAL_MIRROR}/runs/"
rsync "${RSYNC_FLAGS[@]}" \
  "${H200_HOST}:${H200_SCRATCH}/runs/" \
  "${LOCAL_MIRROR}/runs/"

echo "fetching ${H200_HOST}:${H200_SCRATCH}/logs/ → ${LOCAL_MIRROR}/logs/"
rsync "${RSYNC_FLAGS[@]}" \
  "${H200_HOST}:${H200_SCRATCH}/logs/" \
  "${LOCAL_MIRROR}/logs/"

echo "ok mirror=${LOCAL_MIRROR}"
