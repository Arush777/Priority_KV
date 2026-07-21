#!/bin/bash
#SBATCH --job-name=pkv_smoke
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Gate P1: single-GPU harness-integrity smoke. docs/PRAJNA_BFCL_HANDOFF.md calls
# this smoke_h100.sbatch, but Prajna has no H100 and no sm_90 device at all: it
# runs on L40S (sm_89). The arch deviation is recorded as DEV_NO_H100.
# Do not submit the primary array unless every check below passes.

source "$SLURM_SUBMIT_DIR/cluster/prajna/_common.sh"
banner

require_local "$PKV_MODEL_DIR"    "pinned Qwen3-8B snapshot"
require_local "$PKV_GORILLA_ROOT" "pinned Gorilla/BFCL checkout"

cd "$REPO_ROOT"

echo "--- 1..2 device identity ---"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
uv run python - <<'PY'
import torch, transformers, kvpress
assert torch.cuda.is_available(), "no CUDA device"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("device:", name, "capability:", cap)
print("torch", torch.__version__, "transformers", transformers.__version__)
try:
    import flashinfer; print("flashinfer", flashinfer.__version__)
except Exception as e:
    print("flashinfer unavailable:", e)
# sm_80+ is required for bf16 SDPA at this scale; sm_90 does not exist on Prajna.
assert cap >= (8, 0), f"unsupported compute capability {cap}"
print("DEVICE_OK")
PY

echo "--- 3 repository tests ---"
uv run pytest tests/test_external_bfcl.py tests/test_tau_spans.py -q

echo "--- 4 FlashInfer packed parity (re-run on the real arch, DEV_NO_H100) ---"
uv run python scripts/run_flashinfer_packed_parity.py \
  --head-dim 128 --seq-len 256 --out-tag prajna_l40s_smoke || echo "PARITY_STEP_FAILED"

echo "--- 5..8 one task through full / structure / snapkv, officially scored ---"
# Self-contained: the smoke gets its own manifest and results root so it never
# reads from, or writes points into, the frozen primary run. --n 4 yields one
# conversation per category, which also exercises all four render paths.
export SMOKE_ROOT="$PRAJNA_ROOT/results/smoke_external_bfcl"
uv run python scripts/prepare_bfcl_external.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  --n 4 --arms full,structure,snapkv --out "$SMOKE_ROOT"

SMOKE_TASK="${PKV_SMOKE_TASK:-}"
if [[ -z "$SMOKE_TASK" ]]; then
  SMOKE_TASK=$(uv run python -c "
import json
from pathlib import Path
rows=[json.loads(l) for l in
      (Path('$SMOKE_ROOT')/'manifest'/'tasks.jsonl').read_text().splitlines() if l.strip()]
rows.sort(key=lambda r: r['prompt_token_count'])
print(rows[0]['task_id'])
")
fi
export SMOKE_TASK
echo "smoke task: $SMOKE_TASK"
uv run python scripts/run_bfcl_external.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  --task-ids "$SMOKE_TASK" --arms full,structure,snapkv \
  --out "$SMOKE_ROOT"

echo "--- 9 interruption and resume ---"
uv run python - <<'PY'
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = os.environ["SMOKE_ROOT"]
store = Path(root) / "points"
before = {p.name for p in store.glob("*.json")}
print("points before resume test:", len(before))
assert before, "no points were produced; resume test is vacuous"
# Re-run: every completed point must be skipped, none duplicated or rewritten.
mt = {p: p.stat().st_mtime for p in store.glob("*.json")}
subprocess.run([sys.executable, "scripts/run_bfcl_external.py",
                "--config", "configs/external_bfcl_prajna_v1.yaml",
                "--task-ids", os.environ["SMOKE_TASK"],
                "--arms", "full,structure,snapkv",
                "--out", root], check=False)
after = {p.name for p in store.glob("*.json")}
assert after == before, f"resume changed the point set: {after ^ before}"
for p, t in mt.items():
    assert p.stat().st_mtime == t, f"resume rewrote a completed point: {p}"
print("RESUME_OK")
PY

echo "--- official scoring ---"
uv run python scripts/score_bfcl_external.py \
  --config configs/external_bfcl_prajna_v1.yaml \
  --out "$SMOKE_ROOT" --tag smoke

echo "SMOKE_DONE $(date -Is)"
