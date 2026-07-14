# Handoff: PriorityKV W3 — INT4 / quanto_cuda (continue here)

**Audience:** collaborator taking over H200 INT4 debug from Arush’s Cursor session.  
**Date:** 2026-07-15 · **Repo:** https://github.com/Arush777/Priority_KV · **Commits:** W3 code `3abfe3d` · this handoff `f35adeb+` (`main`)  
**There is no Cursor `/export`.** This file + §9 prompt *is* the portability layer. **§A–B below list every H200 command already run and the exact next commands.**

---

## 0. Non-negotiables (read before touching anything)

1. **Deps = `uv` only.** Never `pip install` in this venv. Ad-hoc pip already broke the env once (`torch` → 2.13, broke `vllm==0.25.1`). Recover with `./scripts/sync.sh --cuda` only.
2. **Agents write code on the CCC/agent box; humans run GPU on H200.** Do not run long GPU jobs on the login/agent node. H200 GPUs: `CUDA_VISIBLE_DEVICES=6,7`.
3. **Do not flip `allow_fake_fallback: true`** to “make INT4 green.” W3 assert mode is intentional — silent fake INT4 already lied to us in W2.
4. **Do not retune locked test examples** after SHA256 lock (below). Calibration/val iteration only; locked ids are frozen evidence.
5. **Commits as Arush777** only if Arush asks you to push from his remotes; on your clone use your identity. Prefer PRs if you are not Arush.
6. **Never `pip` into `.venv` for anything** — including `ninja` / build helpers. Use `uv` or a system package manager only.

---

## 1. Dual-machine map

| Machine | User / path | Role |
|---|---|---|
| Agent (IBM CCC) | `arushh` · `/u/arushh/Arush/Priority_KV` | Cursor agents write code, CPU tests, push |
| H200 | `anupam` · `dgre2` · `/data/anupam/scratch/Priority_KV` | Human-only GPU: `git pull`, `uv sync`, runs |
| Scratch / models | `/data/anupam/scratch/prioritykv` · model `Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | `$PRIORITYKV_SCRATCH` |

Git remote: `git@github.com:Arush777/Priority_KV.git`

---

## 2. Project claim (one paragraph)

PriorityKV-Agent: structure-protected mixed BF16/INT4 paged KV for long agent traces. Uniform KV compression can look fine on average metrics while quietly killing tool schemas, instruction supersession, and multi-turn IDs. W2 showed structure keep beats uniform keep at matched budget when state is role/length-separable; buried adversarial scoped that claim. W3 locks the 240-ex bench and must land a **real** uniform INT4 baseline (Q2) — not fake groupwise prefill.

---

## 3. Progress so far

### W2 — CLOSED

- G1 freeze in `docs/decisions.md`: S0 FullKV, S1 FP8 (δ≈0 ≤16k), Q_dropkeep interim eviction.
- Structure @ `keep_frac=0.25` (token-level): structure=1.0 vs uniform/random=0; keep_all=1.0.
- Buried adversarial: structure→0.429 (tool still 1.0; super/multi 0) — no length-oracle leak.
- Q2 INT4 / Q3 SnapKV deferred into W3.

### W3 — CPU + lock DONE; H200 partial

**Landed on `main` (`3abfe3d`):**

| Deliverable | Path / note |
|---|---|
| Locked bench 240 (80/cat) | `data/prioritybench/manifests/w3_lock.json` |
| SHA256 lock | `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89` |
| Audit | `docs/audit_w3.md` (PASS) · W2d preserved 145 · buried 20/80 super+multi; tool 0 |
| Mixed dequant-then-attend ref | `src/prioritykv/mixed_cache_reference.py` + tests |
| INT4 path CPU tests | `tests/test_int4_path_w3.py` |
| Page-level keep | `keep_policy.py` `granularity=page`, floor to token budget |
| Assert-no-fake INT4 | `allow_fake_fallback=False` in `int4_baseline.py` + `configs/w3_int4_assert.yaml` |
| Baselines loud-skip | `scripts/run_w3_baselines_check.py` |

**Cut intentionally (Fable):** `label_page_perturb`, FlashInfer multi-call, attention-KL → W4.

### H200 results already in hand

**Page-level structure** (`configs/w3_structured_paged.yaml`) — **SUCCESS:**

| Arm | mean | notes |
|---|---|---|
| uniform | 0.000 | |
| structure | **0.643** | tool_schema 1.00; instruction_supersession 0.00 (see run JSON for multi) |
| random | 0.286 | |
| keep_all | 1.000 | gate OK |

Out: `/data/anupam/scratch/prioritykv/runs/stress_structured/w3_structured_paged_r1.json`

**Uniform INT4 assert** — **BLOCKED (active bug):**

```text
RuntimeError: INT4 quanto path failed and allow_fake_fallback=False
quanto_impl_error: Error building extension 'quanto_cuda'
  nvcc compiling gptq_marlin_repack / marlin_cuda_kernel under torch JIT extensions
```

So: `optimum.quanto` *imports* (“READY”), but **JIT CUDA extension build fails** on first `generate(..., cache_implementation="quantized")`.

---

## A. Commands already run on H200 (`anupam@dgre2`)

Host path: `/data/anupam/scratch/Priority_KV` · env: `.venv` / `(priority-kv)` · GPUs: `6,7`.

### A1 — DONE (do not redo unless pulling new commits)

```bash
cd /data/anupam/scratch/Priority_KV
git pull
# rebuild gitignored JSONL from lock (done once after W3 pull)
python scripts/mk_bench.py --mode w3_lock          # or: uv run python …
python scripts/audit_bench.py                      # PASS · SHA256 fc44b966…

# PAGE-LEVEL STRUCTURE — SUCCEEDED
# (exports typically already set in their session; CUDA_VISIBLE_DEVICES=6,7)
python scripts/run_stress_structured.py --config configs/w3_structured_paged.yaml
# → out: $PRIORITYKV_SCRATCH/runs/stress_structured/w3_structured_paged_r1.json
# → uniform=0.000 · structure=0.643 · random=0.286 · keep_all=1.000
```

### A2 — FAILED attempts (forensics; do not copy blindly)

```bash
# BAD — broke the env (torch 2.11 → 2.13). NEVER REPEAT.
pip install optimum-quanto
python -m pip install --upgrade pip          # also failed: No module named pip (briefly)
python -m pip install --no-cache-dir --force-reinstall "optimum-quanto"
# → pulled torch 2.13; broke vllm/torchvision; Qwen3ForCausalLM import exploded

# Recovery that was intended (use this if env still broken):
./scripts/sync.sh --cuda                     # uv sync --extra gpu --extra dev

# INT4 assert — LAST COMMAND THAT FAILED (still the active bug)
python scripts/run_w3_baselines_check.py
# printed: quanto INT4: READY … LOUD SKIP SnapKV

python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only
# THIS IS THE LAST FAILED COMMAND. Full error below in §4.
```

### A3 — Diagnostics they already printed

```text
torch 2.11.0+cu130  (before the bad pip) / later briefly 2.13 then sync back
cuda 13.0
cap (9, 0)
gpu NVIDIA H200
nvcc: command not found   when CUDA_HOME unset
CUDA_HOME=                (empty at first diagnostics)
# first RuntimeError line still showed nvcc at /usr/local/cuda/bin/nvcc during JIT
```

---

## B. Exact commands YOU run next (copy in order)

Do **not** re-run page stress unless you changed keep code. Focus = INT4.

```bash
################################################################################
# 0) Identity + pull latest handoff
################################################################################
cd /data/anupam/scratch/Priority_KV
git fetch origin && git checkout main && git pull --ff-only
# expect docs/HANDOFF_W3_INT4.md present

################################################################################
# 1) Restore uv.lock stack (if anyone pip'd) — ALWAYS SAFE
################################################################################
./scripts/sync.sh --cuda
# same as: uv sync --extra gpu --extra dev

################################################################################
# 2) Session env (every new shell)
################################################################################
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=6,7
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
rm -rf ~/.cache/torch_extensions

################################################################################
# 3) HARD GATE — stop here if this exits non-zero
################################################################################
which nvcc
nvcc --version
uv run python - <<'PY'
import re, shutil, subprocess, torch
nvcc = shutil.which("nvcc")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
if not nvcc:
    raise SystemExit("FAIL: nvcc missing — install/link CUDA 13.x toolkit; do not JIT yet")
out = subprocess.check_output(["nvcc", "--version"], text=True)
print(out)
tmaj = str(torch.version.cuda).split(".")[0]
m = re.search(r"release (\d+)\.", out)
nmaj = m.group(1) if m else "?"
if nmaj != tmaj:
    raise SystemExit(f"FAIL: nvcc major {nmaj} != torch.cuda major {tmaj}")
print("OK: toolkit/torch CUDA major match")
PY

################################################################################
# 4) Bench present + baselines status
################################################################################
uv run python scripts/mk_bench.py --mode w3_lock      # no-op-ish if already built; OK to re-run
uv run python scripts/audit_bench.py                  # expect PASS + SHA256 fc44b966…
uv run python scripts/run_w3_baselines_check.py       # quanto READY; SnapKV LOUD SKIP OK

################################################################################
# 5) THE JOB — same as last failed command, with FULL log
################################################################################
uv run python scripts/run_pilot3.py \
  --config configs/w3_int4_assert.yaml \
  --modes int4_only \
  2>&1 | tee /tmp/w3_int4_assert.log

# On success: summary line with modes=[hf_cache_implementation_quantized|quanto_quantized_cache]
# On failure: grep -n "error:\|Error\|FAILED\|undefined" /tmp/w3_int4_assert.log | head -40
# Fix FIRST nvcc/g++ error under gptq_marlin_repack / marlin_cuda_kernel — not torch upgrades.
```

**Done when:** `int4_modes_seen` shows a real quanto mode **and** ≥1 example scored. Then append `docs/decisions.md`.

---

## 4. Active bug — detailed issue write-up

### What we want

Uniform INT4 KV baseline (plan **Q2**) via HuggingFace `cache_implementation="quantized"` / `QuantoQuantizedCache`, with `configs/w3_int4_assert.yaml` setting:

```yaml
int4:
  allow_fake_fallback: false   # MUST stay false
```

Code: `src/prioritykv/int4_baseline.py` → `run_transformers_int4(..., allow_fake_fallback=False)`.

### What happens instead

1. Import check passes → `run_w3_baselines_check.py` prints **`quanto INT4: READY`**.
2. Model weights load (Qwen3-8B, 399 shards).
3. First of 6 prompts starts (`tool_schema.search_docs.v1__c8000__s20260804`).
4. Transformers/quanto tries to **JIT-compile** native extension `quanto_cuda` (Marlin kernels: `gptq_marlin_repack`, `marlin_cuda_kernel`) via `torch.utils.cpp_extension` + `nvcc`.
5. Build fails → caught as `quanto_impl_error` → Path B may also fail → with `allow_fake_fallback=False` we **raise** (correct W3 behavior; not a silent fake):

```text
RuntimeError: INT4 quanto path failed and allow_fake_fallback=False
(id=tool_schema.search_docs.v1__c8000__s20260804
 errors={'quanto_impl_error': "Error building extension 'quanto_cuda':
   [1/9] /usr/local/cuda/bin/nvcc -MD -MF marlin_cuda_kernel.cuda.o.d …"
   # later retries showed [1/6] … gptq_marlin_repack.cuda.o.d …
})
```

The traceback tip truncates the **real** compiler diagnostic. Collaborator must `tee` a full log (§B step 5) and find the first `error:` from `nvcc`/`g++`.

### Why this is hard / likely causes (ranked)

1. **CUDA toolkit major ≠ torch CUDA major** (torch is `2.11.0+cu130` → expect toolkit **13.x**). Mismatch → Marlin JIT fails.
2. **`nvcc` not on PATH** until `CUDA_HOME=/usr/local/cuda` is exported (already seen).
3. Stale/partial JIT cache under `~/.cache/torch_extensions` after failed builds.
4. H200 arch: set `TORCH_CUDA_ARCH_LIST=9.0` (sm_90).
5. Broken env after pip upgraded torch to 2.13 (must `./scripts/sync.sh --cuda` first).

### What is NOT the bug

- Page-level structure stress (already green).
- PriorityBench lock/SHA256 (already audited PASS).
- SnapKV missing (expected LOUD SKIP).
- Assert raising on fake fallback (that is the *feature*).

### Symptom one-liner

```bash
uv run python scripts/run_w3_baselines_check.py   # READY
uv run python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only
# ↑ last failed command: loads model, then quanto_cuda JIT dies on example 0/6
```

Example id at fail: `tool_schema.search_docs.v1__c8000__s20260804`

### Known-bad env state (do not repeat)

Someone ran bare `pip install --force-reinstall optimum-quanto`, which:

- Upgraded `torch` **2.11 → 2.13**
- Broke `vllm` / `torchvision` pins
- Broke `Qwen3ForCausalLM` import via torchvision mismatch

**Recovery:** `./scripts/sync.sh --cuda` (≡ `uv sync --extra gpu --extra dev`) from repo root. Lock pins `torch==2.11.0` and `optimum-quanto==0.2.7`.

### Environment facts when diagnosing

- Host: `dgre2` · H200 · `cap (9,0)`
- Working tree: `/data/anupam/scratch/Priority_KV` · venv `.venv` (Python 3.11)
- Previously: `nvcc: command not found` with empty `CUDA_HOME`, but error line showed `/usr/local/cuda/bin/nvcc` — set explicitly:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="9.0"
  export CUDA_VISIBLE_DEVICES=6,7
  export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
  rm -rf ~/.cache/torch_extensions   # after any failed JIT
  ```
- JSONL splits are **gitignored** — after pull must rebuild:
  ```bash
  uv run python scripts/mk_bench.py --mode w3_lock
  uv run python scripts/audit_bench.py   # expect same SHA256
  ```

### Success criteria for THIS handoff

1. `int4_modes_seen` contains a **real** quanto mode (`hf_cache_implementation_quantized` or `quanto_quantized_cache`), **not** `fake_groupwise_prefill`.
2. At least **one** example is actually scored under that real mode (a zero-example “green” run does **not** count).
3. `configs/w3_int4_assert.yaml` completes without `RuntimeError` under `allow_fake_fallback=false`.
4. Log notes in `docs/decisions.md` (W3 INT4 path status).
5. Optional: SnapKV ≤4-day attempt via `uv sync --extra gpu --extra kvpress` + loud-skip otherwise (G1 keeps DropKeep).

### Acceptable outs if quanto_cuda cannot be fixed on this box

Document loudly in `docs/decisions.md` (Q2 still open / platform blocker). **Do not** quietly score fake INT4 as Q2. Propose next engineering path (prebuilt wheel, different HF cache backend, or PriorityKV’s own INT4 path) — concept calls go to **Fable** (see §6).

---

## 5. Key code / config map

| What | Where |
|---|---|
| INT4 generate + assert | `src/prioritykv/int4_baseline.py` (`allow_fake_fallback`) |
| INT4 quant math | `src/prioritykv/int4_kv.py`, `int4_path.py` |
| Mixed ref attend | `src/prioritykv/mixed_cache_reference.py` |
| Page keep | `src/prioritykv/baselines/keep_policy.py` |
| Stress runner | `scripts/run_stress_structured.py` · `structured_stress.py` |
| Triple/INT4 pilot | `scripts/run_pilot3.py` · `bench_pilot.py` |
| W3 configs | `configs/w3_structured_paged.yaml`, `configs/w3_int4_assert.yaml` |
| Decisions log | `docs/decisions.md` |
| H200 cookbook | `docs/H200_SETUP.md` |
| Week plan | `docs/IMPLEMENTATION_PLAN.md` |
| uv extras | `pyproject.toml` (`gpu` includes `optimum-quanto`; `kvpress` optional) |

---

## 6. How to use Claude like this project does

Arush’s protocol (replicate even on another Anthropic account):

| Model | Role | When |
|---|---|---|
| **Fable** (`claude -p --model fable`) | Senior applied scientist — concept, gates, what to cut | Ambiguous research tradeoffs; “is fake INT4 OK?”; W4 scoping |
| **Opus** (`claude -p --model opus`) | Ruthless code review — MUST-FIX only | After you change INT4 / quanto wiring; before push |

CLI hygiene (bad env vars break Claude Code auth):

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u CLAUDE_CODE_API_KEY \
  claude -p --model fable <<'EOF'
…concept question…
EOF

env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u CLAUDE_CODE_API_KEY \
  claude -p --model opus <<'EOF'
…code review of named files; MUST-FIX or PASS…
EOF
```

In Cursor: you can also @-mention files and ask the agent to shell out to that CLI the same way. Prefer **Ask Fable before changing success criteria**; prefer **Opus before merging** H200-path changes.

---

## 7. First 15 minutes on H200 (checklist)

**Prefer §B** (numbered exact commands). This section is the same flow compressed for muscle memory.

```bash
cd /data/anupam/scratch/Priority_KV
git fetch origin && git checkout main && git pull --ff-only
./scripts/sync.sh --cuda

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=6,7
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
rm -rf ~/.cache/torch_extensions

which nvcc && nvcc --version
# HARD GATE (Opus MUST-FIX): toolkit major must match torch CUDA major (expect 13).
uv run python - <<'PY'
import shutil, subprocess, torch
nvcc = shutil.which("nvcc")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("nvcc", nvcc)
if not nvcc:
    raise SystemExit("FAIL: nvcc missing after CUDA_HOME export — install/link CUDA 13.x toolkit; do not JIT-build yet")
out = subprocess.check_output(["nvcc", "--version"], text=True)
print(out)
tmaj = str(torch.version.cuda).split(".")[0]
# nvcc prints "release 13.0," etc.
import re
m = re.search(r"release (\d+)\.", out)
nmaj = m.group(1) if m else "?"
print(f"gate: torch_cuda_major={tmaj} nvcc_major={nmaj}")
if nmaj != tmaj:
    raise SystemExit(f"FAIL: CUDA toolkit {nmaj} ≠ torch {tmaj} — get matching 13.x toolkit or CUDA-matched torch wheel; do not build")
print("CUDA toolkit/torch major: OK")
PY
uv run python -c "import torch, optimum.quanto as q; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(), q.__version__)"

uv run python scripts/mk_bench.py --mode w3_lock
uv run python scripts/audit_bench.py
uv run python scripts/run_w3_baselines_check.py

# Capture FULL build log, not truncated RuntimeError:
uv run python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only 2>&1 | tee /tmp/w3_int4_assert.log
```

If JIT fails again: open `/tmp/w3_int4_assert.log`, find the **first real nvcc/g++ error** under `gptq_marlin_repack` / `marlin_cuda_kernel`, and fix *that*. Do **not** pip-upgrade torch. If the §7 CUDA major gate failed, fix toolkit/torch match first — Marlin JIT will keep failing on mismatch.

---

## 8. What not to invent

- Do not claim Q2 closed on fake prefill.
- Do not change `w3_lock` gold / scoring templates to chase INT4 numbers.
- Do not run SnapKV until INT4 path is resolved or explicitly deferred with Fable note.
- Do not use `pip` for stack surgery.

---

## 9. Cursor starter prompt (copy-paste for collaborator)

Opus-confirmed (2026-07-15):

```
Continuing PriorityKV-Agent. Read docs/HANDOFF_W3_INT4.md fully — especially §A (commands already run), §B (exact next commands), §4 (issue detail).
Mission: unblock real uniform INT4 (Q2) on H200. Last failed command was:
  python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only
(quanto_cuda JIT build fails; allow_fake_fallback=False correctly raises). Success = int4_modes_seen has hf_cache_implementation_quantized or quanto_quantized_cache AND ≥1 example scored — never fake_groupwise_prefill.
Run §B steps 0→5 exactly (uv sync, CUDA_HOME, CUDA major gate, tee full log). Do not re-run page stress (already green: structure=0.643).
Constraints: uv only; never pip into .venv; do not weaken allow_fake_fallback; do not retune SHA256-locked bench.
Ask Fable for concept/acceptance changes; Opus for code review before push.
```

---

## 10. Contact / ownership

- Research owner: Arush (`Arush777` on GitHub).
- H200 operator for this streak: `anupam@dgre2`.
- When INT4 assert greens: paste the `run_pilot3` summary line + `modes=[…]` back and append `docs/decisions.md`.
