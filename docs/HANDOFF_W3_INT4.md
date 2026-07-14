# Handoff: PriorityKV W3 — INT4 / quanto_cuda (continue here)

**Audience:** collaborator taking over H200 INT4 debug from Arush’s Cursor session.  
**Date:** 2026-07-15 · **Repo:** https://github.com/Arush777/Priority_KV · **Commit at handoff:** `3abfe3d` (`main`)  
**There is no Cursor `/export`.** This file + the starter prompt below *is* the portability layer. Paste the prompt into a fresh Cursor chat on your machine after `git pull`.

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

## 4. Active bug — full forensic context

### Symptom

```bash
uv run python scripts/run_w3_baselines_check.py   # says quanto READY
uv run python scripts/run_pilot3.py --config configs/w3_int4_assert.yaml --modes int4_only
# loads Qwen3-8B, then dies on first example with quanto_cuda build error
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
Continuing PriorityKV-Agent. Read docs/HANDOFF_W3_INT4.md end-to-end (§0 non-negotiables, §3–4) before acting.
Mission: unblock real uniform INT4 (Q2) on H200 for configs/w3_int4_assert.yaml under allow_fake_fallback=False. Success = int4_modes_seen has hf_cache_implementation_quantized or quanto_quantized_cache AND ≥1 example scored — never fake_groupwise_prefill, never 0-example "passes".
FIRST diagnostic: confirm `nvcc --version` major == `python -c "import torch;print(torch.version.cuda)"` major (expect 13). If mismatch or nvcc absent, that IS the bug — get a matching CUDA 13.x toolkit or CUDA-matched torch wheel; do not build until they match.
Constraints: uv only (./scripts/sync.sh --cuda); never pip install anything into .venv, including ninja/build tools. Do not weaken allow_fake_fallback. Do not retune SHA256-locked bench (docs/audit_w3.md). Page-structure run already passed — don't re-litigate.
Env: GPU only on H200, CUDA_VISIBLE_DEVICES=6,7, CUDA_HOME=/usr/local/cuda, TORCH_CUDA_ARCH_LIST=9.0. Keep torch at uv.lock pin (2.11.x).
Repro with a full tee'd log; find the FIRST nvcc/g++ error under gptq_marlin_repack/marlin_cuda_kernel and fix that.
Ask Fable (claude --model fable) for concept/acceptance changes; Opus (--model opus) for code review before push.
```

---

## 10. Contact / ownership

- Research owner: Arush (`Arush777` on GitHub).
- H200 operator for this streak: `anupam@dgre2`.
- When INT4 assert greens: paste the `run_pilot3` summary line + `modes=[…]` back and append `docs/decisions.md`.
