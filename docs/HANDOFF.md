# PriorityKV — Project Handoff (2026-07-16)

**Audience:** next collaborator / agent / reviewer picking up the repo cold.  
**Read order:** this file → [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) → [`decisions.md`](decisions.md) → [`H200_SETUP.md`](H200_SETUP.md).  
**Older INT4-only handoff:** [`HANDOFF_W3_INT4.md`](HANDOFF_W3_INT4.md) (kept for history; superseded for current status).

**Repo:** `github.com:Arush777/Priority_KV` · branch `main`  
**Primary model:** Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`  
**Primary hardware:** NVIDIA H200 on `dgre2` (`anupam@169.38.10.80`) · GPUs **6,7** by default  
**Scratch:** `/data/anupam/scratch/prioritykv` · clone often `/data/anupam/scratch/Priority_KV`

---

## 1. What we are trying to achieve (first principles)

### The systems problem

Autoregressive Transformers store past tokens in a **KV cache**. For long multi-turn *agent* traces (tool schemas, superseding instructions, early IDs), that cache dominates memory and often latency. Serving stacks therefore compress or evict KV (FP8, INT4, SnapKV-style drop, StreamingLLM sink+recent).

### The reliability bet

**What you drop or make missing matters more than how much you compress.**  
Uniform eviction / missing-state can keep average long-context scores looking fine while silently destroying tool-schema conformance, instruction supersession, and multi-turn state — the failure modes that matter for agents.

### The product bet (systems half)

A **structure-protected mixed BF16/INT4 paged cache**:

- Keep sink / recent / tool / system / constraint pages in BF16.
- Demote filler to packed INT4 under a byte budget.
- Attend with **FlashInfer homogeneous page multicall + native LSE merge**.
- Show **measured** H200 TTFT / TPOT / throughput gains vs FullKV and calibrated vLLM FP8 **at equal agent quality** (or an honest reliability-at-parity reframe).

### Locked headline claim (use this wording)

> Uniform KV **eviction / missing-state** silently corrupts tool schemas and instruction hierarchies in long multi-turn agent traces. Structure-aware retention preserves those traces at matched keep budgets. A structure-protected BF16/INT4 paged cache targets the same reliability below the all-INT4 byte floor, with measured end-to-end serving gains on H200 and FlashInfer-verified attention.

**Do not claim** that uniform *INT4 quantization* hurts PriorityBench quality at the tested operating point (corrected 4-bit and 2-bit quality forwards both stayed **1.000**). That hypothesis was falsified; systems value must come from **packed bytes + latency**, not a fake-quant quality gap.

### Two experiment families (do not mix them)

| Family | What changes | What it answers |
|---|---|---|
| **(A) Matched KEEP** | Drop tokens; regenerate at BF16 | Which tokens matter for agent reliability |
| **(B) Matched MIXED DTYPE** | Keep all tokens; some positions INT4 / zero | Role-aware precision / byte floor / serving |

---

## 2. What has been done

### Gates

| Gate | Intent | Status |
|---|---|---|
| **G0** | Env + FullKV stable | **CLOSED** |
| **G1** | Freeze baselines | **CLOSED** (SnapKV → DropKeep interim) |
| **G2** | Structure ≥3pt or INT4 drop | **CLOSED path (b)** — matched keep |
| **G3** | Q6/Q7/Q8/P2 ablations | **CLOSED (with honest negative)** |
| **G4** | Freeze final run manifest | **CLOSED** — `FINAL_RUN_MANIFEST.yaml` (middle-ground 2026-07-19) |

### Reliability evidence (family A) — strong

- Uniform keep @ matched budget → **0.000**; structure page keep → **0.643** across keep_frac 0.15 / 0.25 / 0.35 (`w3`/`w4_structured_paged_*`).
- Mid-context discriminator (gold relocated off the prefix): **FixedHot 0.125** vs **structure / P2 0.688**; uniform **0**.
- Honest negative: once position is controlled, **P2 ≃ Q7** (linear risk does not beat pure structure on this set).
- Guardrails (RULER / SCBench-style): **Δ = 0** (no blanket long-context collapse).
- PriorityBench-A **locked**: 240 examples, SHA256 `fc44b966…ae89` · audit PASS.

### Systems / mixed path (family B) — partially closed

| Piece | Status | Key artifact / job |
|---|---|---|
| Real uniform INT4 (quanto) | **GREEN** | `w3_int4_assert_r4` · `hf_cache_implementation_quantized` |
| Per-position dtype planner | **Done** | `mixed_kv.plan_int4_mask` (structure vs uniform, matched `int4_frac`) |
| Quality-forward mixed harness | **Done + corrected** | Split-prefill; `first_token_from_degraded_cache=true` |
| Soft INT4 / 2-bit quality gap | **Falsified** | Corrected replays: both policies **1.000** @ 0.75 |
| Zero-degrade wiring proof | **PASS** | structure **0.688** ≫ uniform **0.312** |
| FlashInfer LSE multicall | **PARITY_PASS** | `w6e` max abs **4.88e-4** (`merge_state`, head_dim=128) |
| **True packed BF16/INT4 storage** | **Landed** | `packed_mixed_cache.py` · wired into `mixed_kv_run` (`storage=packed`) |
| FI over packed pages (coalesced) | **PARITY_PASS** | `w6i` synthetic · `w6j` serve gate on layers 0/18/35 |
| FI attention *instead of* SDPA decode | **Not done** | Decode still materializes → Transformers SDPA |
| D4 TTFT/TPOT/Nsight | **Not started** | — |
| Paper / PR / Gemma / outreach | **Not started** | D5–D9 |

### Bugs found and fixed (do not reintroduce)

1. **FlashInfer LSE contract:** FI LSE is native/base-2 → use `flashinfer.merge_state`, never NumPy natural-log merge on FI LSE.
2. **Mixed harness first-token leak:** must split-prefill `n-1`, degrade, replay last prompt token.
3. **SM90 head_dim:** only `{64,128,256}`; Qwen3 uses **128**.
4. **Many tiny run-length pages + `merge_state`:** coalesce by dtype (one BF16 chunk + one INT4 chunk) before FI multicall (`9be6128`).
5. **Deps:** never `pip` into `.venv` on H200 — **`uv` / `./scripts/sync.sh --cuda` only**.

### Key code map

| Path | Role |
|---|---|
| `src/prioritybench/` | Bench generator + scorers |
| `src/prioritykv/page_manager.py` | Role/dtype page table + budget demotion |
| `src/prioritykv/mixed_kv.py` | Per-position INT4 mask planner |
| `src/prioritykv/packed_mixed_cache.py` | **True** BF16 tensors vs packed INT4 pages |
| `src/prioritykv/mixed_kv_run.py` | Prefill → pack/degrade → decode (`storage`, `attn_backend`) |
| `src/prioritykv/flashinfer_multicall.py` | Page multicall + `merge_state` + packed parity |
| `src/prioritykv/baselines/keep_policy*.py` | Matched-keep stress (G2b / G3) |
| `scripts/run_mixed_serve.py` | H200 mixed-serve driver |
| `scripts/run_flashinfer_packed_parity.py` | Packed mixed BF16/INT4 FI parity |
| `jobs/` | Optional bookkeeping for runs (pending/done/failed) |

---

## 3. What metrics we are targeting

### Agent reliability (primary motivator)

- **PriorityBench-A score** ∈ {0,1} mean over tool_schema / instruction_supersession / multi_turn_state.
- Target story: structure (or P2) **≫** uniform / FixedHot at **matched** `keep_frac` (operating point often **0.25**).
- Guardrails: RULER (2 tasks) + SCBench (2) + optional MATH-500 — **no regression** (already Δ≈0 on tested probes).

### Systems / serving (headline product)

| Metric | Target / comparison |
|---|---|
| **Realized bytes** | ≤ **50%** and **30%** of FullKV BF16 (`byte_model.realized_bytes`) |
| **Compression ratio** | Packed mixed already ~**0.47** at int4_frac≈0.75 in `w6j` smoke |
| **Agent quality at equal bytes** | Prefer structure-mixed ≥ uniform-INT4 / FP8; if quality ties, win on latency/mem |
| **TTFT / TPOT / throughput** | Beat calibrated **vLLM FP8** and FullKV at equal agent quality (D4) |
| **FlashInfer parity** | Multicall vs dense max abs **≪ 5e-2** (achieved ~1e-3–1e-4) |
| **Nsight** | Optional roofline once D4 numbers exist |

### What we are *not* optimizing for anymore

- Escalating fake-quant severity (nbits=2, int4_frac=0.92, …) to force a PriorityBench INT4 quality drop — **falsified / result-seeking**.

---

## 4. How far we are in the implementation

```
Reliability story     ████████████████████████  100%  (G0–G4 closed; middle-ground)
Packed FI serving     ███████████████████████░   ~95%  (shim decode; cold-scratch caveat)
Serving metrics       ████████████████████████  100%  (M3c latency + peak-mem + lock-240)
Publish track         ████████████████████░░░░   ~80%  (GPU jobs done; packaging open)
```

**One-line status:** Science core **HOME** (`SCIENCE_CORE_HOME_2026_07_19`). D3 **CLOSED**
with cold-scratch caveat. Soft-INT4 quality gap falsified; claim is eviction
reliability + packed bytes + honest latency. DeepMind track remaining = paper/blog/PR/outreach.

### Deliverable checklist

| ID | Deliverable | Progress |
|---|---|---|
| D1 | PriorityBench-A | 🟢 Locked + audited |
| D2 | Failure atlas | 🟢 Folded denser sweeps |
| D3 | Mixed paged backend | 🟢 Packed + FI shim decode (cold-scratch caveat) |
| D4 | TTFT/TPOT + peak/payload | 🟢 M3c + `mg_a` + lock-240 |
| D5–D9 | PR / paper / blog / outreach | ⬜ Out of middle-ground scope |
| D10 | Pallas/TPU appendix | ⬜ conditional buffer |
| G4 | Final run manifest | 🟢 `FINAL_RUN_MANIFEST.yaml` |

---

## 5. Next steps / still pending (priority order)

**Middle-ground close: DONE (G4 frozen 2026-07-19).**  
Canonical artifacts + claim: [`FINAL_RUN_MANIFEST.yaml`](../FINAL_RUN_MANIFEST.yaml).

Optional later (not required for this close): thin guardrails re-check; publish
track; LongBench/RULER; Gemma.

**Do not:** reopen soft-INT4 severity hunts; grow D4 latency n for optics.

---

## 6. Current blockers

| Blocker | Type | Severity | Notes |
|---|---|---|---|
| None for middle-ground close | — | — | G4 frozen |
| FI cold scratch ≈ BF16 peak | Systems honesty | Low | Documented in peak-mem job |
| Publish / LongBench / Gemma | Out of scope | — | Only if scope reopens |

---

## 7. Ops & how to run work

### Machines

| Where | Role |
|---|---|
| Laptop / agent box | Code, docs, CPU pytest, git push |
| **H200 (`anupam@169.38.10.80`, `dgre2`)** | Direct GPU runs: `git pull`, activate `.venv`, `CUDA_VISIBLE_DEVICES=6,7`, run `python scripts/….py` |

Coding agents (Cursor/Claude) should **not** be installed on H200. SSH from the agent machine to run allowlisted scripts is fine — H200 is a **directly available** machine, not a submit-and-wait batch queue. `jobs/pending/*.yaml` + `remote_worker.sh` remain optional bookkeeping.

### Typical H200 session

```bash
ssh anupam@169.38.10.80
cd /data/anupam/scratch/Priority_KV
git fetch origin && git reset --hard origin/main   # or git pull
source .venv/bin/activate
set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=6,7
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
export PYTHONPATH=$PWD/src

# Packed FI parity (fast)
python scripts/run_flashinfer_packed_parity.py --head-dim 128 --out-tag r1

# Mixed serve with packed + FI parity gate
python scripts/run_mixed_serve.py --config configs/w6_mixed_serve_flashinfer.yaml
```

Fetch artifacts to the laptop:

```bash
./scripts/fetch_results.sh   # → scratch_mirror/{runs,logs}/
```

### Deps rule

**Only** `./scripts/sync.sh` / `uv sync` (use `--cuda` on H200). Ad-hoc `pip` already corrupted torch/vLLM once.

### Recent green H200 jobs (2026-07-16)

| Job | Result |
|---|---|
| `w6e_flashinfer_lse_parity_r3` | PARITY_PASS · max abs 4.88e-4 |
| `w6i_flashinfer_packed_parity_r1` | PARITY_PASS · packed mixed pages |
| `w6j_mixed_serve_flashinfer_r1` | exit 0 · FI gate PASS · compression ≈0.47 · n=3 smoke |

---

## 8. Claim hygiene (what you may / may not say)

**Allowed**

- Uniform eviction/missing-state kills agent tasks; structure keep preserves them at matched budget.
- Mid-context FixedHot fails; structure holds.
- Soft INT4/2-bit at 75% does **not** separate policies on this set; zero-stress proves the planner.
- Packed mixed storage realizes ~half FullKV bytes; FlashInfer page multicall matches dense within tolerance.

**Not allowed**

- “Uniform INT4 quantization breaks PriorityBench” (at tested OP).
- “P2 uniquely beats FixedHot / Q7” without the mid-context caveats.
- “Shipping mixed paged server with FI decode” (materialize+SDPA still in the hot path).
- “Measured serving speedup” before D4.

---

## 9. Suggested first tasks for the next owner

1. Skim this handoff + `decisions.md` entries from 2026-07-15 onward.  
2. On H200: re-run `run_flashinfer_packed_parity.py` once to confirm env.  
3. Implement **FI decode path** (attention hook over coalesced packed pages + BF16 decode tail).  
4. Prototype a **D4 microbench** (TTFT/TPOT) on a short PriorityBench slice vs FullKV.  
5. Only then expand to paper-facing matrices and Gemma.

---

## 10. Pointers

| Doc | Why |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Scope, deliverables, week overlay |
| [`decisions.md`](decisions.md) | Chronological evidence + pivots |
| [`H200_SETUP.md`](H200_SETUP.md) | Host paths, two-GPU rule, sync |
| [`failure_atlas.md`](failure_atlas.md) | Reliability figures |
| [`HANDOFF_W3_INT4.md`](HANDOFF_W3_INT4.md) | Historical Q2 INT4 debug trail |
| [`../README.md`](../README.md) | Quick status + results tables |

**Canvas (optional):** Cursor comprehensive report under the local canvases folder (`PriorityKV-comprehensive-report.canvas.tsx`).

---

*Last updated: 2026-07-16 · after `w6i` / `w6j` green and dtype-coalesced FlashInfer multicall.*
