# PriorityKV-Agent — Implementation Plan (v2.1, execution overlay)

**Supersedes:** PRIORITYKV_RESEARCH_PLAN.md (v1)  
**Plan lineage:** v2 DeepMind-targeted scope (below) · **v2.1 = this file**, annotated with what actually shipped through **2026-07-15**  
**Execution status log:** [`docs/decisions.md`](decisions.md) · **H200 / INT4 handoff:** [`docs/HANDOFF_W3_INT4.md`](HANDOFF_W3_INT4.md) · **README results:** [`../README.md`](../README.md)

**Team:** originally scoped as 2 students (A = research/eval, B = systems); **current execution is solo-first** with an H200 collaborator for GPU runs. Owner labels A/B still describe *workstreams*, not headcount.  
**Duration:** 10 weeks core + 2-week buffer · **Calendar W0 in v2 copy = Mon 2026-07-20**; **actual work began ~2026-07-14** (ahead of formal W0). Treat week numbers as *gates*, not calendar prisons.  
**Primary hardware:** H200 (`dgre2`, `CUDA_VISIBLE_DEVICES=6,7`) · Agent box writes code / CPU only  
**Primary model:** Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218` · Gemma secondary still planned  
**Deps:** **`uv` / `./scripts/sync.sh --cuda` only** — never ad-hoc `pip` into `.venv`

---

## Status snapshot (2026-07-15) — read this first

| Gate / item | Plan intent | Actual |
|---|---|---|
| **G0** | Env + FullKV stable | **Met** — vLLM FullKV pilots green |
| **G1** (end W2) | Freeze baselines; FP8/INT4/SnapKV reproduce | **Met with written deferrals** — FullKV + FP8 frozen; Q2 now **real** on H200; Q3 SnapKV → DropKeep lock attempt |
| **G2** (end W4) | ≥5pt uniform drop *or* structure≥3pt vs uniform | **CLOSED path (b)** — page structure 0.643 vs uniform 0.000 @ 0.15/0.25/0.35; guardrails gate Δ=0 |
| **D1** PriorityBench-A 240 + audit | End W3 | **Lock+auto-audit+15% dual audit PASS** · SHA256 `fc44b966…` |
| **W3 page-perturb labels** | Begin W3 | **Moved to W4** — `scripts/label_page_perturb.py` + `configs/linear_risk_fit.json` |
| **INT4 append/decode ref** | W3–4 | **CPU numpy ref + tests green** · **H200 Q2 GREEN** (`hf_cache_implementation_quantized`, n=6) via C++20 JIT patch |
| **Multi-call + LSE (FlashInfer)** | Begin W3–4 | **CPU LSE multi-call == dense mixed attend** · FlashInfer CUDA **DEFERRED_W5_W6** (loud-skip) |
| **Guardrails RULER/SCBench** | W2 harness | **PASS** on H200 (`guardrails_w4_r2`, gate Δ=0) |

**Do not claim Q2 closed on fake groupwise INT4.** W2 already showed quiet fake-INT4 / quanto-miss can look “perfect.” **Real Q2 path is green (2026-07-15).**

---

## 0. What changed from v1 and why

| v1 | v2 | Reason |
|---|---|---|
| 3 contributions (benchmark + policy + backend) | 1 headline (backend) + 1 motivator (agent-reliability finding) | Guarantee a *finished* artifact in 10 weeks |
| BF16 + FP8 + INT4 storage classes | **BF16 + INT4 only**; calibrated FP8 kept as the *baseline to beat* | Halves kernel/test surface; FP8 storage added only if trivially free |
| Linear + GBT + MLP predictors | **Linear/heuristic predictor only** | v1's own falsification path predicted this outcome; skip to it |
| 480-example PriorityBench, 6 categories | **PriorityBench-A: 240 examples, 3 agentic categories** | Tool schemas, instruction supersession, multi-turn persistence — the DeepMind-legible failure modes |
| RULER + LongBench v2 + SCBench + MATH + AIME full matrices | RULER (2 tasks) + SCBench (2 multi-turn tasks) + MATH-500 greedy as *guardrails* | Guardrails prove no regression; they are not the story |
| Llama-3.1-8B secondary | **Gemma secondary** | Google-ecosystem hook; a Gemma reliability finding is a natural reason for DeepMind people to read the work |
| Workshop paper + upstream PR as stretch | **Workshop submission + upstream PR + blog post are core deliverables** | These are the artifacts that actually move a DeepMind RE application |
| No interview-prep track | Parallel 6 h/week prep track from Week 4 | DeepMind RE loops hinge on coding + ML fundamentals regardless of portfolio |
| — | Optional Pallas/TPU appendix (Week 11–12 buffer only) | Speaks DeepMind's stack; strictly conditional |

**Headline claim being built:**
> "Uniform KV-cache compression silently corrupts tool schemas and instruction hierarchies in long multi-turn agent traces even when average benchmark accuracy is flat. A structure-protected mixed-precision paged cache (BF16/INT4) removes these failures at 25–30% of FullKV bytes with measured end-to-end serving gains on H200, implemented and verified against vLLM/FlashInfer."

Everything below exists to make that one paragraph true, reproducible, and public.

### 0.1 Execution corrections vs original v2 text (learned on the wire)

These are **not** scope expansions — they are precision updates so the plan matches reality:

| Original v2 assumption | What we learned | Current practice |
|---|---|---|
| FP8 / gentle INT4 would show PriorityBench drops ≤16k | FP8 δ≈0; wrong `group_size` kw → quanto never engaged; weak fake INT4 stayed ~1.0 | Prefer **matched-budget eviction / keep** stress + **assert-no-fake** INT4 (`allow_fake_fallback=false`) |
| SnapKV ready by G1 | Scaffold / loud-skip only | **Q_dropkeep** interim eviction (StreamingLLM-style sink+recent, **prompt-level / RoPE-safe** — KV-tensor surgical drop broke RoPE) |
| Structure tagging safe | `"FINAL"` substring → RECENT smelled like oracle | Removed; trailing ask via `force_recent` only |
| Structure = universal win | Buried-state adversarial: structure **0.429** (tool still 1.0) | Claim scoped: wins when state is **role/length-separable**; buried free-form needs better tags / page risk |
| Page-perturb + KL in W3 | Timeboxed cut | **W4**; score_delta tuples OK if revived early |
| `pip install` OK on H200 | Broke torch 2.11→2.13 / vLLM | **`uv sync` / `./scripts/sync.sh --cuda` only** |

---

## 1. Locked scope decisions (change requires written note in `docs/decisions.md`)

1. **Models:** Qwen3-8B primary (revision pinned above); Gemma secondary (reduced matrix only). One small model for unit tests, never for evidence.
2. **Storage classes:** BF16 and INT4 (group-wise asymmetric, KIVI-style, group size 32 — confirm against KVPress/quanto once Q2 runs). No FP8 *storage* in the shipped policy; **calibrated vLLM FP8 remains S1 baseline**. No 2/3-bit. Eviction only as baseline (now: DropKeep; SnapKV if reproduced).
3. **Page layout:** backend-native **16-token** physical pages; 128-token allocation units (one ablation at 64). **Page-granularity keep** stress implemented (floor page count to token budget).
4. **Policy (shipping target):** ProtectedRole++ — deterministic structural rules (protect system/tool/constraint/sink/recent-window in BF16, rest INT4) plus a calibrated **linear** risk score only to break ties. No MLP/trees in the shipped system. **Today:** structure keep policies + page manager scaffolding; linear risk **not fitted yet** (needs atlas / labels).
5. **Budgets:** 50% and 30% of FullKV bytes for final comparisons; **matched `keep_frac=0.25`** used as the early G2 path-(b) operating point on H200.
6. **Backend:** multi-call homogeneous paged attention (FlashInfer) + exact LSE merge — **CUDA deferred W5–6**. CPU **dequant-then-attend** + `lse_merge_pair` / `mixed_attend_kv_multicall` are the correctness oracle.
7. **Serving comparison target:** calibrated vLLM FP8. Beating it at *equal agent quality* at ~30% bytes is the systems win. Reliability-at-parity remains an allowed reframe (G4).

---

## 2. Deliverables (all core) — progress

| # | Deliverable | Due | Status |
|---|---|---|---|
| D1 | PriorityBench-A: 240 scored ex + generator + audit | End W3 | **🟢 Lock+audit+generator** · manual dual audit open · templates v2 non-leaking |
| D2 | Failure-atlas tech note + headline figures | End W4 | **🟢** denser 0.15/0.25/0.35 folded (`docs/atlas_w4_structure_rows.jsonl`) · Q2 real INT4 logged |
| D3 | Mixed-precision paged backend + correctness suite | End W6 | **🟡** page manager / tagging / INT4 CPU+H200 path · FlashInfer CUDA deferred · CPU LSE ✅ |
| D4 | H200 TTFT/TPOT/throughput + Nsight | End W9 | **⬜** |
| D5 | Upstream PR | Open W8 | **⬜** |
| D6 | Workshop paper | Draft W9 | **⬜** (CFP check still due) |
| D7 | Public blog + repro | W10 | **⬜** |
| D8 | One-command smoke + full-run script | W10 | **🟡** partial scripts; not one-command clean-machine yet |
| D9 | Outreach log ≥6 | Continuous | **⬜** |
| D10 | Pallas/TPU appendix (buffer) | W11–12 | **⬜** conditional |

---

## 3. Workstream A — PriorityBench-A (agent reliability)

### 3.1 Categories (80 examples each, 240 total) — **LOCKED**

1. **Tool-schema conformance** — JSON-schema + required/enum checks.  
2. **Instruction supersession** — latest-constraint checkers (v2 templates: **no gold leakage** in FINAL ask).  
3. **Multi-turn state persistence** — exact-match slots (v2 non-leaking).

**Locked artifact:** `data/prioritybench/manifests/w3_lock.json`  
**SHA256:** `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89`  
**Audit:** `docs/audit_w3.md` · W2d 145 IDs preserved · buried 20/80 for supersession + multi_turn (tool 0 — W2d filled quota)

### 3.2 Construction rules — status

| Rule | Status |
|---|---|
| Templates × lexical/filler variation | **Done** (tool + supersession v2 + multi_turn v2) |
| Strata 8K / 16K / 32K | **Done** in lock |
| Splits 40/20/40 via stable hash | **Done** (`assign_split`) |
| Dedup / 8-gram | Partial — ID uniqueness enforced; deepen if needed |
| Manual audit 15% dual | **Open** |
| Scorer unit tests ≥3/cat | **Partial** — tests exist under `tests/` / scripts; keep expanding |

Rebuild on H200 (JSONL gitignored):

```bash
uv run python scripts/mk_bench.py --mode w3_lock
uv run python scripts/audit_bench.py
```

### 3.3 Failure atlas (W3–4) — refined schedule

**Original:** FullKV / FP8 / uniform INT4 / SnapKV @ 50%/30% + ~600 page-perturb pairs with KL.

**Updated:**

| Piece | When | Notes |
|---|---|---|
| FullKV vs FP8 on PriorityBench | **Done** ≤16k (δ≈0) | Not informative alone |
| DropKeep / matched keep_frac structure | **Done** pilots | Primary early atlas signal |
| Page-level structure | **Done** `w3_structured_paged_r1` (structure **0.643**) | Keep densifying |
| Real uniform INT4 (Q2) | **Blocking** | `configs/w3_int4_assert.yaml` · see handoff |
| SnapKV @ matched bytes | ≤4-day W3 attempt else keep DropKeep | Import OK · `run_snapkv_quality.py` enqueued |
| Page-perturb labels (~score_delta OK; KL later) | **W4** | Fable cut from W3 finish package |
| Guardrails RULER/SCBench | **Before W4 G2 close** | Stub must become real |

**Gate G2 (end W4)** — unchanged logically:

Proceed with PriorityKV only if **(a)** uniform compression ≥5pt drop in ≥1 category while guardrails move <1pt, **OR (b)** oracle/structure-aware allocation beats uniform by ≥3pt at equal bytes.

- **(b)** **CLOSED** — token structure 1.0 vs uniform 0; page 0.643 vs 0 at keep_frac 0.15/0.25/0.35; guardrails Δ=0.  
- **(a)** still needs **working Q2**.

Pivot rule unchanged: measurement paper + static hot/cold is still a valid RE artifact.

---

## 4. Workstream B — Mixed-precision serving backend

### 4.1 Components — status

| Component | Plan weeks | Status |
|---|---|---|
| Byte model & accounting | W1 | **Done early** (`byte_model`, reports) |
| Page manager + structural tagging | W2–3 | **Substantial** — roles, protected invariants, keep policies (token + page) |
| INT4 append/decode | W3–4 | **CPU ref + tests** · HF quanto on H200 **JIT-blocked** (`quanto_cuda`) |
| Multi-call attention + LSE merge | W4–5 | **CPU LSE done** · FlashInfer CUDA → W5–6 |
| Fused decode kernel | W6–7 cond. | **Not started** |

### 4.2 Correctness suite

**Partial (CPU):** all-BF16 / all-INT4 / mixed / empty / partial page tests against dequant-then-attend.  
**Missing:** FlashInfer parity, GQA mapping, CUDA-graph, batch sweeps, CI on small model.

### 4.3 Systems measurement protocol

Unchanged as *target*. Not yet run. Agent-trace replay should use W3-lock sessions once Q2/atlas allow honest baselines.

**Gate G4 (end W7):** unchanged.

---

## 5. Baselines — as frozen at G1 (with deferrals)

| ID | Method | Role | Status |
|---|---|---|---|
| S0 | FullKV BF16 (vLLM) | Ceiling | **🟢 Frozen** |
| S1 | Calibrated vLLM FP8 | Deployment baseline | **🟢 Frozen** (δ≈0 on PB ≤16k — cite, don't overclaim stress) |
| **Q_dropkeep** | Prompt-level sink+recent | Interim eviction (plan Q3 stand-in) | **🟢 In use** for stress / RoPE-safe |
| Q2 | Uniform INT4 (quanto / KIVI-style) | Low-bit quality ref | **🔴 Blocking** — assert-no-fake; `quanto_cuda` JIT fail on H200 |
| Q3 | SnapKV @ matched bytes | Eviction baseline | **🟡** import OK · matched-byte quality job queued (`w4_snapkv_quality_r1`) |
| Q6 | FixedHot | Static hot/cold | **⬜** |
| Q7 | ProtectedRole (no risk score) | Critical ablation | **🟡 Early** via structure keep policies (not yet full P2 stack) |
| Q8 | Random @ matched bytes | Sanity | **🟢** in structure stress |
| P2 | PriorityKV (structure + linear risk) | Proposed | **⬜** risk not fitted |

Primary comparisons at paper time still: P2 vs S1, P2 vs Q3 (or DropKeep), P2 vs Q7 on locked test.

---

## 6. Week-by-week plan — original targets + reality overlay

Legend: ✅ done · 🚧 in progress / partial · ⏸ deferred with note · ⬜ not started

**W0 (setup).** ✅ Repo, uv lock, dual-machine workflow, Qwen3-8B pin, smoke CPU. 🚧 Gemma license + live CFP still open. ⬜ First outreach.

**W1.** ✅ Template engine + pilot examples + scorers + byte model + FullKV/FP8 path. (Executed compressed into mid-July ahead of formal calendar W0.)

**W2.** ✅ ~145→expanded pilots (w2d non-leak) · FullKV/FP8 · DropKeep kill · structure matched-keep HIT · buried adversarial · **G1 freeze with INT4/SnapKV/guardrails deferred in writing.**  
⏸ RULER/SCBench harness real runs.  
≠ Original “INT4/SnapKV reproduce within tolerance” — **explicitly not met**; substituted Q_dropkeep + written deferral.

**W3.** ✅ Locked 240 + audit SHA · 15% dual audit · INT4 CPU path + mixed ref · page-level structure stress (0.643) · assert-no-fake · **Q2 H200 GREEN** (`hf_cache_implementation_quantized`) · baselines check.  
⏸ FlashInfer CUDA (CPU LSE parity ✅).  
✅ SnapKV day-count attempt scripted (`run_snapkv_attempt.py`) → DropKeep lock if import fails.

**W4.** ✅ Denser atlas 0.15+0.35 · page-perturb + linear risk fit · guardrails PASS · CPU LSE multicall · **G2 CLOSED path (b)** · FlashInfer CUDA deferred · SnapKV quality job queued. Interview-prep track = process (non-code).

**W5–W6.** As original (allocators Q6/Q7/Q8/P2, ablations, fused go/no-go) — **G3**.

**W7.** Pilot 15% IDs · `FINAL_RUN_MANIFEST.yaml` · **G4**.

**W8–W10.** Locked quality/systems · Gemma reduced · paper/blog/PR/outreach — as original.

**W11–12.** Buffer — unfinished D1–D8 first; never new scope.

---

## 7. Google/DeepMind-specific hooks (unchanged intent)

1. **Gemma finding** — still planned (reduced matrix).  
2. **TurboQuant engagement** — still planned (W5 email with F1).  
3. **Pallas appendix** — buffer only.

---

## 8. Compute & storage envelope

Original ~540 H200 GPU-h cap still the ceiling.  
**Spent so far (order-of-magnitude):** W1–W3 pilots (FP8, DropKeep sweeps, structure ×2, buried, page stress) — tens of GPU-h, not hundreds. Reserve remains for Q2 debug, atlas, locked W8–9.

Storage paths on H200: `$PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv` · clone often `/data/anupam/scratch/Priority_KV`.

---

## 9. Risks & pivots — updated signals

| Risk | Signal | Response | Live signal (2026-07-15) |
|---|---|---|---|
| G2 fails | Flat PB deltas W4 | Measurement-paper pivot | Path (b) looking **good** on pilots; don't declare G2 early |
| SnapKV won't reproduce | >4 days | Substitute StreamingLLM/DropKeep; document | **Already substituted** interim; attempt still open |
| INT4 path won't run on box | quanto JIT / CUDA mismatch | Document platform blocker; CPU ref + own kernels; **never silent fake** | **Active** — see `HANDOFF_W3_INT4.md` |
| Fake/missed INT4 looks perfect | modes=`fake_*` or config kw bugs | Assert-no-fake; log `int4_modes_seen` | Learned in W2; gated in W3 |
| Q7 == P2 | W6 val | Ship Q7; risk = negative result | Too early |
| Multi-call overhead >12% | W5 profile | Fused kernel / claim cap | N/A yet |
| Concurrent paper | Weekly arXiv | Cite; sharpen lifecycle+serving | Ongoing |
| CFP earlier than results | W1 check | Atlas-only then full system | **CFP pick still open** |
| Compute overrun | Extrapolation | Cut Gemma/64K first | Watch after Q2 |
| Env drift via pip | torch/vLLM skew | **uv only** | Incident already happened — documented |

**Kill rule (unchanged):** no branch lives >1 week without a correctness, quality-frontier, systems, or falsification result.

---

## 10. Definition of done (unchanged targets)

- [ ] One-command smoke test passes on a clean H100/H200
- [ ] All correctness tests green; controller <2% latency; unexplained bytes <5%
- [ ] Figure F1 (flat accuracy vs collapsing agent reliability) reproduced from frozen manifest
- [ ] P2 (or Q7 per G3) beats S1 and Q3/DropKeep on ≥2 of 3 PriorityBench-A categories at 30% bytes, corrected CIs excluding zero
- [ ] Full H200 latency–throughput frontier + Nsight roofline published
- [ ] Workshop paper submitted; blog post live; upstream PR open with maintainer engagement
- [ ] Gemma generalization figure included
- [ ] ≥6 substantive outreach contacts logged; ≥2 referral conversations requested
- [ ] Negative results and limitations section written with the same care as the wins

### Partial progress toward Done (checklist helper)

- [x] PriorityBench-A locked 240 + SHA256 audit artifact  
- [x] Reproducible structure > uniform signal at matched keep (token + page)  
- [x] Buried-state scope check  
- [x] CPU mixed BF16/INT4 attend reference + tests  
- [x] Real Q2 INT4 on H200 (`int4_modes_seen` ∈ {`hf_cache_implementation_quantized`,`quanto_quantized_cache`})  
- [x] Guardrails real (<1pt move) — H200 `guardrails_w4_r2` gate Δ=0.0  
- [x] G2 formally closed in `docs/decisions.md` (path b)  
- [x] FlashInfer multi-call == mixed reference (**CPU LSE**; CUDA optional)  
- [x] Linear risk calibrated (seed fit from page-perturb labels)  

---

## 11. Immediate next actions (execution queue)

1. **H200:** finish Q2 per [`HANDOFF_W3_INT4.md`](HANDOFF_W3_INT4.md) §B (uv sync, CUDA major gate, tee'd `w3_int4_assert`).  
2. Log INT4 outcome in `docs/decisions.md` (green modes or platform blocker).  
3. Real guardrails stub → runnable before treating G2 as closed.  
4. W4: denser atlas + page-perturb pilot + linear risk inputs; begin LSE/multi-call only after Q2 story is honest.  
5. Pick D6 CFP; note Gemma license in decisions.
