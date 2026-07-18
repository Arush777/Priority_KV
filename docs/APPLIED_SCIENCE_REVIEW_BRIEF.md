# Applied-science review brief — PriorityKV (middle-ground close)

**Audience:** external LLM / applied scientist judging seriousness of results.  
**Freeze:** `FINAL_RUN_MANIFEST.yaml` · `G4_MIDDLE_GROUND_2026_07_19`  
**Repo:** `github.com:Arush777/Priority_KV` · model Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`  
**Hardware:** NVIDIA H200 (`dgre2`)

---

## 1. Point of the project (one paragraph)

Long agent traces (tools, superseding instructions, multi-turn IDs) make the **KV cache** dominate memory. Serving stacks compress or **evict** KV. **PriorityKV** asks whether *what* you drop or demote matters more than *how much*: uniform eviction can look fine on average metrics while destroying agent-critical state. The project builds (1) a **PriorityBench** of agent failure modes, (2) evidence that **structure-aware keep** beats uniform keep at matched budgets, and (3) a **structure-protected mixed BF16/INT4 packed cache** with FlashInfer-backed decode, evaluated for **honest bytes + latency** on H200 — *not* for a fake soft-INT4 accuracy gap (that hypothesis was falsified).

---

## 2. What we are / are not claiming

**Claim (allowed):**
- Uniform **eviction / missing-state** hurts tool-schema / supersession / multi-turn reliability; structure-aware retention preserves it at matched keep budgets.
- Soft INT4 at `int4_frac=0.75` does **not** open a PriorityBench quality gap vs FullKV.
- Systems value = **packed payload bytes** + **honest TTFT/TPOT** (e2e≈FullKV, TPOT ~1.2× FI shim) with quality matched to FullKV on lock-240, plus measured peak/payload (**cold-scratch caveat**: peak VRAM can stay near FullKV).

**Do not claim:**
- Uniform INT4 quantization alone collapses PriorityBench quality at this operating point.
- Peak CUDA memory ≪ FullKV without the scratch caveat.
- Paper-grade LongBench/RULER coverage (out of middle-ground scope; not publishing).

---

## 3. Progress so far (middle-ground definition of done)

| Stage | Status | Evidence |
|---|---|---|
| PriorityBench-A lock (240, 3 cats) | Done | `w3_lock.json` SHA `fc44b966…ae89` |
| Matched-keep reliability (family A) | Done | structure ≫ uniform at matched keep; zero-degrade wiring proof |
| Soft-INT4 quality gap | **Falsified** | corrected mixed @ 0.75 → both policies ~FullKV |
| Packed BF16/INT4 + FI shim decode | Done | pack/cold batched; M3c latency |
| D4 latency (8k∥16k) | **PASS** | job `d4_latency_m3c_gpu56_r1` |
| Peak mem + payload | **PASS** | job `mg_a_peak_mem_gpu5_r1` |
| Lock-240 Full/uniform/structure @ 0.75 | **PASS** | job `mg_b_lock240_quality_gpu01_r1` |
| G4 freeze / final manifest | **Done** (this close) | `FINAL_RUN_MANIFEST.yaml` |
| Publish / LongBench / Gemma | Out of scope | — |

**Overall middle-ground: ~100% of chosen scope** (optional thin guardrail re-run deferred; prior W4 Δ=0).

---

## 4. Canonical quantitative results (cite these)

### A. Latency — `d4_latency_m3c_gpu56_r1` (`D4_M3_PASS`)
- Structure vs FullKV: e2e ~**1.11–1.12×**, TPOT ~**1.20–1.21×**, pack ~35–48 ms, cold ~14–20 ms.
- Scores matched FullKV on the latency slice.

### B. Peak / payload — `mg_a_peak_mem_gpu5_r1` (`MG_PEAK_MEM_PASS`)
- Structure decode peak ~**0.87×** FullKV.
- Measured payload ~**0.72×** BF16; modeled bit-pack ~**0.47×**.
- Caveat: FI cold scratch expands INT4→BF16 for attend.

### C. Lock-240 quality — `mg_b_lock240_quality_gpu01_r1` (`MG_LOCK240_PASS`, ~23 min, GPUs 0∥1)
| Arm | mean (n=240) | Δ vs FullKV |
|---|---|---|
| FullKV | **0.888** | — |
| uniform INT4 @ 0.75 | **0.879** | −0.008 |
| structure mixed @ 0.75 | **0.883** | −0.004 |

By context: **8k/16k all 1.0**; **32k** drops for everyone (full 0.645 / uniform 0.618 / structure 0.632) — structure slightly above uniform, both near FullKV. Hard category at 32k is multi-turn (full 0.16).

---

## 5. What we are doing now

**Closing the middle-ground project:** freeze claim + repro IDs in `FINAL_RUN_MANIFEST.yaml` / `docs/decisions.md` (G4). No further required GPU jobs. Not running LongBench or a paper track unless scope is explicitly reopened.

---

## 6. Judge rubric (suggested)

Score as **applied systems + measurement science**, not as a venue paper draft.

1. **Hypothesis clarity:** eviction reliability vs soft-INT4 quality — were they separated?
2. **Falsification honesty:** was the soft-INT4 quality gap retired cleanly?
3. **Measurement integrity:** latency phases (pack/cold/decode), score-prefix under fixed-length decode, peak vs payload reported separately?
4. **Evidence strength:** lock-240 + M3c + peak-mem enough for a *serious internal result*; not enough alone for a LongBench-style paper claim.
5. **Reproducibility:** pinned model rev, bench SHA, job IDs, configs in the freeze manifest?

**Suggested verdict bands:**  
- **Strong applied close** if (1)–(3) and (5) hold.  
- **Partial** if systems numbers exist but claim wording still implies INT4 quality win.  
- **Weak** if only pilots, no lock-240 / no honest latency.
