# PriorityKV-Agent — Implementation Plan (v2, DeepMind-targeted)

**Supersedes:** PRIORITYKV_RESEARCH_PLAN.md (v1)
**Team:** 2 students (A = research/eval lead, B = systems/kernel lead)
**Duration:** 10 weeks core + 2-week buffer, starting Week 0 = Mon 2026-07-20
**Primary hardware:** 2× H200 · **Secondary:** 1× H100 or A100 (validation only)
**Primary model:** Qwen3-8B · **Secondary model:** Gemma (latest 7–12B instruct variant; verify current release at kickoff)
**Positioning:** Research Engineer application artifact — systems-first, motivated by an agent-reliability finding

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

---

## 1. Locked scope decisions (change requires both students + written note)

1. **Models:** Qwen3-8B primary; Gemma secondary (reduced matrix only). One 0.5–1.7B model for unit tests, never for evidence.
2. **Storage classes:** BF16 and INT4 (group-wise asymmetric, KIVI-style, group size 32 along the token axis for K, channel for V — confirm against KVPress reference in Week 1). No FP8 storage, no 2/3-bit, no eviction in the shipped policy (eviction appears only as a baseline).
3. **Page layout:** backend-native 16-token physical pages; 128-token allocation units (one ablation at 64).
4. **Policy:** ProtectedRole++ — deterministic structural rules (protect system/tool/constraint/sink/recent-window pages in BF16, rest INT4) plus a calibrated **linear** risk score used only to break ties when the byte budget forces demoting protected pages. No MLP, no trees in the shipped system.
5. **Budgets:** 50% and 30% of FullKV bytes (30% not 25%: BF16-protected pages plus INT4 metadata make 25% unreachable without eviction; verify with the byte model in Week 1 and adjust once, before freeze).
6. **Backend:** multi-call homogeneous paged attention (FlashInfer) + exact log-sum-exp merge. Fused Triton decode kernel only if merge/launch overhead > 12% at 32K, and only for decode, batch 1 and 8.
7. **Serving comparison target:** calibrated vLLM FP8 (per-head scales via llm-compressor). Beating it at *equal quality on agent workloads* at the 30% budget is the systems win condition. At 50% we expect parity-ish throughput with better reliability — frame accordingly.

---

## 2. Deliverables (all core, none stretch)

| # | Deliverable | Owner | Due |
|---|---|---|---|
| D1 | PriorityBench-A: 240 programmatically scored agent-reliability examples + generator + audit log | A | End W3 |
| D2 | Failure-atlas tech note + 3 headline figures (the "flat accuracy, collapsing schema conformance" figure is F1) | A | End W4 |
| D3 | Mixed-precision paged cache backend: page manager, INT4 append/decode, LSE merge, full correctness suite | B | End W6 |
| D4 | Optimized H200 results: TTFT/TPOT/throughput/concurrency frontier + Nsight roofline analysis | B | End W9 |
| D5 | Upstream PR (target order: KVPress method PR → FlashInfer example/kernel PR → vLLM RFC), opened by W8, iterated to merge | B (A reviews) | Open W8 |
| D6 | 6-page workshop paper (verify live CFPs in W1: NeurIPS 2026 efficiency/agents workshops if deadlines permit, else EuroMLSys / ICLR 2027 workshop cycle) | A (B: systems section) | Draft W9, submit per CFP |
| D7 | Public blog post: failure atlas + system, 5 figures, reproduction script | A | W10 |
| D8 | Repro artifact: one-command smoke test + full-run script on a single H100/H200 | Both | W10 |
| D9 | Outreach log: ≥6 substantive contacts (paper authors, Gemma/FlashInfer/vLLM maintainers) | Both | Continuous, reviewed W5/W8/W10 |
| D10 | (Buffer only) Pallas decode-kernel port + 1-page TPU memory-hierarchy mapping appendix | B | W11–12 if D1–D8 done |

---

## 3. Workstream A — PriorityBench-A (agent reliability)

### 3.1 Categories (80 examples each, 240 total)

1. **Tool-schema conformance under long context.** A tool schema (JSON, 3–8 tools, nested params) defined early; 8–32K of interleaved turns/tool results; final task requires an exactly valid call. Score: JSON-schema validation + required-field/enum checks.
2. **Instruction supersession.** Constraint issued, later explicitly updated or revoked mid-conversation; model must follow the *latest* version. Score: deterministic constraint checkers on the output (regex/parser per template, no LLM judge).
3. **Multi-turn state persistence.** Facts/IDs established in early turns must be reused verbatim 10–30 turns later (order IDs, file paths, user preferences). Score: exact-match slot extraction.

### 3.2 Construction rules
- 12–15 templates per category × held-out lexical/filler variations; filler sampled independently of the target span so page-position confounds are controlled.
- Context-length strata: 8K / 16K / 32K per category.
- Splits: 40% calibration, 20% validation, 40% locked test. Dedup across splits by normalized template hash + 8-gram overlap check.
- Manual audit: 15% of examples, both students independently, disagreements resolved in `docs/decisions.md`.
- Publish: templates, seeds, generator, scorer unit tests (≥3 per category, including adversarial near-miss outputs).

### 3.3 Failure atlas (W3–4)
- Run FullKV, calibrated FP8, uniform INT4, SnapKV at 50%/30% bytes on calibration+validation splits (Qwen3-8B).
- Page-perturbation study: sample ~600 (page, demotion) pairs across page classes × layer quartiles × positions; label with teacher-forced KL + attention-output error + outcome flip. This calibrates the linear risk score and produces figure F5 (predicted vs measured risk).
- **Gate G2 (end W4):** proceed with PriorityKV only if (a) uniform compression produces a ≥5-point absolute drop in at least one PriorityBench-A category while guardrail accuracy moves <1 point, OR (b) oracle structure-aware allocation beats uniform INT4 by ≥3 points at equal bytes. Otherwise pivot: the deliverable becomes the benchmark + failure atlas + backend with a *static* hot/cold policy, and the paper becomes a measurement paper. (This pivot is still a strong RE artifact — do not treat it as failure.)

---

## 4. Workstream B — Mixed-precision serving backend

### 4.1 Components
1. **Byte model & accounting** (W1): exact formula for bytes/page incl. INT4 scales/zero-points, page table, controller metadata; unit tests for partial pages. All budgets defined in *realized* bytes.
2. **Page manager** (W2–3): allocation units, structural tagging from chat-template spans, demotion of aged generated pages every 128 tokens, protected invariants (sinks never demoted below BF16, newest W tokens BF16, budget never exceeded).
3. **INT4 path** (W3–4): append (quantize on write) and decode (dequant fused into gather where FlashInfer permits; else explicit dequant kernel in Triton). Verify against KIVI/KVPress reference within tolerance.
4. **Multi-call attention + LSE merge** (W4–5): partition page table by dtype; homogeneous FlashInfer calls; exact merge `O = Σ_g exp(LSE_g − LSE)·O_g`. Reuse index/workspace buffers; CUDA-graph capture for stable shapes.
5. **Fused decode kernel** (W6–7, conditional): only if profiling shows >12% overhead from launches/partitioning at 32K, batch 8.

### 4.2 Correctness suite (blocking; runs in CI on the small model)
- all-BF16 == dense paged attention (bitwise where kernels permit, else tol 1e-3 rel);
- all-INT4 == homogeneous INT4 reference;
- mixed == dequantize-then-attend reference;
- empty dtype group; single page; partial last page; page-boundary context lengths; GQA head mapping; batch 1/8/32; large-logit numerical stress; CUDA-graph replay.

### 4.3 Systems measurement protocol
- Workload: agent-trace replay built from PriorityBench-A sessions (long input, multi-turn, 256 and 2,048-token outputs) — *not* synthetic uniform prompts.
- Configs: contexts 8K/32K/64K × concurrency 1/8/32 (+ sweep to OOM or p95 TPOT > 2× concurrency-1); 5 timed reps after warmup; randomized order; clocks pinned; contaminated runs rejected by predeclared utilization rule.
- Report full latency–throughput frontier, controller overhead (<2% budget), metadata overhead (<5% unexplained bytes), Nsight kernel breakdown, roofline placement of the INT4 decode path.
- **Gate G4 (end W7):** backend passes all correctness tests AND beats FullKV decode at ≥32K AND shows ≥1.15× throughput or a clear concurrency gain vs FullKV. If it cannot beat calibrated FP8 at the 30% budget at equal quality, the claim becomes "reliability at parity cost" — decided here, in writing, before final runs.

---

## 5. Baselines (reduced, frozen end of W2)

| ID | Method | Role |
|---|---|---|
| S0/Q0 | FullKV BF16 (vLLM) | Ceiling |
| S1/Q1 | Calibrated vLLM FP8 (per-head scales) | Deployment baseline to beat |
| Q2 | Uniform INT4 (KIVI-style via KVPress) | Low-bit quality reference |
| Q3 | SnapKV @ matched bytes | The one eviction baseline (drop H2O/StreamingLLM/MiKV unless SnapKV fails to reproduce in ≤4 days; then substitute, don't add) |
| Q6 | FixedHot (recent BF16, old INT4) | Static hot/cold control |
| Q7 | ProtectedRole (structure rules, no risk score) | The critical ablation — is the linear risk score worth anything? |
| Q8 | Random allocation @ matched bytes | Sanity |
| P2 | PriorityKV (structure + linear risk) | Proposed |

Primary statistical comparisons (Holm–Bonferroni corrected, paired bootstrap ×10,000, margins pre-registered at W7 freeze): P2 vs Q1, P2 vs Q3, P2 vs Q7 on PriorityBench-A locked test; optimized P2 vs S1 end-to-end. Everything else exploratory.

---

## 6. Week-by-week plan

**W0 (setup, part-time).** Repo, CI, env lock, issue tracker. Pin all versions. Verify current Gemma release + license; verify live workshop CFPs and set D6 target. First outreach touch: introduce PriorityBench-A idea in KVPress/FlashInfer discussion threads.

**W1.** A: template engine + first 40 PriorityBench-A examples + scorer tests. B: byte model + accounting tests; FullKV in vLLM validated vs Transformers on 20 prompts; calibrated FP8 up; first profiler traces at 8K/32K. **G0:** manifests reproduce runs; FullKV stable.

**W2.** A: finish calibration+validation splits (~145 ex); run FullKV/FP8/INT4 pilot on them; RULER 2-task + SCBench 2-task guardrail harness. B: page manager + structural tagging + protected invariants; SnapKV reproduction started. **G1:** FP8/INT4/SnapKV reproduce documented numbers within tolerance; realized bytes measured. Freeze baseline list.

**W3.** A: locked test split done; audit; page-perturbation labeling begins. B: INT4 append/decode verified vs reference; begin multi-call attention.

**W4.** A: failure atlas complete; figures F1/F5 drafted; fit + calibrate linear risk score. B: LSE merge working end-to-end; correctness suite green on small model. **G2 (§3.3).** Publish failure-atlas tech note (early visibility; also stakes the claim against concurrent work). Begin interview-prep track (6 h/wk each: LeetCode-style + ML fundamentals + one mock/wk from W7).

**W5.** A: implement ProtectedRole, FixedHot, Random, P2 allocators in KVPress reference path; validation sweep at 50/30%. B: backend integration of P2 policy; overhead measurement; fused-kernel go/no-go data. Outreach: email TurboQuant authors + 2 closest-paper authors with F1 attached.

**W6.** A: mandatory ablations (structure vs risk vs combined; shuffled roles; no recent-window; 64 vs 128 unit). B: fused kernel if triggered, else optimization polish (buffer reuse, CUDA graphs). **G3:** P2 Pareto-beats ≥2 strong baselines incl. Q7 on validation — if Q7 ties P2, ship Q7 as the system and reframe the risk score as a negative result (say so plainly in the paper).

**W7.** Pilot full pipeline on 15% of final IDs. Freeze: commit, revisions, sample IDs, budgets, seeds, margins, analysis notebook → signed `FINAL_RUN_MANIFEST.yaml`. **G4 (§4.3).**

**W8.** Locked quality runs, Qwen3-8B: PriorityBench-A locked test + guardrails (RULER 2×3 lengths×100, SCBench 2×50, MATH-500 greedy) × {S0,S1,Q2,Q3,Q6,Q7,Q8,P2} × {50%,30%}. Open D5 upstream PR with the verified reference method.

**W9.** Locked systems runs (H200 matrix §4.3; H100/A100 reduced: 32K/64K × conc 1/16). Gemma reduced matrix: FullKV, S1, Q2, P2@50/30 on PriorityBench-A locked + RULER 8K/32K. Paper draft complete.

**W10.** Frozen analysis notebook executed untouched; 50-output audit + every failure category inspected; paper finalized per CFP; blog post + repro artifact published; outreach round 3 (share paper + PR with all prior contacts; ask two for referral conversations, not jobs).

**W11–12 (buffer).** Priority order: unfinished D1–D8 → PR review iteration → Pallas decode port + TPU appendix (D10) → extra ablations. Never new scope.

---

## 7. Google/DeepMind-specific hooks (do all three, they're cheap)

1. **Gemma finding:** one figure dedicated to Gemma's PriorityBench-A behavior under compression, replicated or contrasted with Qwen. If Gemma shows a distinct fragility pattern, that's the email subject line to Gemma team contacts.
2. **TurboQuant engagement:** cite it as the quantization-quality frontier, position PriorityKV as orthogonal (allocation, not quantizer), and send authors the failure-atlas note in W5 with one concrete question.
3. **Pallas appendix (buffer):** port the decode path's inner loop to Pallas on CPU-simulated or Colab TPU; write one page mapping page layout → TPU memory hierarchy (HBM/VMEM tiling, why 128-token units align with lane width). Signals stack fluency without claiming TPU performance numbers.

---

## 8. Compute & storage envelope (revise once after W2 measurements)

- W1–2 env/baselines: 40 H200 GPU-h · W3–4 atlas/labels: 70 · W5–6 policy+backend: 90 · W7 pilot: 30 · W8–9 locked runs: 220 · buffer 20% → **~540 H200 GPU-h cap.**
- Storage: models <50GB; datasets/frozen IDs <100GB; generations + sampled logits <500GB (top-k logits only, full logits solely for KL calibration positions); traces <150GB; artifacts <20GB. Secondary machine: stage model to scratch, delete after runs (30GB persistent limit).

---

## 9. Risks & pivots

| Risk | Signal | Response |
|---|---|---|
| G2 fails (no reliability gap) | Flat PriorityBench-A deltas W4 | Measurement-paper pivot (§3.3); backend ships with static policy |
| SnapKV won't reproduce | >4 days off documented numbers | Substitute StreamingLLM; document attempt; never run baseline-less |
| INT4 decode too slow | Dequant dominates ≥32K | Restrict INT4 to K or to old pages only; report honestly; FP8-storage fallback only if backend-native and <2 days |
| Q7 (ProtectedRole) == P2 | W6 validation | Ship Q7; risk score becomes reported negative result — still publishable |
| Multi-call overhead >12% | W5 profiling | Trigger fused kernel; if fused kernel slips past W7, cap systems claim at 50% budget where 2 dtype groups suffice |
| Concurrent paper lands | Weekly arXiv audit (owner alternates) | Cite immediately; sharpen to agent-trace lifecycle + serving implementation, which concurrent quality papers rarely have |
| CFP deadline before results | W1 CFP check | Submit failure atlas alone to earlier venue; full system to next cycle |
| Compute overrun | W2 pilot extrapolation | Cut Gemma matrix and 64K systems points first; never cut Q1/Q3/Q7 or locked-test size |

**Kill rule (unchanged from v1):** no branch lives >1 week without a correctness, quality-frontier, systems, or falsification result.

---

## 10. Definition of done

- [ ] One-command smoke test passes on a clean H100/H200
- [ ] All correctness tests green; controller <2% latency; unexplained bytes <5%
- [ ] Figure F1 (flat accuracy vs collapsing agent reliability) reproduced from frozen manifest
- [ ] P2 (or Q7 per G3) beats S1 and Q3 on ≥2 of 3 PriorityBench-A categories at 30% bytes, corrected CIs excluding zero
- [ ] Full H200 latency–throughput frontier + Nsight roofline published
- [ ] Workshop paper submitted; blog post live; upstream PR open with maintainer engagement
- [ ] Gemma generalization figure included
- [ ] ≥6 substantive outreach contacts logged; ≥2 referral conversations requested
- [ ] Negative results and limitations section written with the same care as the wins
-
