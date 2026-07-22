# When Does Structure-Aware KV Cache Retention Help? A Budget-Relative Boundary

**Status:** draft skeleton · numbers auto-populated from
`$PRAJNA_ROOT/results/external_bfcl_prajna_v1/summaries/`
**Target:** 4–8 page workshop paper (efficiency / negative-results / agents track)
**Freeze:** `EXTERNAL_BFCL_PRAJNA_V1`

---

## Abstract

*(≈150 words)*

KV-cache eviction is not uniformly harmful: dropping a filler token is free,
dropping the token holding an order ID or a "never refund without a certificate"
policy silently breaks an agent while average metrics look unchanged. This
motivates *structure-aware* retention — protect system prompts, tool schemas,
constraints, sinks and the recent window; evict the rest. On a purpose-built
synthetic agent benchmark this works dramatically (0.933 vs 0.008 at a 25%
budget). We test whether it transfers, using the official Berkeley Function
Calling Leaderboard V3 multi-turn scorer and 4,856 public τ-bench trajectories.
**It does not.** Structure-aware retention scores 0.000 on BFCL where SnapKV is
statistically indistinguishable from FullKV. We identify the cause as a single
measurable quantity — the *protected fraction* of the context — and show it
predicts the outcome across three workloads. We propose ADAPT, which treats
structure as a budget-relative prior rather than a hard constraint.

**Contributions.**
1. An external, officially scored evaluation of structure-aware KV retention.
2. The **protected-fraction criterion**: a CPU-computable diagnostic that
   predicts whether a structure-aware policy can help, before any GPU time.
3. **ADAPT**, a one-parameter policy that interpolates between structure and
   attention using only measurable quantities, subsuming both endpoints.
4. Corrections to three defects, one affecting a previously published baseline.

---

## 1. Introduction

- Long agent traces put schemas / constraints / identifiers into the KV cache.
- Serving stacks evict. Eviction damage is *heterogeneous*, not uniform.
- Structure-aware retention is the natural response, and it works on synthetic
  agent benchmarks.
- **Question:** does it transfer to benchmarks the authors did not construct?
- **Answer:** no — and the reason is measurable and predictive, not incidental.

## 2. Setup

**Policies.** All non-FullKV arms are `kvpress` presses over an *identical* full
prefill at an identical compression ratio, so arms differ only in which KV
entries survive. This matters: the repo's original path physically rewrote the
prompt, which is a strictly more destructive intervention and confounds policy
with mechanism (§6.1).

| Arm | Press |
|---|---|
| FullKV | none |
| Structure | role-scored `ScorerPress` |
| Uniform | `StreamingLLMPress` (sink + recent) |
| Random | seeded position-blind `ScorerPress` |
| SnapKV | `SnapKVPress` (real attention) |
| **ADAPT** | budget-relative blend (§5) |

**Benchmarks.** BFCL V3 multi-turn (Gorilla `cd9429cc`, unmodified official
`multi_turn_checker`, state + response comparison against stateful APIs);
τ-bench trajectories (`AgentSuite/tau-bench-trajectories` `382e57d1`).

**Model.** Qwen3-8B `b968826d`, bf16, 25% keep, L40S/sm_89.

**Statistics.** Unit = conversation. Exact paired McNemar + paired bootstrap CIs,
restricted to conversations scored in *every* arm.

## 3. Structure-aware retention does not transfer

**Table 1 — BFCL V3 multi-turn, n=141 paired.**

| Arm | Accuracy |
|---|---:|
| FullKV | 0.192 |
| SnapKV | 0.135 |
| Structure | 0.000 |
| Uniform | 0.000 |
| Random | 0.000 |
| ADAPT | *(pending)* |

| Comparison | McNemar | Δ | 95% CI |
|---|---:|---:|---|
| FullKV vs Structure | 1.5e-08 | +0.191 | [+0.128, +0.262] |
| FullKV vs SnapKV | 0.152 (n.s.) | +0.057 | [−0.007, +0.128] |
| Structure vs SnapKV | 3.8e-06 | −0.135 | [−0.191, −0.078] |

Two findings: attention-based eviction **preserves capability at 4× compression**
(CI spans zero vs FullKV), and structure-aware retention **fails outright**.

## 4. Why: the protected-fraction criterion

A role-based policy can only *rank* while protected mass is smaller than the keep
budget. Above that it selects everything and degenerates to index order.

**Table 2 / Figure 1 — measured at `keep_frac=0.25`.**

| Workload | Protected | Oversubscribed | Structure |
|---|---:|---:|---:|
| PriorityBench-A (synthetic) | 6.1% | 0% | 0.933 |
| τ-bench (real traces) | 79.5% | 99% | retention-only |
| BFCL (external) | 98.8% | 100% | 0.000 |

Role mix is decisive: PriorityBench-A is **94.9% filler**; a BFCL system prompt
*is* 32 JSON tool schemas, so ~98% of tokens carry the protected `TOOL` role.
The synthetic benchmark was not wrong — it measured one regime.

**Corroboration (τ-bench, 828k spans).** Structure retains explicit policy lines
at 0.820 vs uniform's 0.001 (~680×) but loses on reused identifiers
(0.055 vs 0.315): it wins exactly where protected spans are *scarce*.

## 5. ADAPT: structure as a budget-relative prior

The failure is a mis-specification, not a dead end. Prior work here force-protects
structural positions (a hard union), which swallows the whole budget when
oversubscribed — consistent with its recorded collapse at small budgets.

```
α = min(1, keep_budget / protected_mass)
score = α · rank(structure) + (1 − α) · rank(attention)
```

α uses only quantities known from the prompt: **no tuning, no fitting, no free
parameter.** Ranks are required because structure bands (~1e6) and attention
scores (~1e-3) are not commensurate. α = 1 provably reproduces the structure
arm's exact selection; α → 0 reduces to SnapKV.

Predicted: PriorityBench-A α=1.00, τ-bench α=0.31, BFCL α=0.25.

**Table 3 — ADAPT on BFCL.** *(pending)*

## 6. Threats to validity

1. **Mechanism confound (addressed).** Comparing prompt-deletion against KV
   eviction measures the intervention, not the policy. All arms are presses.
2. **Tagger artifact?** No — the tagger is correct; the workload genuinely is all
   tool schemas. Table 2 is the evidence.
3. **FullKV ceiling.** 16 conversations exceeded the 40,960-token context and were
   excluded, concentrated on arms that make working calls (their transcripts
   grow). This penalises the *strong* arms, so gaps are conservative.
4. **Scope.** One model, one budget, n=141. τ-bench is retention-only.

## 7. Related work

SnapKV, StreamingLLM/attention sinks, H2O, PyramidKV, kvpress; BFCL; τ-bench.
Position: prior work asks *which* policy is best on average; we ask *when* each
policy can work, and give a measurable criterion.

## 8. Conclusion

Structure-aware KV retention helps only when protected content is a minority of
context. In real tool-calling agents it is the majority, role priority carries no
signal, and attention-based selection dominates. The protected fraction is
cheap to measure and predicts which regime you are in.

---

## Appendix A — Reproduction

Pinned revisions, `uv.lock` hash, Slurm job IDs, per-conversation checkpoints,
exclusion and failure ledgers: `configs/external_bfcl_prajna_v1.yaml`, all
deviations recorded under `deviations:`.

## Appendix B — Defects found

1. `random` ≡ `uniform` in the frozen core (RNG branch unreachable) — affects a
   published baseline column.
2. Reasoning blocks passed whole to the decoder, zeroing correct tool calls
   (FullKV 0.000 → 0.105 once fixed).
3. Arms compared across two mechanisms.

## TODO before submission

- [ ] ADAPT numbers (Table 3)
- [ ] Llama-3.1-8B transfer — is the boundary model-independent?
- [ ] `keep_frac` sweep (0.10 / 0.25 / 0.50) — does the boundary move with budget?
- [ ] Manual precision audit of the τ span extractor (sample already generated)
- [ ] Re-run or explicitly correct the frozen PriorityBench-A `random` column
