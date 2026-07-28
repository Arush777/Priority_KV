# Pre-registration — BFCL tight-budget test of ADAPT

**Status: written and committed BEFORE any tight-budget conversation is scored.**
Namespace: `EXTERNAL_BFCL_TIGHT_V1`. Does not modify `EXTERNAL_BFCL_PRAJNA_V1`
or anything under `FINAL_RUN_MANIFEST.yaml`.

This document exists because the original ADAPT result carries a documented
pre-registration defect: the formula was written before the run but committed
after the results were seen, so the agreement between predicted and measured
alpha could not be treated as confirmatory. This test is specified in advance so
its outcome — in either direction — is interpretable.

## 1. What prompted this test

`SWEEP_PM_V3` (synthetic, Qwen3-8B and Llama-3.1-8B, 15,360 generations) found:

- A retention boundary at oversubscription ratio `protected_mass / keep_budget`
  approximately 1. Structure is at ceiling below it and collapses above it.
- ADAPT held 0.95–1.00 in every cell of the grid.
- ADAPT **significantly exceeded SnapKV** on Qwen at the tightest budget
  (`keep_frac = 0.02`) in schema-dense contexts: 1.000 vs 0.900 (`p = 0.031`),
  1.000 vs 0.867 (`p = 0.0078`, twice), and 0.950 vs 0.750 (`p = 0.00049`),
  with `c = 0` in every cell (ADAPT never lost a discordant pair).

Crucially, that advantage appeared **only where SnapKV was degraded but not
floored** (SnapKV between 0.75 and 0.90). Where SnapKV was at 1.000 there was no
room to improve on it. The existing external BFCL evaluation was run at
`keep_frac = 0.25`, where ADAPT and SnapKV tied (0.129 vs 0.136, `p = 1.0`).
**The external evaluation has therefore never tested the regime in which the
method is predicted to help.** That is the gap this test closes.

## 2. Hypotheses

**H1 (primary).** On BFCL V3 multi-turn, at a keep budget where SnapKV is
degraded but not floored, ADAPT achieves higher conversation-level accuracy than
SnapKV under the unmodified official `multi_turn_checker`.

**H2 (secondary, boundary replication).** At the same budget, the hard
`structure` arm scores at or near zero, because BFCL's protected mass (98.8%)
oversubscribes any tight budget by more than an order of magnitude.

**H3 (diagnostic validity).** The measured alpha equals
`min(1, keep_budget / protected_mass)` computed from the prompt alone, with no
value fitted to any outcome.

## 3. Budget selection — decided by pilot, rule fixed in advance

FullKV on BFCL is budget-independent (no eviction), so the existing 140 FullKV
conversation outcomes are reused as the reference ceiling rather than recomputed.

A pilot runs **SnapKV only** at `keep_frac` in {0.10, 0.05, 0.02} on the same 40
externally-sampled conversations.

**Pre-specified selection rule.** Choose the *tightest* budget whose SnapKV
accuracy falls in the band `[0.25, 0.70] x FullKV accuracy on the same
conversations`. This is the "degraded but not floored" regime the synthetic
sweep identified.

- If **no** budget lands in the band because all are below it, the workload
  floors before it discriminates. H1 is then **not testable on BFCL**, and that
  is reported as the result — no budget is chosen post hoc to manufacture a
  contrast.
- If **no** budget lands in the band because all are above it, the ladder is
  extended downward to 0.01 once, and the same rule is reapplied.

### Amendment 1 (2026-07-28, before any ADAPT run at any budget)

The original ladder {0.10, 0.05, 0.02} returned SnapKV/FullKV ratios of 0.11,
0.00 and 0.00 — every rung *below* the band. Section 3 as written covers only
the mirror case (all rungs above the band) and therefore under-specifies this
one.

The frozen `EXTERNAL_BFCL_PRAJNA_V1` run at `keep_frac = 0.25` gives SnapKV
19/140 against FullKV 27/140, a ratio of 0.70 — the band's upper edge. The
discriminative window is therefore *bracketed* between 0.10 and 0.25 rather than
absent, and the original ladder simply started too tight.

The ladder is extended **upward once** to {0.15, 0.20}, and the Section 3 rule is
reapplied unchanged. This is symmetric to the downward extension already
specified. The integrity condition is preserved: **ADAPT has not been run at any
budget**, so no budget can be selected to flatter it. If neither added rung lands
in the band, H1 is recorded as not testable on BFCL, per Section 5.

## 4. Primary analysis

- **Unit:** one BFCL multi-turn conversation, scored by the unmodified official
  `multi_turn_checker`. Pairing is on `task_id`.
- **Test:** exact two-sided McNemar, ADAPT vs SnapKV, `alpha = 0.05`.
- **Target n:** 300 conversations per arm (up from 150 in the frozen run), to
  roughly double the discordant-pair count available for the test.
- **Reported alongside:** paired accuracy difference with a 95% bootstrap CI,
  Wilson intervals per arm, paired completeness, and matched-budget violations.

### Minimum detectable effect

With `c = 0` (the pattern observed throughout the sweep, where ADAPT never lost a
discordant pair), exact two-sided McNemar reaches `p < 0.05` at `b >= 6`. At
n = 300 with a SnapKV base rate near 0.10, that corresponds to an absolute
accuracy gain of about **+0.02 or larger** being detectable.

If instead reversals occur at the rate implied by the frozen 25% run, roughly
`b + c ~= 20` discordant pairs at n = 300, then detecting significance requires
about a 70/30 split of those pairs — an absolute gain near **+0.05**.

**A null is therefore informative**: failing to reject at n = 300 bounds any true
ADAPT advantage on this workload to below roughly 0.05 absolute, which is a
reportable negative result rather than an underpowered shrug.

## 5. Interpretation fixed in advance

| Outcome | What we conclude | What goes in the paper |
|---|---|---|
| ADAPT > SnapKV, `p < 0.05` | The synthetic advantage transfers to an externally authored workload | Method claim, scoped to tight budgets in schema-dense contexts |
| No significant difference | No external benefit detected; upper bound on the effect reported | Bounded negative; ADAPT justified as matching attention at no additional retention cost |
| ADAPT < SnapKV, `p < 0.05` | The synthetic advantage does not transfer and is benchmark-specific | Reported as a negative that scopes the sweep's claim to synthetic traces |
| SnapKV floors at every budget | H1 is not testable on BFCL | Reported as a limitation of the workload, not of the method |

## 6. Stopping and integrity rules

- The budget is chosen by the Section 3 rule applied to pilot SnapKV numbers.
  ADAPT is **not** run at any budget before that choice is fixed.
- n is fixed at 300 in advance. No extension after inspecting the p-value.
- No arm, task category, or subset is dropped after seeing outcomes. Exclusions
  are limited to `MODEL_CONTEXT_LIMIT`, as in the frozen run, and are reported.
- Every arm is a `kvpress` press over an identical full prefill at an identical
  compression ratio, so arms differ only in which KV entries survive.
- All deviations forced by the cluster or the benchmark are recorded in the
  config's `deviations:` block, as in `EXTERNAL_BFCL_PRAJNA_V1`.

## 7. Known threat this test does not remove

The structural tagger shares regex markers with the PriorityBench generator, and
the sweep's distractor schemas were authored to be tagged `TOOL` by that same
tagger. BFCL is externally authored, so this test is not subject to that
circularity — which is precisely why it is the load-bearing evidence for any
method claim, and why the synthetic sweep alone cannot be.
