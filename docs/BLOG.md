# What You Drop From an Agent's KV Cache Matters

*Arush Sharma and Anupam Rawart*

Long LLM conversations are not homogeneous text. A tool-using agent trace mixes system
instructions, tool schemas, superseding user constraints, persistent identifiers, tool
outputs, and ordinary dialogue. They all consume KV-cache memory, but deleting them does
not have the same consequence.

PriorityKV started with a simple question: **if the serving layer already knows the role
of each span, should the cache policy use that information?**

![PriorityKV overview](../paper/figures/prioritykv_overview.svg)

## Two experiments that should not be confused

KV "compression" often combines two different interventions:

- **Eviction** removes positions from the cache. Their information is gone.
- **Quantization** retains every position in an approximate low-precision representation.

We initially expected both interventions to expose the same agent-specific quality gap.
They did not.

Under aggressive matched eviction, a sink-and-recent policy kept the correct number of
tokens but often deleted the middle state needed for the next tool action. On our targeted
Qwen3-8B stress slice, role-blind page retention scored 0.000 while structure-aware
retention scored 0.643 at a 25% keep budget. The gap remained across 15%, 25%, and 35%
page budgets.

![Matched keep sweep](../paper/figures/reliability_keep_sweep.svg)

The clean result needed an adversarial check. When we buried state so short-turn tagging
could no longer identify it, the structure score fell to 0.429. When we relocated state
away from the prefix, structure scored 0.688 while a fixed-prefix policy scored 0.125.
Together, these checks give a narrower and more useful conclusion: explicit protocol roles
protect tool and instruction state, but a simple heuristic does not discover every
free-form memory.

## The INT4 hypothesis failed

PriorityKV also assigns 75% of KV positions to INT4 while keeping protected positions in
BF16. The role-blind control assigns the same number of INT4 positions without consulting
message roles.

On the locked 240-example PriorityBench-A evaluation, the results were:

| Arm | Score |
|---|---:|
| FullKV | 0.8875 |
| Role-blind mixed INT4 | 0.8792 |
| Structure-aware mixed INT4 | 0.8833 |

That is not a meaningful quality separation. The structure-aware and role-blind arms
differ by one aggregate success out of 240. At 8k and 16k, all three arms scored 1.0; at
32k, FullKV itself degraded.

![Lock-240 quality](../paper/figures/lock240_quality_by_length.svg)

This negative result changed the project. We stopped searching for a convenient INT4
accuracy collapse and evaluated the mixed path as a systems artifact instead.

## Packed bytes are not peak memory

The implementation stores protected pages as BF16 and cold pages as packed INT4 with
group metadata. A Qwen3 attention shim uses FlashInfer for at most two homogeneous calls
per layer and merges their attention states using the correct log-sum-exp contract.

The current cold path still expands INT4 pages to a BF16 GPU scratch buffer before
attention:

![FlashInfer decode path](../paper/figures/flashinfer_decode_dataflow.svg)

This distinction matters. Relative to FullKV, the structure-aware path measured:

- `0.719x` packed payload;
- `0.473x` idealized modeled cache bytes;
- `0.868x` peak allocated CUDA memory;
- `1.11--1.12x` end-to-end time to first token; and
- `1.20--1.21x` time per output token.

The prototype saves real bytes and some peak memory, but it does not speed up decode. A
fused low-bit kernel or bounded page streaming would be required to turn payload savings
into a stronger peak and latency result.

![Systems tradeoff](../paper/figures/systems_tradeoff.svg)

## What the project establishes

PriorityKV is not a claim that application roles replace attention-based eviction methods.
It establishes that agent-serving systems possess a cheap prior they should not discard:
tool, system, and constraint spans are structurally different from filler.

It also reinforces three measurement rules:

1. Compare policies at the same token or byte budget.
2. Treat eviction and quantization as different hypotheses.
3. Report payload, allocator peak, packing cost, and decode latency separately.

## Reproduce it

The repository contains the benchmark generator and scorers, the locked manifest hash,
model revision, experiment configurations, H200 job records, packed-cache code, and the
script that regenerates every figure.

- Code: https://github.com/Arush777/Priority_KV
- Results: https://github.com/Arush777/Priority_KV/blob/main/RESULTS.md
- Reproducibility: https://github.com/Arush777/Priority_KV/blob/main/docs/REPRODUCIBILITY.md
- Technical report: **arXiv link to add after submission**

The report states the limitations directly: the main eviction evidence uses small
synthetic stress slices, Qwen3-8B on H200 is the only full matrix, and the cold scratch is
not a production low-bit attention kernel.
