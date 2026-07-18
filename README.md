# PriorityKV

Structure-aware KV retention and mixed BF16/INT4 storage for long language-model
agent traces.

**Arush Sharma** (IIT (ISM) Dhanbad), **Anupam Rawart** (IIT Bombay)
Apache-2.0 | Python 3.11--3.12 | Primary evaluation: Qwen3-8B on NVIDIA H200

![PriorityKV system overview](paper/figures/prioritykv_overview.svg)

## Research question

Agent traces contain tool schemas, superseding instructions, persistent identifiers, and
ordinary dialogue. All of these tokens occupy the KV cache, but losing them has different
behavioral consequences. PriorityKV asks whether application-visible message structure
should influence which KV pages are retained or stored at high precision.

The frozen evidence supports three scoped conclusions:

1. **Eviction:** role-blind sink-and-recent eviction destroys targeted agent state at
   aggressive matched keep budgets; structure-aware retention preserves substantially
   more of it.
2. **Quantization:** assigning 75% of positions to soft INT4 does **not** create a
   meaningful PriorityBench quality separation from FullKV. This hypothesis was
   falsified.
3. **Systems:** real packed storage reduces payload bytes, but the current BF16 cold
   scratch limits peak savings and makes decode slower. Payload, peak, and latency are
   reported separately.

## Key results

Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`, NVIDIA H200:

| Experiment | Result |
|---|---|
| Token eviction, 25% keep (`n=14`) | role-blind **0.000**, structure **1.000** |
| Page eviction, 25% keep (`n=14`) | role-blind **0.000**, structure **0.643** |
| Middle-relocated page state (`n=16`) | fixed prefix **0.125**, structure **0.688** |
| Locked mixed quality (`n=240`) | FullKV **0.8875**, role-blind **0.8792**, structure **0.8833** |
| Packed payload / modeled bytes | **0.719x / 0.473x** FullKV |
| Peak allocated CUDA memory | **0.868x** FullKV |
| E2E / TPOT at 8k--16k | **1.11--1.12x / 1.20--1.21x** FullKV |

The positive eviction experiments are controlled stress slices, not a broad benchmark
matrix. The lock-240 result is a negative quantization finding, not evidence that
structure-aware INT4 improves quality.

![Matched keep-budget results](paper/figures/reliability_keep_sweep.svg)

## Artifacts

| Artifact | Purpose |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Canonical metrics and claim boundary |
| [`docs/DATASET.md`](docs/DATASET.md) | PriorityBench-A tasks, strata, and generation |
| [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml) | Frozen model, benchmark, configs, and job IDs |
| [`paper/prioritykv.tex`](paper/prioritykv.tex) | Standalone arXiv source |
| [`paper/prioritykv_manuscript.md`](paper/prioritykv_manuscript.md) | Readable source manuscript |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Reproduction levels and commands |
| [`docs/BLOG.md`](docs/BLOG.md) | Accessible research summary |
| [`CITATION.cff`](CITATION.cff) | Citation metadata |

Main source modules:

```text
src/prioritybench/    deterministic benchmark generator and scorers
src/prioritykv/       roles, policies, packed cache, and FlashInfer decode
configs/              frozen experiment configurations
jobs/                 canonical H200 commands and result bundles
tests/                CPU unit and contract tests
paper/figures/        reproducibly generated SVG/PDF figures
```

## Local reproduction

Install the CPU development environment and run the complete local check:

```bash
git clone https://github.com/Arush777/Priority_KV.git
cd Priority_KV
./scripts/sync.sh
./scripts/check.sh
```

Regenerate and audit PriorityBench-A:

```bash
PYTHONPATH=src uv run python scripts/mk_bench.py --mode w3_lock
PYTHONPATH=src uv run python scripts/audit_bench.py
```

Regenerate all paper figures from tracked frozen artifacts:

```bash
uv run python scripts/make_publication_figures.py
```

## GPU reproduction

GPU dependencies are isolated from the CPU environment:

```bash
./scripts/sync.sh --cuda
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
```

Canonical commands and device assignments are indexed in
[`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml). The H200 worker contract and result
bundle format are documented in [`jobs/README.md`](jobs/README.md). Do not run GPU code on
a login node; the original environment used at most two H200 GPUs per job.

## Scope and limitations

- PriorityBench-A is synthetic and agent-specific; it is not LongBench or RULER.
- The decisive matched-eviction runs contain 14--16 examples.
- Qwen3-8B on H200 is the only complete model/device matrix.
- The structure tagger is heuristic and misses some unmarked free-form state.
- The current cold path expands INT4 pages into BF16 scratch before attention.
- The latency study is single-request and does not measure serving throughput or tail
  latency under concurrency.

See the manuscript for the full threats-to-validity discussion.

## Citation and license

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

PriorityKV is licensed under the [Apache License 2.0](LICENSE). Model weights, benchmark
dependencies, and third-party libraries retain their respective licenses. Author
affiliations do not imply institutional endorsement.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing benchmark semantics, frozen
claims, or canonical run configurations. Security reports should follow
[`SECURITY.md`](SECURITY.md).
