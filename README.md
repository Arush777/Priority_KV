# PriorityKV

Structure-aware KV retention for long agent traces — keep the tokens that matter
(tool schemas, constraints, state) when the cache must shrink — plus a packed
BF16/INT4 systems path with honest latency and memory reporting.

**Arush Sharma** (IIT (ISM) Dhanbad), **Anupam Rawart** (IIT Bombay)  
Apache-2.0 · Python 3.11–3.12 · Primary eval: Qwen3-8B on NVIDIA H200

![PriorityKV system overview](paper/figures/prioritykv_overview.svg)

## Research question

Agent traces mix tool schemas, superseding instructions, persistent IDs, and ordinary
dialogue in one KV cache. Losing the wrong tokens can look fine on average metrics
while silently breaking agent behavior. PriorityKV asks whether **application-visible
message structure** should decide what to keep (and at what precision).

Scoped conclusions from the evidence:

1. **Eviction (strong on Qwen):** at matched keep budgets, structure-aware retention
   beats role-blind keep and common attention eviction (SnapKV / Pyramid / hybrid).
2. **Quantization (falsified):** soft INT4 at `int4_frac=0.75` does **not** open a
   PriorityBench quality gap vs FullKV.
3. **Systems:** packed storage cuts payload bytes; peak/latency are reported with the
   FI cold-scratch caveat — not as a free VRAM win.
4. **Transfer (honest):** Llama-3.1-8B at kf=0.25 is ceiling-saturated; do not claim a
   universal structure≫SnapKV transfer.

## Key results

Qwen3-8B (`b968826d…`) and Llama-3.1-8B-Instruct (`0e9e39f…`) on NVIDIA H200.
Full tables and job IDs: [`RESULTS.md`](RESULTS.md) · [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

| Experiment | Result |
|---|---|
| P0 token keep 25% (Qwen, n=120) | structure **0.933** vs uniform/random **~0.008** |
| P1 vs SnapKV/H2O/Pyramid (Qwen, n=120) | structure **0.933** > SnapKV/Pyr/hybrid **0.900** > H2O **~0.68** |
| P3 same protocol (Llama, n=120, kf=0.25) | all arms **1.000** (saturated / SnapKV matches) |
| Locked mixed quality (`n=240`) | FullKV **0.8875**, structure **0.8833**, uniform **0.8792** |
| Packed payload / modeled bytes | **0.719× / 0.473×** FullKV |
| Peak CUDA / E2E / TPOT | **0.868×** · **1.11–1.12×** · **1.20–1.21×** FullKV |

![Matched keep-budget results](paper/figures/reliability_keep_sweep.svg)

## Artifacts

| Artifact | Purpose |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Canonical metrics and claim boundary |
| [`docs/EVIDENCE.md`](docs/EVIDENCE.md) | P0–P3 evidence track and claim strength |
| [`docs/DATASET.md`](docs/DATASET.md) | PriorityBench-A tasks, strata, and generation |
| [`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml) | Frozen model, benchmark, configs, and job IDs |
| [`paper/prioritykv.tex`](paper/prioritykv.tex) | Standalone arXiv source |
| [`paper/prioritykv_manuscript.md`](paper/prioritykv_manuscript.md) | Readable source manuscript |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Reproduction levels and commands |
| [`docs/H200_QUEUE.md`](docs/H200_QUEUE.md) | Git→H200 job queue |
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
[`FINAL_RUN_MANIFEST.yaml`](FINAL_RUN_MANIFEST.yaml). Live git→H200 job queue (no agent SSH):
[`docs/H200_QUEUE.md`](docs/H200_QUEUE.md) · [`jobs/README.md`](jobs/README.md).
Do not run GPU code on a login node; use at most two H200 GPUs per job.

## Scope and limitations

- PriorityBench-A is synthetic and agent-specific; it is not LongBench or RULER.
- Early matched-eviction stress slices were small (n=14–16); P0/P1 now use n=120.
- Qwen carries the positive structure≫SnapKV result; Llama at kf=0.25 is saturated.
- The structure tagger is heuristic and misses some unmarked free-form state.
- The current cold path expands INT4 pages into BF16 scratch before attention (P2 streams it).
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
