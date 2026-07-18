# Reproducibility guide

PriorityKV separates CPU-verifiable logic, tracked frozen result bundles, and GPU
re-execution. Reproducing the complete H200 environment is not required to inspect the
claim or regenerate the paper figures.

## Frozen identifiers

| Item | Value |
|---|---|
| Science freeze | `SCIENCE_CORE_HOME_2026_07_19` |
| Primary model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Benchmark manifest | `data/prioritybench/manifests/w3_lock.json` |
| Benchmark SHA256 | `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89` |
| Primary hardware | NVIDIA H200 |

Do not modify a frozen configuration under an existing job ID. A changed configuration
requires a revision bump and a new job ID.

## Level 1: local logic

```bash
./scripts/sync.sh
./scripts/check.sh
uv run ruff check src tests scripts
```

This verifies benchmark generation and scoring, keep-policy budgets, role assignment,
packed INT4 round trips, byte accounting, page-manager invariants, and the CPU-visible
FlashInfer state contract. CUDA-specific tests skip when the GPU dependency set is absent.

## Level 2: dataset

```bash
PYTHONPATH=src uv run python scripts/mk_bench.py --mode w3_lock
PYTHONPATH=src uv run python scripts/audit_bench.py
sha256sum data/prioritybench/manifests/w3_lock.json
```

The manifest hash must match the value above. Generated JSONL split files are ignored by
Git and can be rebuilt from the templates and seeds.

## Level 3: frozen results

The main paper tables and figures use these tracked summaries:

| Claim | Job | Tracked summary |
|---|---|---|
| D4 latency | `d4_latency_m3c_gpu56_r1` | `jobs/results/d4_latency_m3c_gpu56_r1/summary.json` |
| Peak and payload | `mg_a_peak_mem_gpu5_r1` | `jobs/results/mg_a_peak_mem_gpu5_r1/summary.json` |
| Lock-240 mixed quality | `mg_b_lock240_quality_gpu01_r1` | `jobs/results/mg_b_lock240_quality_gpu01_r1/summary.json` |
| Gemma reduced | `pub_c_gemma_reduced_gpu01_r6` | `jobs/results/pub_c_gemma_reduced_gpu01_r6/summary.json` |

Generate all SVG and PDF figures with:

```bash
uv run python scripts/make_publication_figures.py
```

The matched-keep sweep figure reads `docs/atlas_w4_structure_rows.jsonl`, whose rows retain
the original H200 scratch result names. Results without tracked raw bundles are excluded
from the public evidence table and manuscript claims.

## Level 4: H200 re-execution

Install the GPU environment only on a GPU worker:

```bash
./scripts/sync.sh --cuda
export PRIORITYKV_SCRATCH=/data/anupam/scratch/prioritykv
```

Every canonical entry in `FINAL_RUN_MANIFEST.yaml` names its configuration, script, and
GPU assignment. Job YAML files under `jobs/manifests/` preserve the exact command. Follow
`jobs/README.md` for the output bundle schema.

The original operational constraints were:

- no GPU work on the login node;
- no more than two H200 GPUs per job;
- greedy deterministic generation unless the frozen config states otherwise;
- raw logs, summaries, and GPU snapshots retained together; and
- no silent fake-quantization fallback in real INT4 assertion runs.

## Expected deviations

CUDA, Transformers, vLLM, FlashInfer, and compiler versions can change kernel selection
and timing. A rerun should first reproduce output parity and qualitative decisions; exact
millisecond equality is not expected. Report allocator peak, reserved memory, packed
payload, pack time, cold-scratch time, end-to-end TTFT, and TPOT separately.
