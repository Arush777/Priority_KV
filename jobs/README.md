# Experiment artifacts

This directory contains the frozen commands and tracked result bundles used by the
PriorityKV manuscript. It is an evidence archive, not a live job queue.

## Layout

| Directory | Contents |
|---|---|
| `manifests/` | Exact commands and GPU assignments for canonical H200 runs |
| `results/<job_id>/` | Summary, logs, environment metadata, and GPU snapshots |

## Canonical runs

| Job ID | Measurement |
|---|---|
| `d4_latency_m3c_gpu56_r1` | 8k/16k end-to-end and per-token latency |
| `mg_a_peak_mem_gpu5_r1` | Peak CUDA memory and packed payload bytes |
| `mg_b_lock240_quality_gpu01_r1` | Locked 240-example Qwen3-8B quality |
| `pub_c_gemma_reduced_gpu01_r6` | Reduced secondary-model stress check |

Each `summary.json` is machine-readable. `meta.json` records bundle metadata,
`nvidia_smi_before.txt` and `nvidia_smi.txt` capture device state, and log files preserve
the original program output. The canonical decision and process exit code are recorded in
the summary and manifest.

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for environment setup,
expected deviations, and the claim-to-artifact index.
