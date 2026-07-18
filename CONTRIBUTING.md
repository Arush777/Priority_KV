# Contributing to PriorityKV

PriorityKV is a frozen research artifact with an active packaging and maintenance track.
Changes should preserve the distinction between eviction results, mixed-precision quality,
and systems measurements.

## Before opening a change

1. Read `RESULTS.md`, `FINAL_RUN_MANIFEST.yaml`, and `docs/REPRODUCIBILITY.md`.
2. Open an issue for changes to benchmark semantics, frozen claims, or canonical jobs.
3. Do not retune a frozen configuration under an existing job ID. Bump the configuration
   revision and use a new job ID.
4. Do not include model tokens, API keys, private prompts, or third-party personal data.

## Local checks

```bash
./scripts/sync.sh
./scripts/check.sh
uv run ruff check src tests scripts
```

GPU changes require a separate H200 job artifact with the exact command, device assignment,
model revision, configuration, logs, and summary. Follow `jobs/README.md`; use at most two
GPUs and do not run GPU workloads on a login node.

## Pull requests

Keep changes focused. A pull request should state:

- the hypothesis or defect being addressed;
- whether it changes a frozen claim;
- the exact local and GPU verification performed;
- the artifact paths for new measurements; and
- any limitations or negative results.

By contributing, you agree that your contribution is licensed under Apache-2.0.
