#!/usr/bin/env python3
"""Generate PriorityBench-A JSONL from templates + seeds (W1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import (  # noqa: E402
    W1_MASTER_SEED,
    generate_tool_schema_pilot,
    write_jsonl,
    write_split_dirs,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=40, help="Number of tool_schema examples")
    p.add_argument("--seed", type=int, default=W1_MASTER_SEED)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "prioritybench",
        help="Root for calibration/validation/test JSONL (gitignored)",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional small combined JSONL (e.g. data/prioritybench/fixtures/...)",
    )
    p.add_argument(
        "--fixture-n",
        type=int,
        default=3,
        help="How many examples to write into --fixture (default 3)",
    )
    args = p.parse_args()

    examples = generate_tool_schema_pilot(args.n, master_seed=args.seed)
    counts = write_split_dirs(args.out_dir, examples)
    print(f"wrote {sum(counts.values())} examples -> {args.out_dir} ({counts})")

    if args.fixture is not None:
        # Keep committed fixtures short-context only (CI size).
        short = [
            ex
            for ex in generate_tool_schema_pilot(
                args.fixture_n,
                master_seed=args.seed + 1,
                context_lengths=(8_000,),
            )
        ]
        write_jsonl(args.fixture, short)
        print(f"wrote fixture {len(short)} (8k) -> {args.fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
