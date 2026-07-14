#!/usr/bin/env python3
"""Build W1 PriorityBench pilot + compact ID manifest (CPU).

Usage:
  python scripts/mk_bench.py
  python scripts/mk_bench.py --n 40 --out-dir data/prioritybench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import (  # noqa: E402
    W1_MASTER_SEED,
    W2_MASTER_SEED,
    generate_tool_schema_pilot,
    generate_w2_mixed_pilot,
    gold_tool_call,
    write_split_dirs,
)
from prioritybench.scoring import score_example  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--mode",
        choices=["w1", "w2"],
        default="w1",
        help="w1=tool_schema only; w2=tool+supersession mixed",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "prioritybench",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    if args.manifest is None:
        name = "w1_pilot.json" if args.mode == "w1" else "w2_pilot.json"
        args.manifest = ROOT / "data" / "prioritybench" / "manifests" / name
    if args.seed is None:
        args.seed = W1_MASTER_SEED if args.mode == "w1" else W2_MASTER_SEED

    if args.mode == "w1":
        examples = generate_tool_schema_pilot(args.n, master_seed=args.seed)
    else:
        # Default w2 mix: 80 tool + 40 supersession (=120); --n ignored unless set via knobs later
        examples = generate_w2_mixed_pilot(master_seed=args.seed)

    # Validate gold for tool_schema; soft-check supersession patterns exist.
    for ex in examples:
        if ex.category.value == "tool_schema":
            g = gold_tool_call(ex)
            if score_example(ex, g) != 1.0:
                print(f"gold fail {ex.example_id}", file=sys.stderr)
                return 1
        else:
            # Supersession: synthesize a passing string from latest constraint.
            latest = ex.scoring.get("latest_constraint") or ex.scoring.get(
                "constraint_pattern"
            )
            if latest and score_example(ex, f"answer with {latest}") != 1.0:
                # language_flip forbids nothing; format_flip has forbidden — avoid old.
                payload = str(latest)
                forbidden = ex.scoring.get("forbidden_pattern")
                if forbidden:
                    payload = f"using {latest} only"
                if score_example(ex, payload) != 1.0:
                    print(f"synth fail {ex.example_id}", file=sys.stderr)
                    return 1

    counts = write_split_dirs(args.out_dir, examples)
    manifest = {
        "master_seed": args.seed,
        "n": len(examples),
        "split_counts": counts,
        "context_hist": dict(Counter(ex.context_length for ex in examples)),
        "template_hist": dict(Counter(ex.template_id for ex in examples)),
        "examples": [
            {
                "example_id": ex.example_id,
                "split": ex.split.value,
                "category": ex.category.value,
                "template_id": ex.template_id,
                "context_length": ex.context_length,
                "seed": ex.seed,
                "approx_tokens": ex.meta.get("approx_tokens"),
            }
            for ex in examples
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"n={len(examples)} splits={counts} manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
