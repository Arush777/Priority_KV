#!/usr/bin/env python3
"""Build PriorityBench pilots + compact ID manifests (CPU)."""

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
    W2B_MASTER_SEED,
    W2D_MASTER_SEED,
    generate_tool_schema_pilot,
    generate_w2_mixed_pilot,
    generate_w2b_pilot,
    generate_w2d_pilot,
    gold_tool_call,
    write_split_dirs,
)
from prioritybench.scoring import score_example  # noqa: E402


def _synth_pass(ex) -> str | None:
    cat = ex.category.value
    if cat == "tool_schema":
        return gold_tool_call(ex)
    if cat == "instruction_supersession":
        latest = ex.scoring.get("latest_constraint")
        if latest:
            return f"[[FMT:{latest}]] ok sentence about topic."
        for tok in ("alpha", "bravo", "charlie"):
            if tok in str(ex.scoring.get("constraint_pattern", "")):
                return f"Short reply with {tok}."
        return None
    if cat == "multi_turn_state":
        slots = ex.scoring.get("required_slots") or {}
        # Prefer the joined line if present.
        if "line" in slots:
            return str(slots["line"])
        return " ".join(str(v) for v in slots.values())
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--mode",
        choices=["w1", "w2", "w2b", "w2d"],
        default="w1",
        help="w1=tool; w2=tool+super; w2b=all 3 cats v1 (~145); w2d=all 3 cats v2 non-leak",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "prioritybench",
    )
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    if args.manifest is None:
        name = {
            "w1": "w1_pilot.json",
            "w2": "w2_pilot.json",
            "w2b": "w2b_pilot.json",
            "w2d": "w2d_pilot.json",
        }[args.mode]
        args.manifest = ROOT / "data" / "prioritybench" / "manifests" / name
    if args.seed is None:
        args.seed = {
            "w1": W1_MASTER_SEED,
            "w2": W2_MASTER_SEED,
            "w2b": W2B_MASTER_SEED,
            "w2d": W2D_MASTER_SEED,
        }[args.mode]

    if args.mode == "w1":
        examples = generate_tool_schema_pilot(args.n, master_seed=args.seed)
    elif args.mode == "w2":
        examples = generate_w2_mixed_pilot(master_seed=args.seed)
    elif args.mode == "w2b":
        examples = generate_w2b_pilot(master_seed=args.seed)
    else:
        examples = generate_w2d_pilot(master_seed=args.seed)

    for ex in examples:
        payload = _synth_pass(ex)
        if payload is None or score_example(ex, payload) != 1.0:
            print(f"synth fail {ex.example_id} payload={payload!r}", file=sys.stderr)
            return 1

    counts = write_split_dirs(args.out_dir, examples)
    manifest = {
        "master_seed": args.seed,
        "n": len(examples),
        "split_counts": counts,
        "context_hist": dict(Counter(ex.context_length for ex in examples)),
        "template_hist": dict(Counter(ex.template_id for ex in examples)),
        "category_hist": dict(Counter(ex.category.value for ex in examples)),
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
    print(f"n={len(examples)} splits={counts} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
