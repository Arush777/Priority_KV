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
    W3_MASTER_SEED,
    W5_STRESS_SEED,
    generate_tool_schema_pilot,
    generate_w2_mixed_pilot,
    generate_w2b_pilot,
    generate_w2d_pilot,
    generate_w3_lock_pilot,
    generate_w5_stress_large_pilot,
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
        choices=["w1", "w2", "w2b", "w2d", "w3_lock", "w5_stress_large"],
        default="w1",
        help="w1=tool; …; w3_lock=240 locked; w5_stress_large=P0 120-ex stress pool",
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
            "w3_lock": "w3_lock.json",
            "w5_stress_large": "w5_stress_large.json",
        }[args.mode]
        args.manifest = ROOT / "data" / "prioritybench" / "manifests" / name
    if args.seed is None:
        args.seed = {
            "w1": W1_MASTER_SEED,
            "w2": W2_MASTER_SEED,
            "w2b": W2B_MASTER_SEED,
            "w2d": W2D_MASTER_SEED,
            "w3_lock": W3_MASTER_SEED,
            "w5_stress_large": W5_STRESS_SEED,
        }[args.mode]

    if args.mode == "w1":
        examples = generate_tool_schema_pilot(args.n, master_seed=args.seed)
    elif args.mode == "w2":
        examples = generate_w2_mixed_pilot(master_seed=args.seed)
    elif args.mode == "w2b":
        examples = generate_w2b_pilot(master_seed=args.seed)
    elif args.mode == "w2d":
        examples = generate_w2d_pilot(master_seed=args.seed)
    elif args.mode == "w5_stress_large":
        examples = generate_w5_stress_large_pilot(master_seed=args.seed)
    else:
        examples = generate_w3_lock_pilot(master_seed=args.seed)

    for ex in examples:
        payload = _synth_pass(ex)
        if payload is None or score_example(ex, payload) != 1.0:
            print(f"synth fail {ex.example_id} payload={payload!r}", file=sys.stderr)
            return 1

    counts = write_split_dirs(args.out_dir, examples)
    w2d_ids = set()
    if args.mode == "w3_lock":
        w2d_ids = {ex.example_id for ex in generate_w2d_pilot()}
    manifest = {
        "master_seed": args.seed,
        "n": len(examples),
        "mode": args.mode,
        "split_counts": counts,
        "context_hist": dict(Counter(ex.context_length for ex in examples)),
        "template_hist": dict(Counter(ex.template_id for ex in examples)),
        "category_hist": dict(Counter(ex.category.value for ex in examples)),
        "buried_hist": {
            cat: {
                "buried": sum(
                    1
                    for ex in examples
                    if ex.category.value == cat and bool(ex.meta.get("buried_state"))
                ),
                "plain": sum(
                    1
                    for ex in examples
                    if ex.category.value == cat and not bool(ex.meta.get("buried_state"))
                ),
            }
            for cat in sorted({ex.category.value for ex in examples})
        },
        "w2d_preserved_n": (
            sum(1 for ex in examples if ex.example_id in w2d_ids) if w2d_ids else None
        ),
        "examples": [
            {
                "example_id": ex.example_id,
                "split": ex.split.value,
                "category": ex.category.value,
                "template_id": ex.template_id,
                "context_length": ex.context_length,
                "seed": ex.seed,
                "approx_tokens": ex.meta.get("approx_tokens"),
                "buried_state": bool(ex.meta.get("buried_state")),
                "w2d_preserved": bool(ex.meta.get("w2d_preserved")),
                "replication_slice": ex.meta.get("replication_slice"),
            }
            for ex in examples
        ],
    }
    if args.mode == "w5_stress_large":
        manifest["replication"] = {
            "n_slices": 3,
            "slice_counts": dict(
                Counter(
                    int(ex.meta.get("replication_slice", -1))
                    for ex in examples
                    if ex.meta.get("replication_slice") is not None
                )
            ),
            "note": "Three disjoint ~40-example replications; run each slice separately.",
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"n={len(examples)} splits={counts} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
