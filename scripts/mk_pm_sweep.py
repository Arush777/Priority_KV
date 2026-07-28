#!/usr/bin/env python3
"""Build the SWEEP_PM_V1 manifest: sweep examples + measured protected mass.

CPU-only.  Generates the protected-mass sweep (see prioritybench.pm_sweep),
verifies every instance still scores 1.0 on its gold answer, measures the
token-level protected-role fraction with the pinned tokenizer (same measure as
scripts/analyze_protected_fraction.py), and writes one manifest JSON embedding
the full messages so compute nodes need no generator or network.

    uv run python scripts/mk_pm_sweep.py \
        --tokenizer-dir "$PRIORITYKV_SCRATCH/models/Qwen3-8B"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import gold_tool_call  # noqa: E402
from prioritybench.pm_sweep import (  # noqa: E402
    PM_CONTEXT_LENGTH,
    PM_LEVELS,
    STRUCTURE_ROLES,
    SWEEP_PM_SEED,
    generate_pm_sweep,
)
from prioritybench.scoring import score_example  # noqa: E402


def _synth_pass(ex) -> str | None:
    """Gold answer for the sanity check (mirrors scripts/mk_bench.py)."""
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
        if "line" in slots:
            return str(slots["line"])
        return " ".join(str(v) for v in slots.values())
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SWEEP_PM_SEED)
    ap.add_argument("--context-length", type=int, default=PM_CONTEXT_LENGTH)
    ap.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Pinned tokenizer dir; when given, records token-level protected "
        "fraction and prompt length per instance (recommended).",
    )
    ap.add_argument(
        "--placement",
        choices=["prefix", "middle"],
        default="prefix",
        help="prefix = frozen template layout (v1); middle = gold relocated into "
        "the schema band, the condition under which oversubscription is testable",
    )
    ap.add_argument("--freeze-id", default="SWEEP_PM_V1")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prioritybench" / "manifests" / "pm_sweep_v1.json",
    )
    args = ap.parse_args()

    examples = generate_pm_sweep(
        master_seed=args.seed,
        levels=PM_LEVELS,
        n_per_category=args.n_per_category,
        context_length=args.context_length,
        placement=args.placement,
    )

    # Every instance must still be solvable-by-construction after conversion.
    for ex in examples:
        payload = _synth_pass(ex)
        if payload is None or score_example(ex, payload) != 1.0:
            print(f"synth fail {ex.example_id} payload={payload!r}", file=sys.stderr)
            return 1

    measured: dict[str, dict] = {}
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        from prioritybench.pins import chat_template_kwargs_for_tokenizer
        from prioritykv.baselines.keep_policy import assign_token_roles

        tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
        chat_kwargs = chat_template_kwargs_for_tokenizer(tok)
        for i, ex in enumerate(examples):
            roles = assign_token_roles(tok, list(ex.messages), chat_kwargs=chat_kwargs)
            n = len(roles)
            prot = sum(1 for r in roles if r in STRUCTURE_ROLES)
            measured[ex.example_id] = {
                "prompt_tokens": n,
                "protected_tokens": prot,
                "measured_protected_frac": prot / n if n else 0.0,
            }
            if (i + 1) % 60 == 0:
                print(f"[measure] {i + 1}/{len(examples)}", flush=True)

    rows = []
    for ex in examples:
        row = ex.to_dict()
        row.update(measured.get(ex.example_id, {}))
        rows.append(row)

    by_level: dict[str, list[float]] = {}
    for row in rows:
        lvl = row["meta"]["pm_level"]
        frac = row.get("measured_protected_frac", row["meta"]["pm_approx_frac"])
        by_level.setdefault(lvl, []).append(frac)

    manifest = {
        "freeze_id": args.freeze_id,
        "master_seed": args.seed,
        "levels": list(PM_LEVELS),
        "context_length": args.context_length,
        "placement": args.placement,
        "n": len(rows),
        "n_per_category": args.n_per_category,
        "buried_state_excluded": True,
        "measure": "assign_token_roles + STRUCTURE_ROLES, same as "
        "scripts/analyze_protected_fraction.py",
        "tokenizer_dir": args.tokenizer_dir,
        "category_hist": dict(Counter(r["category"] for r in rows)),
        "level_measured_frac_mean": {
            lvl: sum(v) / len(v) for lvl, v in sorted(by_level.items())
        },
        "examples": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"n={len(rows)} out={args.out}")
    for lvl, mean in sorted(manifest["level_measured_frac_mean"].items()):
        print(f"  {lvl}: measured_frac_mean={mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
