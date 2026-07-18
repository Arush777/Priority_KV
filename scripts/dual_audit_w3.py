#!/usr/bin/env python3
"""D1 dual audit: deterministic 15% sample of w3_lock + second-pass synth scoring.

Writes an optional dual-audit artifact. Does not retune locked gold and fails on a score miss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import gold_tool_call, load_jsonl  # noqa: E402
from prioritybench.schema import validate_example_shape  # noqa: E402
from prioritybench.scoring import score_example  # noqa: E402


def _synth_pass_local(ex):
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "prioritybench" / "manifests" / "w3_lock.json",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "prioritybench",
    )
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "dataset_audit_dual.md",
    )
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {}
    for split in ("calibration", "validation", "test"):
        p = args.data_root / split / "examples.jsonl"
        if p.exists():
            for ex in load_jsonl(p):
                by_id[ex.example_id] = ex
    ids = [row["example_id"] for row in man.get("examples", [])]
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"FAIL: missing {len(missing)} examples on disk — run mk_bench --mode w3_lock", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n = max(1, int(round(len(ids) * args.frac)))
    sample = sorted(rng.sample(ids, n))

    errors: list[str] = []
    cat_hist: Counter[str] = Counter()
    split_hist: Counter[str] = Counter()
    for eid in sample:
        ex = by_id[eid]
        cat_hist[ex.category.value] += 1
        split_hist[ex.split] += 1
        err = validate_example_shape(ex)
        if err:
            errors.append(f"{eid}: shape {err}")
            continue
        payload = _synth_pass_local(ex)
        if payload is None or score_example(ex, payload) != 1.0:
            if ex.category.value == "tool_schema":
                payload = gold_tool_call(ex)
            if payload is None or score_example(ex, payload) != 1.0:
                errors.append(f"{eid}: dual synth score fail payload={payload!r}")

    sha = sha256_file(args.manifest)
    passed = len(errors) == 0
    lines = [
        "# PriorityBench W3 dual audit (15%)",
        "",
        f"- **Manifest:** `{args.manifest.relative_to(ROOT)}`",
        f"- **SHA256 (must match lock):** `{sha}`",
        f"- **Sample:** n={n} / {len(ids)} ({args.frac:.0%}) seed={args.seed}",
        f"- **category_hist:** {dict(cat_hist)}",
        f"- **split_hist:** {dict(split_hist)}",
        f"- **dual_synth_errors:** {len(errors)}",
        f"- **PASS:** {passed}",
        "",
        "## Method",
        "",
        "Deterministic 15% sample; independent re-validation of shape + synth gold scoring",
        "(same synth path as `audit_bench.py`). Does **not** retune locked examples.",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors[:50]:
            lines.append(f"- `{e}`")
        if len(errors) > 50:
            lines.append(f"- … +{len(errors) - 50} more")
        lines.append("")
    else:
        lines.append("All sampled examples passed second-pass synth scoring.")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"pass={passed} n={n} sha256={sha} out={args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
