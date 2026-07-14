#!/usr/bin/env python3
"""Audit a PriorityBench manifest → docs/audit_w3.md + SHA256 lock line."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import gold_tool_call, load_jsonl  # noqa: E402
from prioritybench.schema import CATEGORIES, CONTEXT_LENGTHS, validate_example_shape  # noqa: E402
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
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "audit_w3.md",
    )
    ap.add_argument(
        "--w2d-manifest",
        type=Path,
        default=ROOT / "data" / "prioritybench" / "manifests" / "w2d_pilot.json",
    )
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = []
    for split in ("calibration", "validation", "test"):
        p = args.data_root / split / "examples.jsonl"
        if p.exists():
            examples.extend(load_jsonl(p))

    # Prefer on-disk rows matching manifest ids (order from jsonl).
    by_id = {ex.example_id: ex for ex in examples}
    ordered = []
    missing = []
    for row in man.get("examples", []):
        eid = row["example_id"]
        if eid in by_id:
            ordered.append(by_id[eid])
        else:
            missing.append(eid)
    if missing:
        print(f"missing {len(missing)} ids on disk (e.g. {missing[:3]})", file=sys.stderr)
        return 1
    examples = ordered

    errors: list[str] = []
    for ex in examples:
        err = validate_example_shape(ex)
        if err:
            errors.append(f"{ex.example_id}: {err}")
        payload = _synth_pass_local(ex)
        if payload is None or score_example(ex, payload) != 1.0:
            errors.append(f"{ex.example_id}: synth score fail payload={payload!r}")

    cat_hist = Counter(ex.category.value for ex in examples)
    split_hist = Counter(ex.split.value for ex in examples)
    ctx_hist = Counter(ex.context_length for ex in examples)
    buried_by_cat = {
        c.value: {
            "buried": sum(
                1
                for ex in examples
                if ex.category == c and bool(ex.meta.get("buried_state"))
            ),
            "plain": sum(
                1
                for ex in examples
                if ex.category == c and not bool(ex.meta.get("buried_state"))
            ),
            "n": sum(1 for ex in examples if ex.category == c),
        }
        for c in CATEGORIES
    }

    w2d_preserved = 0
    if args.w2d_manifest.exists():
        w2d = json.loads(args.w2d_manifest.read_text(encoding="utf-8"))
        w2d_ids = {r["example_id"] for r in w2d.get("examples", [])}
        w2d_preserved = sum(1 for ex in examples if ex.example_id in w2d_ids)
        if w2d_preserved != len(w2d_ids):
            errors.append(
                f"w2d preserve incomplete: {w2d_preserved}/{len(w2d_ids)} ids present"
            )

    lock_sha = sha256_file(args.manifest)
    ok = (
        len(examples) == 240
        and all(cat_hist[c.value] == 80 for c in CATEGORIES)
        and all(L in CONTEXT_LENGTHS for L in ctx_hist)
        and not errors
    )

    lines = [
        "# PriorityBench W3 lock audit",
        "",
        f"- **Manifest:** `{args.manifest.relative_to(ROOT)}`",
        f"- **SHA256:** `{lock_sha}`",
        f"- **n:** {len(examples)} (expect 240)",
        f"- **category_hist:** {dict(cat_hist)}",
        f"- **split_hist:** {dict(split_hist)}",
        f"- **context_hist:** {dict(ctx_hist)}",
        f"- **w2d_preserved:** {w2d_preserved}",
        f"- **buried_by_cat:** `{json.dumps(buried_by_cat)}`",
        f"- **synth_selfcheck_errors:** {len(errors)}",
        f"- **PASS:** {ok}",
        "",
        "## Notes",
        "",
        "- Pool-level buried target is **25% (20/80)** per category where room exists;",
        "  `tool_schema` may be 0 buried when W2d preserve already fills the quota.",
        "- Every example carries `meta.buried_state` for slice reporting.",
        "- Locked test examples must not be retuned after this SHA256 line is written",
        "  to `docs/decisions.md`.",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors[:50]:
            lines.append(f"- {e}")
        if len(errors) > 50:
            lines.append(f"- … and {len(errors) - 50} more")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"pass={ok} sha256={lock_sha} out={args.out}")
    if errors:
        for e in errors[:10]:
            print(f"  ERR {e}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
