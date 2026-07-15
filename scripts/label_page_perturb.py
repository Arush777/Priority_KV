#!/usr/bin/env python3
"""Page-perturb score_delta labels (W4) — structure vs uniform keep preference.

CPU-only: uses KeepPolicyConfig page selection with synthetic role ribbons.
Writes JSONL + fitted linear_risk weights for ProtectedRole++ ties.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import load_jsonl  # noqa: E402
from prioritykv.baselines.keep_policy import (  # noqa: E402
    KeepPolicyConfig,
    select_keep_indices,
)
from prioritykv.linear_risk import fit_ridge, page_features  # noqa: E402
from prioritykv.page_roles import PageRole  # noqa: E402


def _synth_roles(n: int, category: str) -> list[PageRole]:
    """Cheap role ribbon without a tokenizer (labeling assist only)."""
    roles = [PageRole.FILLER] * n
    for t in range(min(16, n)):
        roles[t] = PageRole.SINK
    for t in range(max(0, n - 128), n):
        roles[t] = PageRole.RECENT
    # Mid-context structural mass
    mid0, mid1 = n // 4, n // 4 + max(64, n // 10)
    mid1 = min(mid1, max(0, n - 128))
    role = {
        "tool_schema": PageRole.TOOL,
        "instruction_supersession": PageRole.CONSTRAINT,
        "multi_turn_state": PageRole.OTHER,
    }.get(category, PageRole.OTHER)
    for t in range(mid0, mid1):
        roles[t] = role
    return roles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "prioritybench")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--page-tokens", type=int, default=16)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prioritybench" / "manifests" / "page_perturb_labels.jsonl",
    )
    ap.add_argument(
        "--fit-out",
        type=Path,
        default=ROOT / "configs" / "linear_risk_fit.json",
    )
    args = ap.parse_args()

    path = args.data_root / args.split / "examples.jsonl"
    if not path.exists():
        print(f"missing {path} — run mk_bench --mode w3_lock", file=sys.stderr)
        return 1
    examples = load_jsonl(path)[: args.n]
    cfg = KeepPolicyConfig(
        keep_frac=args.keep_frac,
        sink_tokens=16,
        force_recent=128,
        seed=0,
        page_tokens=args.page_tokens,
        granularity="page",
    )

    rows = []
    for ex in examples:
        seq = int(getattr(ex, "context_length", None) or 8000)
        # Cap CPU cost for labeling
        seq = min(seq, 4096)
        roles = _synth_roles(seq, ex.category.value)
        st = set(int(i) for i in select_keep_indices(seq, cfg, policy="structure", roles=roles))
        uni = set(int(i) for i in select_keep_indices(seq, cfg, policy="uniform"))
        only_st = st - uni
        only_uni = uni - st
        for tok in list(sorted(only_st))[:64]:
            page_id = tok // args.page_tokens
            meta = {
                "roles": [roles[tok].name.lower()],
                "n_tokens": args.page_tokens,
                "page_id": page_id,
            }
            rows.append({
                "example_id": ex.example_id,
                "category": ex.category.value,
                "page_id": page_id,
                "meta": meta,
                "score_delta": 1.0,
                **page_features(meta),
            })
        for tok in list(sorted(only_uni))[:16]:
            page_id = tok // args.page_tokens
            meta = {
                "roles": [roles[tok].name.lower()],
                "n_tokens": args.page_tokens,
                "page_id": page_id,
            }
            rows.append({
                "example_id": ex.example_id,
                "category": ex.category.value,
                "page_id": page_id,
                "meta": meta,
                "score_delta": -0.25,
                **page_features(meta),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    fit = fit_ridge(rows)
    args.fit_out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    args.fit_out.write_text(json.dumps(asdict(fit), indent=2) + "\n", encoding="utf-8")
    print(f"n_label_rows={len(rows)} out={args.out} fit={args.fit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
