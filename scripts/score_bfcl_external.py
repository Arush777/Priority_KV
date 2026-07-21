#!/usr/bin/env python
"""Aggregate EXTERNAL_BFCL_PRAJNA_V1 points into the paired five-arm table.

CPU-only. Reads the atomic point files, restricts every comparison to the set of
conversations scored in *all* arms, and emits overall/category/length breakdowns,
exact paired McNemar, paired bootstrap CIs, and completeness/failure ledgers.

    uv run python scripts/score_bfcl_external.py \
        --config configs/external_bfcl_prajna_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.external.checkpoint import (  # noqa: E402
    ResultStore,
    atomic_write_json,
    load_valid_point,
    read_jsonl,
)
from prioritykv.external.config import load_config  # noqa: E402
from prioritykv.external.stats import (  # noqa: E402
    arm_summary,
    build_paired_table,
    mcnemar,
    paired_bootstrap_ci,
    paired_completeness,
    restrict_to_common,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--length-bins", default="0,8000,16000,32000")
    return ap.parse_args()


def length_bin(n: int, edges: list[int]) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return f">={edges[-1]}"


def main() -> int:  # noqa: C901
    args = parse_args()
    cfg = load_config(args.config)
    store = ResultStore(args.out or cfg["paths"]["results_root"]).ensure()

    work_items = read_jsonl(store.manifest / "work_items.jsonl")
    tasks = {t["task_id"]: t for t in read_jsonl(store.manifest / "tasks.jsonl")}
    exclusions = read_jsonl(store.manifest / "exclusions.jsonl")

    points = [p for p in (load_valid_point(f) for f in sorted(store.points.glob("*.json")))
              if p is not None]
    failures = [json.loads(f.read_text()) for f in sorted(store.failures.glob("*.json"))]
    print(f"[score] points={len(points)} failures={len(failures)} "
          f"work_items={len(work_items)}", flush=True)

    outcomes: dict[str, dict[str, bool]] = defaultdict(dict)
    keep_by_task: dict[str, dict[str, int]] = defaultdict(dict)
    for p in points:
        outcomes[p["arm"]][p["task_id"]] = bool(p.get("score_valid"))
        realized = p.get("realized_keep") or []
        if realized:
            keep_by_task[p["task_id"]][p["arm"]] = max(realized)

    arms = [a for a in cfg["arms"]["primary"] if a in outcomes]
    expected_tasks = sorted({w["task_id"] for w in work_items})

    completeness = paired_completeness(expected_tasks, outcomes)
    paired = restrict_to_common(outcomes)
    n_paired = len(next(iter(paired.values()))) if paired else 0
    print(f"[score] paired completeness "
          f"{completeness.paired_completeness:.3f} over {n_paired} tasks", flush=True)

    # Matched-budget audit: every non-FullKV arm must land on the same keep count.
    budget_violations = []
    for task_id, per_arm in keep_by_task.items():
        vals = {a: v for a, v in per_arm.items() if a != "full" and v >= 0}
        if len(vals) >= 2 and max(vals.values()) - min(vals.values()) > 0:
            budget_violations.append({"task_id": task_id, "realized_keep": vals,
                                      "spread": max(vals.values()) - min(vals.values())})

    edges = [int(x) for x in args.length_bins.split(",")]
    overall = {a: arm_summary(paired.get(a, {})) for a in arms}

    by_category: dict[str, dict] = {}
    by_length: dict[str, dict] = {}
    for a in arms:
        outs = paired.get(a, {})
        cat_split: dict[str, dict[str, bool]] = defaultdict(dict)
        len_split: dict[str, dict[str, bool]] = defaultdict(dict)
        for task_id, v in outs.items():
            meta = tasks.get(task_id, {})
            cat_split[meta.get("category", "unknown")][task_id] = v
            len_split[length_bin(int(meta.get("prompt_token_count", 0)), edges)][task_id] = v
        by_category[a] = {c: arm_summary(o) for c, o in sorted(cat_split.items())}
        by_length[a] = {b: arm_summary(o) for b, o in sorted(len_split.items())}

    comparisons = []
    for a, b in combinations(arms, 2):
        table = build_paired_table(a, b, paired.get(a, {}), paired.get(b, {}))
        row = mcnemar(table)
        row["bootstrap"] = paired_bootstrap_ci(
            paired.get(a, {}), paired.get(b, {}),
            n_boot=int(cfg["statistics"]["bootstrap"]["n_boot"]),
            alpha=float(cfg["statistics"]["bootstrap"]["alpha"]),
            seed=int(cfg["statistics"]["bootstrap"]["seed"]),
        )
        comparisons.append(row)

    # Claim guard: "matches SnapKV" unless the paired test proves superiority.
    claims = []
    for row in comparisons:
        if {row["arm_a"], row["arm_b"]} == {"structure", "snapkv"}:
            if row["p_exact_mcnemar"] < 0.05:
                better = row["arm_a"] if row["a_only"] > row["b_only"] else row["arm_b"]
                claims.append(f"structure vs snapkv: {better} superior "
                              f"(p={row['p_exact_mcnemar']:.4f})")
            else:
                claims.append("structure MATCHES snapkv; the paired test does not "
                              f"establish superiority (p={row['p_exact_mcnemar']:.4f})")

    summary = {
        "freeze_id": cfg["freeze_id"],
        "tag": args.tag,
        "arms": arms,
        "keep_frac": cfg["arms"]["keep_frac"],
        "n_tasks_expected": completeness.n_tasks_expected,
        "n_tasks_paired": n_paired,
        "paired_completeness": completeness.paired_completeness,
        "meets_min_paired_completeness": (
            completeness.paired_completeness
            >= float(cfg["statistics"]["min_paired_completeness"])
        ),
        "per_arm_complete": completeness.per_arm_complete,
        "overall": overall,
        "by_category": by_category,
        "by_length": by_length,
        "paired_comparisons": comparisons,
        "claim_guard": claims,
        "matched_budget_violations": budget_violations[:50],
        "n_matched_budget_violations": len(budget_violations),
        "exclusions": {
            "n": len(exclusions),
            "by_reason": dict(Counter(e.get("reason") for e in exclusions)),
            "by_category": dict(Counter(e.get("category") for e in exclusions)),
        },
        "failures": {
            "n": len(failures),
            "by_status": dict(Counter(f.get("terminal_status") for f in failures)),
            "by_arm": dict(Counter(f.get("arm") for f in failures)),
        },
        "missing_by_arm": {a: v[:50] for a, v in completeness.missing_by_arm.items()},
    }

    out_path = store.summaries / f"summary_{args.tag}.json"
    atomic_write_json(out_path, summary)

    print(json.dumps({
        "overall": {a: round(overall[a]["accuracy"], 4) for a in arms},
        "n_paired": n_paired,
        "paired_completeness": round(completeness.paired_completeness, 4),
        "claims": claims,
        "matched_budget_violations": len(budget_violations),
        "summary": str(out_path),
    }, indent=2))
    print("SCORE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
