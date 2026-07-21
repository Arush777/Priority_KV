#!/usr/bin/env python
"""CPU-only gold-span retention audit on public τ-bench trajectories.

Mechanistic evidence only. This never runs the τ-bench user simulator, never
touches a tool backend, and never generates: it measures whether naturally
occurring schemas, identifiers, tool results, and constraints survive each
deterministic retention policy at a matched keep budget.

    uv run python scripts/audit_tau_retention.py \
        --config configs/external_bfcl_prajna_v1.yaml --limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.baselines.keep_policy import assign_token_roles, select_keep_indices  # noqa: E402
from prioritykv.external.checkpoint import ResultStore, atomic_write_json, write_jsonl  # noqa: E402
from prioritykv.external.config import keep_policy_config, load_config  # noqa: E402
from prioritykv.external.tau_spans import (  # noqa: E402
    SPAN_CLASSES,
    aggregate,
    char_to_token_spans,
    extract_spans,
    load_trajectories,
    measure_retention,
    render,
    sample_for_manual_audit,
    stratified_sample,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="max trajectories (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="tau_audit")
    ap.add_argument("--keep-frac", type=float, default=None)
    return ap.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()
    cfg = load_config(args.config)
    tau_cfg = cfg["tau_audit"]
    keep_frac = float(args.keep_frac if args.keep_frac is not None else tau_cfg["keep_frac"])
    policies = list(tau_cfg["policies"])
    seed = int(tau_cfg["seed"])

    trajectories = load_trajectories(cfg["paths"]["tau_dataset"])
    print(f"[tau] loaded {len(trajectories)} trajectories", flush=True)
    if args.limit and len(trajectories) > args.limit:
        trajectories = stratified_sample(trajectories, n=args.limit, seed=seed)
        print(f"[tau] stratified sample -> {len(trajectories)}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["local_dir"], trust_remote_code=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for offset mapping")

    keep_cfg = keep_policy_config(cfg, keep_frac=keep_frac, seed=seed)

    retentions: dict[str, list] = defaultdict(list)
    per_trajectory: list[dict] = []
    all_spans: list[tuple[str, object]] = []
    class_counts: Counter = Counter()
    n_skipped = 0

    for j, traj in enumerate(trajectories):
        rendered = render(traj)
        spans = extract_spans(rendered)
        if not spans:
            n_skipped += 1
            continue
        token_spans, n_tokens = char_to_token_spans(tok, rendered.text, spans)
        if n_tokens == 0 or not token_spans:
            n_skipped += 1
            continue

        class_counts.update(s.span_class for s in spans)
        all_spans.extend((traj.traj_id, s) for s in spans)

        # Roles come from the real chat messages, exactly as the serving policy
        # would see them — not from anything span-derived.
        roles = assign_token_roles(tok, traj.messages, chat_kwargs={})
        if len(roles) != n_tokens:
            # Chat-template rendering and the flat transcript disagree; fall back
            # to positional roles rather than mis-aligning spans.
            roles = roles[:n_tokens] + [roles[-1]] * max(0, n_tokens - len(roles))

        row = {"traj_id": traj.traj_id, "task_name": traj.task_name,
               "source_model": traj.source_model, "context_tokens": n_tokens,
               "n_spans": len(token_spans), "is_correct": traj.is_correct}
        for policy in policies:
            idx = select_keep_indices(n_tokens, keep_cfg, policy=policy, roles=roles)
            rets = measure_retention(token_spans, idx, n_tokens)
            retentions[policy].extend(rets)
            row[f"{policy}_any_rate"] = (
                sum(r.any_retained for r in rets) / len(rets) if rets else 0.0
            )
            row[f"{policy}_realized_keep"] = int(len(idx))
        per_trajectory.append(row)

        if (j + 1) % 250 == 0:
            print(f"[tau] {j + 1}/{len(trajectories)}", flush=True)

    store = ResultStore(args.out or cfg["paths"]["results_root"]).ensure()
    by_policy = {p: aggregate(retentions[p]) for p in policies}

    # Structure-vs-blind gap on the buried classes is the claim of interest.
    gaps = {}
    if "structure" in by_policy:
        s = by_policy["structure"]["by_visibility"]
        for other in (p for p in policies if p != "structure"):
            o = by_policy[other]["by_visibility"]
            gaps[f"structure_minus_{other}"] = {
                vis: round(
                    s.get(vis, {}).get("any_retained_rate", 0.0)
                    - o.get(vis, {}).get("any_retained_rate", 0.0), 4
                )
                for vis in ("visible_structure", "buried")
            }

    manual = sample_for_manual_audit(
        all_spans, n=int(tau_cfg["manual_audit_sample"]), seed=seed
    )
    write_jsonl(store.summaries / f"{args.tag}_manual_sample.jsonl", manual)
    write_jsonl(store.summaries / f"{args.tag}_per_trajectory.jsonl", per_trajectory)

    summary = {
        "freeze_id": cfg["freeze_id"],
        "evidence_type": "mechanistic_retention_only",
        "limitations": [
            "This is NOT a tau-bench evaluation and measures no task success.",
            "No user simulator, tool backend, or generation was involved.",
            "SnapKV is excluded: it scores tokens by realised attention and "
            "cannot be computed generation-free on CPU.",
            "Trajectories are clustered by task and source model; repeated "
            "trajectories are not independent behavioural evidence.",
        ],
        "dataset_repo": tau_cfg["dataset_repo"],
        "dataset_revision": tau_cfg["revision"],
        "policies": policies,
        "keep_frac": keep_frac,
        "n_trajectories_loaded": len(trajectories),
        "n_trajectories_audited": len(per_trajectory),
        "n_trajectories_skipped_no_spans": n_skipped,
        "span_class_counts": dict(class_counts),
        "span_classes": list(SPAN_CLASSES),
        "by_policy": by_policy,
        "structure_gap_any_retained": gaps,
        "by_source_model": dict(Counter(r["source_model"] for r in per_trajectory)),
        "by_task": dict(Counter(r["task_name"] for r in per_trajectory)),
        "manual_audit_sample_size": len(manual),
        "manual_audit_status": "PENDING_HUMAN_REVIEW",
    }
    out_path = store.summaries / f"{args.tag}_summary.json"
    atomic_write_json(out_path, summary)

    print(json.dumps({
        "n_audited": len(per_trajectory),
        "span_classes": dict(class_counts),
        "any_retained_by_policy": {
            p: round(by_policy[p].get("overall", {}).get("any_retained_rate", 0.0), 4)
            for p in policies
        },
        "buried_any_retained": {
            p: round(by_policy[p]["by_visibility"]["buried"].get("any_retained_rate", 0.0), 4)
            for p in policies
        },
        "structure_gap": gaps,
        "summary": str(out_path),
    }, indent=2))
    print("TAU_AUDIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
