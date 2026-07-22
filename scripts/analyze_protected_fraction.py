#!/usr/bin/env python
"""Measure the protected-role fraction of a workload.

The external evaluation's central finding is a boundary condition: structure-aware
retention can only help when protected content is a *minority* of the context. If
nearly every token carries a protected role, "keep the structure" selects almost
everything and the policy degenerates to index order, carrying no signal.

This script measures that fraction directly, per workload, on CPU. It is the
diagnostic a practitioner would run before choosing a retention policy.

    uv run python scripts/analyze_protected_fraction.py \
        --config configs/external_bfcl_prajna_v1.yaml --per-category 40
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from prioritykv.baselines.keep_policy import assign_token_roles  # noqa: E402
from prioritykv.external.arms import keep_budget  # noqa: E402
from prioritykv.external.bfcl_data import balanced_sample, build_system_prompt, load_tasks  # noqa: E402
from prioritykv.external.bfcl_official import load_official  # noqa: E402
from prioritykv.external.bfcl_rollout import messages_for_turn  # noqa: E402
from prioritykv.external.checkpoint import ResultStore, atomic_write_json  # noqa: E402
from prioritykv.external.config import keep_policy_config, load_config  # noqa: E402
from prioritykv.external.tau_spans import load_trajectories, stratified_sample  # noqa: E402
from prioritykv.page_roles import PROTECTED_ROLES, PageRole  # noqa: E402

STRUCTURE_ROLES = frozenset(PROTECTED_ROLES) | {PageRole.OTHER}


def summarise(name: str, fracs: list[float], budgets: list[float],
              role_mix: Counter) -> dict:
    f = np.asarray(fracs, dtype=float)
    b = np.asarray(budgets, dtype=float)
    total = sum(role_mix.values()) or 1
    return {
        "workload": name,
        "n_examples": int(f.size),
        "protected_fraction_mean": float(f.mean()) if f.size else 0.0,
        "protected_fraction_p50": float(np.median(f)) if f.size else 0.0,
        "protected_fraction_min": float(f.min()) if f.size else 0.0,
        "protected_fraction_max": float(f.max()) if f.size else 0.0,
        "keep_budget_fraction_mean": float(b.mean()) if b.size else 0.0,
        # The diagnostic: when protected mass exceeds the budget the policy is
        # oversubscribed and cannot express a preference.
        "oversubscribed_rate": float((f > b).mean()) if f.size else 0.0,
        "role_mix": {k: v / total for k, v in role_mix.most_common()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml"))
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--tau-limit", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    keep_cfg = keep_policy_config(cfg)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["local_dir"], trust_remote_code=True)
    chat_kwargs = {"enable_thinking": bool(cfg["model"].get("enable_thinking", True))}

    gorilla = cfg["dataset"]["gorilla_root"]
    official = load_official(gorilla)
    categories = list(cfg["dataset"]["categories"])
    tasks = load_tasks(gorilla, categories=categories,
                       doc_mapping=official["MULTI_TURN_FUNC_DOC_FILE_MAPPING"])
    sampled = balanced_sample(tasks, per_category={c: args.per_category for c in categories},
                              seed=0)

    workloads: list[dict] = []
    per_cat: dict[str, tuple[list, list, Counter]] = {
        c: ([], [], Counter()) for c in categories
    }
    all_f, all_b, all_mix = [], [], Counter()

    for i, t in enumerate(sampled):
        sp = build_system_prompt(t, official["DEFAULT_SYSTEM_PROMPT"])
        msgs = messages_for_turn(t, sp, t.n_turns - 1)
        roles = assign_token_roles(tok, msgs, chat_kwargs=chat_kwargs)
        n = len(roles)
        if n == 0:
            continue
        prot = sum(1 for r in roles if r in STRUCTURE_ROLES) / n
        bud = keep_budget(n, keep_cfg) / n
        mix = Counter(r.value for r in roles)
        f, b, m = per_cat[t.category]
        f.append(prot)
        b.append(bud)
        m.update(mix)
        all_f.append(prot)
        all_b.append(bud)
        all_mix.update(mix)
        if (i + 1) % 25 == 0:
            print(f"[pf] bfcl {i + 1}/{len(sampled)}", flush=True)

    for c in categories:
        f, b, m = per_cat[c]
        workloads.append(summarise(f"BFCL-{c}", f, b, m))
    workloads.append(summarise("BFCL-all", all_f, all_b, all_mix))

    # PriorityBench-A: the frozen synthetic core, where structure wins. This is
    # the low-protected-fraction end of the boundary and the reason the frozen
    # result and the external result do not contradict each other.
    try:
        from prioritybench.generate import generate_w3_lock_pilot

        pf, pb, pmix = [], [], Counter()
        for e in generate_w3_lock_pilot()[: args.per_category * 2]:
            msgs = e.messages if hasattr(e, "messages") else e["messages"]
            roles = assign_token_roles(tok, msgs, chat_kwargs=chat_kwargs)
            n = len(roles)
            if not n:
                continue
            pf.append(sum(1 for r in roles if r in STRUCTURE_ROLES) / n)
            pb.append(keep_budget(n, keep_cfg) / n)
            pmix.update(r.value for r in roles)
        workloads.append(summarise("PriorityBench-A", pf, pb, pmix))
    except Exception as exc:  # noqa: BLE001
        print(f"[pf] PriorityBench-A unavailable: {exc}", flush=True)

    # tau-bench: same measurement on real agent transcripts.
    trajs = load_trajectories(cfg["paths"]["tau_dataset"])
    trajs = stratified_sample(trajs, n=args.tau_limit, seed=0)
    tf, tb, tmix = [], [], Counter()
    for j, tr in enumerate(trajs):
        if not tr.messages:
            continue
        roles = assign_token_roles(tok, tr.messages, chat_kwargs={})
        n = len(roles)
        if n == 0:
            continue
        tf.append(sum(1 for r in roles if r in STRUCTURE_ROLES) / n)
        tb.append(keep_budget(n, keep_cfg) / n)
        tmix.update(r.value for r in roles)
        if (j + 1) % 100 == 0:
            print(f"[pf] tau {j + 1}/{len(trajs)}", flush=True)
    workloads.append(summarise("tau-bench", tf, tb, tmix))

    store = ResultStore(args.out or cfg["paths"]["results_root"]).ensure()
    out = store.summaries / "protected_fraction.json"
    atomic_write_json(out, {
        "freeze_id": cfg["freeze_id"],
        "keep_frac": cfg["arms"]["keep_frac"],
        "interpretation": (
            "Structure-aware retention can only express a preference when the "
            "protected mass is smaller than the keep budget. Where "
            "oversubscribed_rate is 1.0, 'keep the structure' selects everything "
            "and degenerates to index order."
        ),
        "workloads": workloads,
    })

    print()
    print(f"{'workload':22s} {'n':>4} {'protected':>10} {'budget':>8} {'oversub':>8}")
    for w in workloads:
        print(f"{w['workload']:22s} {w['n_examples']:4d} "
              f"{w['protected_fraction_mean']:9.1%} "
              f"{w['keep_budget_fraction_mean']:7.1%} "
              f"{w['oversubscribed_rate']:7.0%}")
    print(f"\nwrote {out}")
    print("PROTECTED_FRACTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
