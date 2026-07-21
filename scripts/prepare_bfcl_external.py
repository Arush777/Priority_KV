#!/usr/bin/env python
"""Freeze the BFCL V3 multi-turn work manifest for EXTERNAL_BFCL_PRAJNA_V1.

CPU-only. Renders every sampled conversation with the pinned Qwen tokenizer to
measure real prompt length, excludes over-context tasks *with a reason* rather
than truncating them, and writes the manifest, exclusions, and file hashes.

    uv run python scripts/prepare_bfcl_external.py \
        --config configs/external_bfcl_prajna_v1.yaml --n 600
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.external import FREEZE_ID  # noqa: E402
from prioritykv.external.bfcl_data import (  # noqa: E402
    balanced_sample,
    build_system_prompt,
    file_hashes,
    load_tasks,
    work_id,
)
from prioritykv.external.bfcl_official import (  # noqa: E402
    assert_pinned_revision,
    load_official,
)
from prioritykv.external.bfcl_rollout import messages_for_turn  # noqa: E402
from prioritykv.external.checkpoint import ResultStore, atomic_write_json, write_jsonl  # noqa: E402
from prioritykv.external.config import (  # noqa: E402
    harness_revision,
    load_config,
    uv_lock_hash,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml"))
    ap.add_argument("--n", type=int, default=None, help="total tasks (default: config target)")
    ap.add_argument("--keep-frac", type=float, default=None)
    ap.add_argument("--arms", default=None, help="comma-separated (default: config primary)")
    ap.add_argument("--out", default=None, help="results root (default: config)")
    ap.add_argument("--limit-tasks", type=int, default=None, help="debug: cap tasks scanned")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    gorilla_root = cfg["dataset"]["gorilla_root"]
    dataset_revision = assert_pinned_revision(
        gorilla_root, cfg["dataset"]["gorilla_revision"]
    )
    official = load_official(gorilla_root)

    categories = list(cfg["dataset"]["categories"])
    tasks = load_tasks(gorilla_root, categories=categories,
                       doc_mapping=official["MULTI_TURN_FUNC_DOC_FILE_MAPPING"])
    print(f"[prepare] loaded {len(tasks)} tasks across {categories}", flush=True)

    # Choose the per-category quota.
    total = args.n if args.n is not None else int(cfg["sampling"]["target_n"])
    if total == int(cfg["sampling"]["target_n"]):
        per_cat = dict(cfg["sampling"]["per_category_target"])
    elif total == int(cfg["sampling"]["minimum_n"]):
        per_cat = dict(cfg["sampling"]["per_category_minimum"])
    else:
        base, extra = divmod(total, len(categories))
        per_cat = {c: base + (1 if i < extra else 0) for i, c in enumerate(categories)}
    print(f"[prepare] per-category quota: {per_cat}", flush=True)

    sampled = balanced_sample(tasks, per_category=per_cat,
                              seed=int(cfg["sampling"]["seed"]))
    if args.limit_tasks:
        sampled = sampled[: args.limit_tasks]

    # Render with the pinned tokenizer to get true prompt lengths.
    from transformers import AutoTokenizer

    model_dir = cfg["model"]["local_dir"]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ceiling = int(cfg["model"]["prompt_token_ceiling"])

    kept, exclusions, length_rows = [], [], []
    for task in sampled:
        system_prompt = build_system_prompt(task, official["DEFAULT_SYSTEM_PROMPT"])
        # Longest static prefix = system + every user turn, before any tool
        # output. This is a lower bound on the real rollout prompt; the runner
        # still enforces the ceiling live and records any late exclusion.
        msgs = messages_for_turn(task, system_prompt, task.n_turns - 1)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=cfg["model"]["enable_thinking"])
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        length_rows.append({"task_id": task.task_id, "category": task.category,
                            "prompt_token_count": n_tok, "n_turns": task.n_turns})
        if n_tok > ceiling:
            exclusions.append({
                "task_id": task.task_id, "category": task.category,
                "reason": "MODEL_CONTEXT_LIMIT",
                "prompt_token_count": n_tok, "ceiling": ceiling,
            })
            continue
        kept.append((task, n_tok))

    print(f"[prepare] kept {len(kept)}, excluded {len(exclusions)} "
          f"(MODEL_CONTEXT_LIMIT)", flush=True)
    if exclusions:
        print(f"[prepare] exclusions by category: "
              f"{dict(Counter(e['category'] for e in exclusions))}", flush=True)

    arms = (args.arms.split(",") if args.arms else list(cfg["arms"]["primary"]))
    keep_frac = float(args.keep_frac if args.keep_frac is not None
                      else cfg["arms"]["keep_frac"])
    seed = int(cfg["protocol"]["seed"])
    hrev = harness_revision(REPO_ROOT)

    task_rows, work_rows = [], []
    for task, n_tok in kept:
        task_rows.append({
            "task_id": task.task_id, "category": task.category,
            "n_turns": task.n_turns, "prompt_token_count": n_tok,
            "involved_classes": task.involved_classes,
            "n_functions": len(task.function),
        })
        for arm in arms:
            work_rows.append({
                "work_id": work_id(
                    dataset_revision=dataset_revision, task_id=task.task_id,
                    model_revision=cfg["model"]["revision"], arm=arm,
                    keep_frac=keep_frac, seed=seed, harness_revision=hrev,
                    decision_turn=cfg["protocol"]["decision_turn"],
                ),
                "freeze_id": FREEZE_ID,
                "dataset_revision": dataset_revision,
                "task_id": task.task_id,
                "category": task.category,
                "decision_turn": cfg["protocol"]["decision_turn"],
                "model_id": cfg["model"]["model_id"],
                "model_revision": cfg["model"]["revision"],
                "arm": arm,
                "keep_frac": keep_frac,
                "seed": seed,
                "harness_revision": hrev,
                "prompt_token_count": n_tok,
            })

    dup = len(work_rows) - len({w["work_id"] for w in work_rows})
    if dup:
        raise RuntimeError(f"{dup} duplicate work_ids — identity fields are not unique")

    store = ResultStore(args.out or cfg["paths"]["results_root"]).ensure()
    write_jsonl(store.manifest / "tasks.jsonl", task_rows)
    write_jsonl(store.manifest / "work_items.jsonl", work_rows)
    write_jsonl(store.manifest / "exclusions.jsonl", exclusions)
    write_jsonl(store.manifest / "prompt_lengths.jsonl", length_rows)
    atomic_write_json(store.manifest / "hashes.json", {
        "freeze_id": FREEZE_ID,
        "dataset_revision": dataset_revision,
        "model_id": cfg["model"]["model_id"],
        "model_revision": cfg["model"]["revision"],
        "harness_revision": hrev,
        "uv_lock_sha256": uv_lock_hash(REPO_ROOT),
        "arms": arms,
        "keep_frac": keep_frac,
        "per_category_quota": per_cat,
        "n_tasks": len(task_rows),
        "n_work_items": len(work_rows),
        "n_excluded": len(exclusions),
        "data_file_sha256": file_hashes(gorilla_root, categories),
    })

    lens = sorted(r["prompt_token_count"] for r in length_rows)
    def q(p): return lens[min(len(lens) - 1, int(p * len(lens)))] if lens else 0
    print(json.dumps({
        "n_tasks": len(task_rows), "n_work_items": len(work_rows),
        "n_excluded": len(exclusions),
        "prompt_tokens": {"p50": q(0.50), "p75": q(0.75), "p95": q(0.95),
                          "max": lens[-1] if lens else 0},
        "by_category": dict(Counter(r["category"] for r in task_rows)),
        "manifest": str(store.manifest),
    }, indent=2))
    print("PREPARE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
