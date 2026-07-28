#!/usr/bin/env python3
"""SWEEP_PM_V1 shard runner: retention arms across the protected-mass sweep.

One GPU per shard. Every non-FullKV arm is a kvpress press over the same full
prefill at a matched keep budget (the corrected mechanism from the external
evaluation); `full` uses the plain generate path. Scoring is the deterministic
PriorityBench scorer. Each (instance, arm, budget) work unit lands as one
atomic JSON point; finished points are skipped on resume, so shards are
interruptible and re-submittable.

    uv run python scripts/run_pm_sweep.py --config configs/pm_sweep_v1.yaml \
        --model qwen --shard-index 0 --shard-size 20
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_dotenv = ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from prioritybench.schema import PriorityExample  # noqa: E402
from prioritybench.scoring import score_example  # noqa: E402
from prioritykv.external.arms import (  # noqa: E402
    PressGenerator,
    TokenGatherGenerator,
)
from prioritykv.external.checkpoint import ResultStore, atomic_write_json  # noqa: E402
from prioritykv.external.config import harness_revision, load_config  # noqa: E402
from prioritykv.baselines.keep_policy import KeepPolicyConfig  # noqa: E402

_STOP = False


def _on_term(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"[pm] caught signal {signum}; finishing current point then exiting",
          flush=True)


def load_manifest_examples(path: Path) -> tuple[dict, list[PriorityExample], dict]:
    manifest = json.loads(path.read_text())
    examples = []
    measured = {}
    for row in manifest["examples"]:
        ex = PriorityExample.from_dict(row) if hasattr(PriorityExample, "from_dict") \
            else _example_from_row(row)
        examples.append(ex)
        measured[ex.example_id] = {
            "prompt_tokens_manifest": row.get("prompt_tokens"),
            "protected_tokens": row.get("protected_tokens"),
            "measured_protected_frac": row.get("measured_protected_frac"),
        }
    return manifest, examples, measured


def _example_from_row(row: dict) -> PriorityExample:
    from prioritybench.schema import Category, Split

    return PriorityExample(
        example_id=row["example_id"],
        category=Category(row["category"]),
        split=Split(row["split"]),
        context_length=int(row["context_length"]),
        template_id=row["template_id"],
        seed=int(row["seed"]),
        messages=row["messages"],
        scoring=row.get("scoring", {}),
        meta=row.get("meta", {}),
    )


def work_units(budgets: list[float], press_arms: list[str]) -> list[tuple[str, float | None]]:
    units: list[tuple[str, float | None]] = [("full", None)]
    for kf in budgets:
        for arm in press_arms:
            units.append((arm, kf))
    return units


def work_id(example_id: str, arm: str, kf: float | None) -> str:
    if kf is None:
        return f"{example_id}__{arm}"
    return f"{example_id}__{arm}__kf{int(round(kf * 100)):02d}"


def point_is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    return "score" in data and data.get("terminal_status") == "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pm_sweep_v1.yaml"))
    ap.add_argument("--model", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=20)
    ap.add_argument("--limit-instances", type=int, default=None,
                    help="Debug/smoke: cap instances within the shard")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    cfg = load_config(args.config)
    mcfg = cfg["models"][args.model]
    budgets = [float(b) for b in cfg["arms"]["budgets"]]
    press_arms = list(cfg["arms"]["press"])
    kp = cfg["arms"]["keep_policy"]
    snap = cfg["arms"]["snapkv"]
    seed = int(cfg["protocol"]["seed"])

    manifest_path = ROOT / cfg["manifest"]
    manifest, examples, measured = load_manifest_examples(manifest_path)

    examples.sort(key=lambda e: (e.meta.get("pm_level", ""), e.example_id))
    lo = args.shard_index * args.shard_size
    shard = examples[lo: lo + args.shard_size]
    if args.limit_instances:
        shard = shard[: args.limit_instances]
    if not shard:
        print(f"[pm] shard {args.shard_index} empty (n={len(examples)}); nothing to do")
        return 0

    store = ResultStore(Path(cfg["paths"]["results_root"]) / args.model).ensure()
    harness = harness_revision(ROOT)

    todo = [
        (ex, arm, kf)
        for ex in shard
        for arm, kf in work_units(budgets, press_arms)
        if not point_is_done(store.point_path(work_id(ex.example_id, arm, kf)))
    ]
    total = len(shard) * len(work_units(budgets, press_arms))
    print(f"[pm] shard={args.shard_index} instances={len(shard)} "
          f"pending={len(todo)}/{total}", flush=True)
    if not todo:
        print("[pm] SHARD_DONE (all points already present)")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(mcfg["local_dir"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["local_dir"],
        dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"],
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[pm] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    enable_thinking = mcfg.get("enable_thinking")
    max_new = int(mcfg["max_new_tokens"])

    generators: dict[tuple[str, float | None], object] = {}

    def gen_for(arm: str, kf: float | None):
        key = (arm, kf)
        if key not in generators:
            keep_cfg = KeepPolicyConfig(
                keep_frac=float(kf if kf is not None else 1.0),
                sink_tokens=int(kp["sink_tokens"]),
                force_recent=int(kp["force_recent"]),
                seed=seed,
                granularity=str(kp.get("granularity", "token")),
            )
            if arm == "full":
                generators[key] = TokenGatherGenerator(
                    model, tok, arm="full", keep_cfg=keep_cfg,
                    enable_thinking=enable_thinking,
                )
            else:
                generators[key] = PressGenerator(
                    model, tok, arm=arm, keep_cfg=keep_cfg,
                    window_size=int(snap["window_size"]),
                    kernel_size=int(snap["kernel_size"]),
                    enable_thinking=enable_thinking,
                )
        return generators[key]

    done = failed = 0
    for ex, arm, kf in todo:
        if _STOP:
            print(f"[pm] stopping early: done={done} failed={failed}", flush=True)
            break
        wid = work_id(ex.example_id, arm, kf)
        t1 = time.perf_counter()
        try:
            g = gen_for(arm, kf)
            res = g.generate(list(ex.messages), max_new_tokens=max_new)
            score = score_example(ex, res.text)
            payload = {
                "work_id": wid,
                "freeze_id": cfg["freeze_id"],
                "dataset_revision": manifest.get("master_seed"),
                "task_id": ex.example_id,
                "example_id": ex.example_id,
                "base_example_id": ex.meta.get("pm_base_example_id"),
                "pm_level": ex.meta.get("pm_level"),
                "pm_target_frac": ex.meta.get("pm_target_frac"),
                **measured.get(ex.example_id, {}),
                "category": ex.category.value,
                "template_id": ex.template_id,
                "context_length": ex.context_length,
                "model_id": mcfg["model_id"],
                "model_revision": mcfg["revision"],
                "arm": arm,
                "keep_frac": kf,
                "seed": seed,
                "score": float(score),
                "output_text": res.text,
                "prompt_tokens": res.prompt_tokens,
                "requested_keep": res.requested_keep,
                "realized_keep": res.realized_keep,
                "timings": res.timings,
                "extra": {k: v for k, v in res.extra.items()},
                "harness_revision": harness,
                "terminal_status": "ok",
            }
            atomic_write_json(store.point_path(wid), payload)
            done += 1
            print(f"[pm] ok {wid} score={score:.0f} "
                  f"{time.perf_counter() - t1:.1f}s ({done}/{len(todo)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            atomic_write_json(store.failure_path(wid), {
                "work_id": wid,
                "freeze_id": cfg["freeze_id"],
                "arm": arm,
                "keep_frac": kf,
                "error": repr(exc),
                "traceback": traceback.format_exc()[-4000:],
                "terminal_status": "error",
            })
            print(f"[pm] FAIL {wid}: {exc!r}", flush=True)

    print(f"[pm] SHARD_DONE done={done} failed={failed} pending_left={len(todo) - done - failed}",
          flush=True)
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
