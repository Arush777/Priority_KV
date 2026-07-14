"""W2-close job: FullKV vs uniform / structure / random keep at matched keep_frac."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from prioritybench.scoring import score_example
from prioritykv.baselines.buried_state import bury_short_state_turns
from prioritykv.baselines.keep_policy import KeepPolicyConfig
from prioritykv.baselines.keep_policy_run import run_transformers_keep_policy
from prioritykv.bench_pilot import _mean, materialize_examples
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, resolve_model_path
from prioritykv.stress_pilot import select_stress_rows


def run_structured_stress(
    config_path: Path,
    out_path: Path | None = None,
    *,
    reuse_full_path: Path | None = None,
    buried: bool | None = None,
) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")
    use_buried = bool(cfg.get("buried_state", False)) if buried is None else buried
    prompts = []
    for ex in examples:
        msgs = list(ex.messages)
        if use_buried:
            msgs = bury_short_state_turns(msgs, seed=hash(ex.example_id) % 10_000)
        prompts.append(PromptRow(id=ex.example_id, messages=msgs))
    # Scoring still uses original example gold (unchanged).

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    kcfg_raw = cfg.get("keep", {})
    keep_cfg = KeepPolicyConfig(
        keep_frac=float(kcfg_raw.get("keep_frac", 0.25)),
        sink_tokens=int(kcfg_raw.get("sink_tokens", 16)),
        force_recent=int(kcfg_raw.get("force_recent", 128)),
        seed=int(kcfg_raw.get("seed", 0)),
        page_tokens=int(kcfg_raw.get("page_tokens", 16)),
        granularity=str(kcfg_raw.get("granularity", "token")),
    )
    policies = list(kcfg_raw.get("policies", ["uniform", "structure", "random", "keep_all"]))

    full_texts: dict[str, str] = {}
    t_full = 0.0
    # Buried prompts change the input — never reuse FullKV from the unburied kill run.
    if reuse_full_path is not None and not use_buried:
        prior = json.loads(Path(reuse_full_path).read_text(encoding="utf-8"))
        for r in prior.get("rows", []):
            if r.get("fullkv_text"):
                full_texts[r["example_id"]] = r["fullkv_text"]
        for point in prior.get("curve", []):
            for r in point.get("rows", []):
                if r.get("fullkv_text"):
                    full_texts[r["example_id"]] = r["fullkv_text"]
        for arm in (prior.get("arms_detail") or {}).values():
            for r in arm.get("rows", []):
                if r.get("fullkv_text"):
                    full_texts[r["example_id"]] = r["fullkv_text"]
        if any(ex.example_id not in full_texts for ex in examples):
            full_texts.clear()
    elif use_buried and reuse_full_path is not None:
        print(
            "[structured] buried_state=1 → ignoring --reuse-full (must regen FullKV)",
            flush=True,
        )

    if not full_texts:
        t0 = time.time()
        full_out = _run_vllm_mode(
            model_path,
            prompts,
            max_new_tokens=max_new,
            kv_cache_dtype="auto",
            calculate_kv_scales=False,
            tp=int(vcfg["tensor_parallel_size"]),
            gpu_mem=float(vcfg["gpu_memory_utilization"]),
            max_model_len=int(vcfg["max_model_len"]),
        )
        t_full = time.time() - t0
        for ex, (ft, _) in zip(examples, full_out, strict=True):
            full_texts[ex.example_id] = ft

    arms: dict[str, Any] = {}
    seconds: dict[str, float] = {"fullkv": t_full}
    for policy in policies:
        t1 = time.time()
        outs = run_transformers_keep_policy(
            model_path,
            prompts,
            max_new,
            policy=policy,
            keep_cfg=keep_cfg,
            max_model_len=int(vcfg["max_model_len"]),
        )
        seconds[policy] = time.time() - t1
        detail = []
        by_cat: dict[str, dict[str, list[float]]] = {}
        by_len: dict[str, dict[str, list[float]]] = {}
        for ex, (txt, _, meta) in zip(examples, outs, strict=True):
            cat = ex.category.value
            ctx = str(ex.context_length)
            ft = full_texts[ex.example_id]
            sf = float(score_example(ex, ft))
            sp = float(score_example(ex, txt))
            by_cat.setdefault(cat, {"full": [], "pol": []})
            by_cat[cat]["full"].append(sf)
            by_cat[cat]["pol"].append(sp)
            by_len.setdefault(ctx, {"full": [], "pol": []})
            by_len[ctx]["full"].append(sf)
            by_len[ctx]["pol"].append(sp)
            detail.append(
                {
                    "example_id": ex.example_id,
                    "category": cat,
                    "context_length": ex.context_length,
                    "fullkv_score": sf,
                    "policy_score": sp,
                    "fullkv_text": ft,
                    "policy_text": txt,
                    "meta": meta,
                }
            )
        arms[policy] = {
            "mean": _mean([d["policy_score"] for d in detail]),
            "fullkv_mean": _mean([d["fullkv_score"] for d in detail]),
            "delta_minus_full": _mean([d["policy_score"] - d["fullkv_score"] for d in detail]),
            "by_category": {
                c: {
                    "n": len(v["pol"]),
                    "fullkv_mean": _mean(v["full"]),
                    "policy_mean": _mean(v["pol"]),
                }
                for c, v in sorted(by_cat.items())
            },
            "by_context_length": {
                L: {
                    "n": len(v["pol"]),
                    "fullkv_mean": _mean(v["full"]),
                    "policy_mean": _mean(v["pol"]),
                }
                for L, v in sorted(by_len.items())
            },
            "rows": detail,
        }

    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(examples),
        "buried_state": use_buried,
        "keep": keep_cfg.__dict__,
        "policies": policies,
        "fullkv_mean": _mean(
            [float(score_example(ex, full_texts[ex.example_id])) for ex in examples]
        ),
        "arms": {p: {k: v for k, v in arms[p].items() if k != "rows"} for p in policies},
        "arms_detail": arms,
        "seconds": seconds,
        "selected_ids": [ex.example_id for ex in examples],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "stress_structured"
            if scratch
            else root / "runs" / "stress_structured"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write compact summary + full detail
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
