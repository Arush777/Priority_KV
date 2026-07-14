"""Decisive stress pilot: FullKV vs ~10–60× DropKeep eviction.

This is the run that is *supposed* to leave perfect scores. Gentle INT4/FP8
stayed at 1.0; StreamingLLM-style sink+recent deletes early agent state.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from prioritybench.scoring import score_example
from prioritykv.baselines.drop_keep import DropKeepConfig
from prioritykv.baselines.drop_keep_run import run_transformers_dropkeep
from prioritykv.bench_pilot import materialize_examples, _mean
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, resolve_model_path


def select_stress_rows(bench: dict[str, Any], sel: dict[str, Any]) -> list[dict[str, Any]]:
    """Pool calibration+validation; prefer multi_turn across 8k/16k."""
    splits = set(sel.get("splits", ["calibration", "validation"]))
    lengths = set(int(x) for x in sel.get("context_lengths", [8000, 16000]))
    pool = [
        e
        for e in bench["examples"]
        if e["split"] in splits and int(e["context_length"]) in lengths
    ]
    tools = [e for e in pool if e["category"] == "tool_schema"]
    supers = [e for e in pool if e["category"] == "instruction_supersession"]
    multi = [e for e in pool if e["category"] == "multi_turn_state"]
    # Prefer longer contexts within each list (16k before 8k).
    multi.sort(key=lambda e: -int(e["context_length"]))
    supers.sort(key=lambda e: -int(e["context_length"]))
    tools.sort(key=lambda e: -int(e["context_length"]))
    chosen = (
        multi[: int(sel.get("n_multi_turn_state", 8))]
        + supers[: int(sel.get("n_instruction_supersession", 4))]
        + tools[: int(sel.get("n_tool_schema", 2))]
    )
    if not chosen:
        raise ValueError("no stress examples matched selection")
    return chosen


def run_stress_pilot(config_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")
    prompts = [PromptRow(id=ex.example_id, messages=list(ex.messages)) for ex in examples]

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    dcfg = cfg.get("dropkeep", {})

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

    drop_cfg = DropKeepConfig(
        sink_tokens=int(dcfg.get("sink_tokens", 16)),
        recent_tokens=int(dcfg.get("recent_tokens", 256)),
        keep_tokens=dcfg.get("keep_tokens"),
    )
    t1 = time.time()
    drop_out = run_transformers_dropkeep(
        model_path,
        prompts,
        max_new,
        max_model_len=int(vcfg["max_model_len"]),
        cfg=drop_cfg,
    )
    t_drop = time.time() - t1

    detail = []
    by_cat: dict[str, dict[str, list[float]]] = {}
    for ex, (ft, _), (dt, _, meta) in zip(examples, full_out, drop_out, strict=True):
        cat = ex.category.value
        sf = float(score_example(ex, ft))
        sd = float(score_example(ex, dt))
        bucket = by_cat.setdefault(cat, {"full": [], "drop": []})
        bucket["full"].append(sf)
        bucket["drop"].append(sd)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "fullkv_score": sf,
                "dropkeep_score": sd,
                "fullkv_text": ft,
                "dropkeep_text": dt,
                "dropkeep_meta": meta,
            }
        )

    cat_summary = {
        cat: {
            "n": len(b["full"]),
            "fullkv_mean": _mean(b["full"]),
            "dropkeep_mean": _mean(b["drop"]),
            "delta_drop_minus_full": _mean(b["drop"]) - _mean(b["full"]),
        }
        for cat, b in sorted(by_cat.items())
    }
    all_full = [d["fullkv_score"] for d in detail]
    all_drop = [d["dropkeep_score"] for d in detail]
    comps = [
        (d.get("dropkeep_meta") or {}).get("approx_compression_x")
        for d in detail
        if (d.get("dropkeep_meta") or {}).get("approx_compression_x")
    ]
    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(detail),
        "method": "fullkv_vs_dropkeep",
        "dropkeep": {
            "sink_tokens": drop_cfg.sink_tokens,
            "recent_tokens": drop_cfg.recent_tokens,
            "keep_tokens": drop_cfg.keep_tokens,
            "mean_compression_x": _mean([float(x) for x in comps]) if comps else None,
        },
        "fullkv_mean": _mean(all_full),
        "dropkeep_mean": _mean(all_drop),
        "delta_drop_minus_full": _mean(all_drop) - _mean(all_full),
        "by_category": cat_summary,
        "seconds": {"fullkv": t_full, "dropkeep": t_drop},
        "selected_ids": [d["example_id"] for d in detail],
        "rows": detail,
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "stress_dropkeep"
            if scratch
            else root / "runs" / "stress_dropkeep"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result


def _score_pair(
    examples,
    full_texts: dict[str, str],
    drop_out: list[tuple[str, list[int], dict[str, Any]]],
    *,
    drop_cfg: DropKeepConfig,
) -> dict[str, Any]:
    detail = []
    by_cat: dict[str, dict[str, list[float]]] = {}
    for ex, (dt, _, meta) in zip(examples, drop_out, strict=True):
        cat = ex.category.value
        ft = full_texts[ex.example_id]
        sf = float(score_example(ex, ft))
        sd = float(score_example(ex, dt))
        bucket = by_cat.setdefault(cat, {"full": [], "drop": []})
        bucket["full"].append(sf)
        bucket["drop"].append(sd)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "fullkv_score": sf,
                "dropkeep_score": sd,
                "fullkv_text": ft,
                "dropkeep_text": dt,
                "dropkeep_meta": meta,
            }
        )
    cat_summary = {
        cat: {
            "n": len(b["full"]),
            "fullkv_mean": _mean(b["full"]),
            "dropkeep_mean": _mean(b["drop"]),
            "delta_drop_minus_full": _mean(b["drop"]) - _mean(b["full"]),
        }
        for cat, b in sorted(by_cat.items())
    }
    all_full = [d["fullkv_score"] for d in detail]
    all_drop = [d["dropkeep_score"] for d in detail]
    comps = [
        float((d.get("dropkeep_meta") or {}).get("approx_compression_x"))
        for d in detail
        if (d.get("dropkeep_meta") or {}).get("approx_compression_x") is not None
    ]
    return {
        "dropkeep": {
            "sink_tokens": drop_cfg.sink_tokens,
            "recent_tokens": drop_cfg.recent_tokens,
            "keep_tokens": drop_cfg.keep_tokens,
            "mean_compression_x": _mean(comps) if comps else None,
        },
        "fullkv_mean": _mean(all_full),
        "dropkeep_mean": _mean(all_drop),
        "delta_drop_minus_full": _mean(all_drop) - _mean(all_full),
        "by_category": cat_summary,
        "rows": detail,
    }


def run_stress_sweep(
    config_path: Path,
    out_path: Path | None = None,
    *,
    reuse_full_path: Path | None = None,
) -> dict[str, Any]:
    """FullKV once, then DropKeep across recent_tokens budgets → drop-off curve."""
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")
    prompts = [PromptRow(id=ex.example_id, messages=list(ex.messages)) for ex in examples]

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    dcfg = cfg.get("dropkeep", {})
    sink = int(dcfg.get("sink_tokens", 16))
    budgets = [int(x) for x in dcfg.get("recent_tokens_sweep", [256, 512, 1024, 2048, 4096])]

    full_texts: dict[str, str] = {}
    t_full = 0.0
    if reuse_full_path is not None:
        prior = json.loads(Path(reuse_full_path).read_text(encoding="utf-8"))
        for r in prior.get("rows", []):
            if r.get("fullkv_text"):
                full_texts[r["example_id"]] = r["fullkv_text"]
        # also accept sweep prior
        for point in prior.get("curve", []):
            for r in point.get("rows", []):
                if r.get("fullkv_text"):
                    full_texts[r["example_id"]] = r["fullkv_text"]
        missing = [ex.example_id for ex in examples if ex.example_id not in full_texts]
        if missing:
            full_texts.clear()

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

    curve = []
    t_drop_total = 0.0
    for recent in budgets:
        drop_cfg = DropKeepConfig(sink_tokens=sink, recent_tokens=recent)
        print(f"[sweep] DropKeep recent={recent} sink={sink}", flush=True)
        t1 = time.time()
        drop_out = run_transformers_dropkeep(
            model_path,
            prompts,
            max_new,
            max_model_len=int(vcfg["max_model_len"]),
            cfg=drop_cfg,
        )
        dt = time.time() - t1
        t_drop_total += dt
        point = _score_pair(examples, full_texts, drop_out, drop_cfg=drop_cfg)
        point["seconds_dropkeep"] = dt
        # trim bulky texts in curve summary points (keep in nested rows for debug)
        curve.append(
            {
                "recent_tokens": recent,
                "sink_tokens": sink,
                "mean_compression_x": point["dropkeep"]["mean_compression_x"],
                "fullkv_mean": point["fullkv_mean"],
                "dropkeep_mean": point["dropkeep_mean"],
                "delta_drop_minus_full": point["delta_drop_minus_full"],
                "by_category": point["by_category"],
                "seconds_dropkeep": dt,
                "rows": point["rows"],
            }
        )
        print(
            f"[sweep] recent={recent} drop={point['dropkeep_mean']:.3f} "
            f"d={point['delta_drop_minus_full']:+.3f} "
            f"x≈{point['dropkeep']['mean_compression_x']}",
            flush=True,
        )

    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(examples),
        "method": "fullkv_vs_dropkeep_sweep",
        "sink_tokens": sink,
        "recent_tokens_sweep": budgets,
        "fullkv_mean": _mean([float(score_example(ex, full_texts[ex.example_id])) for ex in examples]),
        "curve": curve,
        "seconds": {"fullkv": t_full, "dropkeep_total": t_drop_total},
        "selected_ids": [ex.example_id for ex in examples],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "stress_dropkeep"
            if scratch
            else root / "runs" / "stress_dropkeep"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
