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
    examples = materialize_examples(rows)
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
