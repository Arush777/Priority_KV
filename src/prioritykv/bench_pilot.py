"""PriorityBench quality pilot: FullKV vs FP8 scores (W2)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from prioritybench.generate import generate_one, template_by_id
from prioritybench.scoring import score_example
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, resolve_model_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def select_rows(bench: dict[str, Any], sel: dict[str, Any]) -> list[dict[str, Any]]:
    split = sel["split"]
    ctx = int(sel["context_length"])
    pool = [
        e
        for e in bench["examples"]
        if e["split"] == split and int(e["context_length"]) == ctx
    ]
    tools = [e for e in pool if e["category"] == "tool_schema"]
    supers = [e for e in pool if e["category"] == "instruction_supersession"]
    multi = [e for e in pool if e["category"] == "multi_turn_state"]
    n_t = int(sel.get("n_tool_schema", 10))
    n_s = int(sel.get("n_instruction_supersession", 5))
    n_m = int(sel.get("n_multi_turn_state", 0))
    chosen = tools[:n_t] + supers[:n_s] + multi[:n_m]
    if not chosen:
        raise ValueError("no examples matched selection")
    return chosen


def materialize_examples(rows: list[dict[str, Any]]):
    examples = []
    for row in rows:
        tmpl = template_by_id(row["template_id"])
        if tmpl is None:
            raise KeyError(row["template_id"])
        ex = generate_one(
            tmpl,
            seed=int(row["seed"]),
            context_length=int(row["context_length"]),
        )
        if ex.example_id != row["example_id"]:
            # seed/template/ctx must reconstruct the locked id
            raise ValueError(
                f"id mismatch: got {ex.example_id} expected {row['example_id']}"
            )
        examples.append(ex)
    return examples


def run_quality_pilot(
    config_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = _load_yaml(config_path)
    bench_path = root / cfg["bench_manifest"]
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    rows = select_rows(bench, cfg["selection"])
    examples = materialize_examples(rows)

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    fp8 = cfg["fp8"]
    prompts = [PromptRow(id=ex.example_id, messages=list(ex.messages)) for ex in examples]

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

    t1 = time.time()
    fp8_out = _run_vllm_mode(
        model_path,
        prompts,
        max_new_tokens=max_new,
        kv_cache_dtype=str(fp8.get("kv_cache_dtype", "fp8")),
        calculate_kv_scales=bool(fp8.get("calculate_kv_scales", True)),
        tp=int(vcfg["tensor_parallel_size"]),
        gpu_mem=float(vcfg["gpu_memory_utilization"]),
        max_model_len=int(vcfg["max_model_len"]),
    )
    t_fp8 = time.time() - t1

    detail = []
    by_cat_full: dict[str, list[float]] = {}
    by_cat_fp8: dict[str, list[float]] = {}
    for ex, (ft, _), (qt, _) in zip(examples, full_out, fp8_out, strict=True):
        sf = float(score_example(ex, ft))
        sq = float(score_example(ex, qt))
        cat = ex.category.value
        by_cat_full.setdefault(cat, []).append(sf)
        by_cat_fp8.setdefault(cat, []).append(sq)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "fullkv_score": sf,
                "fp8_score": sq,
                "fullkv_preview": ft[:240],
                "fp8_preview": qt[:240],
            }
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    cat_summary = {
        cat: {
            "n": len(by_cat_full[cat]),
            "fullkv_mean": _mean(by_cat_full[cat]),
            "fp8_mean": _mean(by_cat_fp8[cat]),
            "delta_fp8_minus_full": _mean(by_cat_fp8[cat]) - _mean(by_cat_full[cat]),
        }
        for cat in sorted(by_cat_full)
    }
    all_full = [d["fullkv_score"] for d in detail]
    all_fp8 = [d["fp8_score"] for d in detail]

    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(detail),
        "fullkv_mean": _mean(all_full),
        "fp8_mean": _mean(all_fp8),
        "delta_fp8_minus_full": _mean(all_fp8) - _mean(all_full),
        "by_category": cat_summary,
        "seconds": {"fullkv": t_full, "fp8": t_fp8},
        "selected_ids": [d["example_id"] for d in detail],
        "rows": detail,
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "w2_pb_quality"
            if scratch
            else root / "runs" / "w2_pb_quality"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
