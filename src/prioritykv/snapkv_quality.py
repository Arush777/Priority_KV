"""Matched-byte SnapKV quality pilot vs DropKeep / FullKV (Q3 close)."""

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
from prioritykv.baselines.snapkv import SnapKVConfig
from prioritykv.baselines.snapkv_run import run_transformers_snapkv
from prioritykv.bench_pilot import _mean, materialize_examples
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, resolve_model_path
from prioritykv.stress_pilot import select_stress_rows


def run_snapkv_quality(
    config_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")
    prompts = [PromptRow(id=ex.example_id, messages=list(ex.messages)) for ex in examples]

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    keep_frac = float(cfg.get("keep_frac", 0.25))
    sink = int(cfg.get("sink_tokens", 16))
    force_recent = int(cfg.get("force_recent", 128))
    # SnapKV: compression_ratio = fraction removed ≈ 1 - keep_frac.
    snap_cfg = SnapKVConfig(
        budget_frac=keep_frac,
        compression_ratio=float(cfg.get("snapkv_compression_ratio", 1.0 - keep_frac)),
        window_size=int(cfg.get("snapkv_window_size", 64)),
        kernel_size=int(cfg.get("snapkv_kernel_size", 5)),
    )
    # DropKeep matched: keep_tokens ≈ keep_frac * typical len; use recent sized
    # so sink+recent ≈ keep_frac of median stress length (set explicitly in yaml).
    drop_keep_tokens = cfg.get("dropkeep_keep_tokens")
    drop_cfg = DropKeepConfig(
        sink_tokens=sink,
        recent_tokens=int(cfg.get("dropkeep_recent_tokens", force_recent)),
        keep_tokens=int(drop_keep_tokens) if drop_keep_tokens is not None else None,
    )

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
    full_texts = {ex.example_id: ft for ex, (ft, _) in zip(examples, full_out, strict=True)}

    t1 = time.time()
    drop_out = run_transformers_dropkeep(
        model_path,
        prompts,
        max_new,
        max_model_len=int(vcfg["max_model_len"]),
        cfg=drop_cfg,
    )
    t_drop = time.time() - t1

    t2 = time.time()
    try:
        snap_out = run_transformers_snapkv(
            model_path,
            prompts,
            max_new,
            max_model_len=int(vcfg["max_model_len"]),
            cfg=snap_cfg,
        )
        snap_error = None
    except Exception as exc:  # noqa: BLE001
        snap_out = None
        snap_error = f"{type(exc).__name__}: {exc}"
        print(f"[snapkv_quality] LOUD FAIL generate: {snap_error}", flush=True)
    t_snap = time.time() - t2

    detail = []
    by_cat: dict[str, dict[str, list[float]]] = {}
    for i, (ex, (dt, _, dmeta)) in enumerate(zip(examples, drop_out, strict=True)):
        cat = ex.category.value
        ft = full_texts[ex.example_id]
        sf = float(score_example(ex, ft))
        sd = float(score_example(ex, dt))
        ss = None
        stxt = ""
        smeta: dict[str, Any] = {}
        if snap_out is not None:
            stxt, _, smeta = snap_out[i]
            ss = float(score_example(ex, stxt))
        bucket = by_cat.setdefault(cat, {"full": [], "drop": [], "snap": []})
        bucket["full"].append(sf)
        bucket["drop"].append(sd)
        if ss is not None:
            bucket["snap"].append(ss)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "context_length": ex.context_length,
                "fullkv_score": sf,
                "dropkeep_score": sd,
                "snapkv_score": ss,
                "fullkv_text": ft,
                "dropkeep_text": dt,
                "snapkv_text": stxt,
                "dropkeep_meta": dmeta,
                "snapkv_meta": smeta,
            }
        )

    all_full = [d["fullkv_score"] for d in detail]
    all_drop = [d["dropkeep_score"] for d in detail]
    all_snap = [d["snapkv_score"] for d in detail if d["snapkv_score"] is not None]
    snap_mean = _mean(all_snap) if all_snap else None
    drop_mean = _mean(all_drop)
    full_mean = _mean(all_full)

    if snap_error is not None:
        decision = (
            "LOCK_Q_DROPKEEP — SnapKV import OK but matched-byte generate failed; "
            "DropKeep remains permanent eviction interim."
        )
    elif snap_mean is not None and snap_mean + 1e-9 >= drop_mean:
        decision = (
            "Q3_PARTIAL — SnapKV matched-byte runnable; mean≥DropKeep on this stress set. "
            "Structure-aware keep remains the PriorityKV path-(b) story."
        )
    else:
        decision = (
            "LOCK_Q_DROPKEEP — SnapKV runnable but not better than DropKeep at matched "
            f"keep_frac={keep_frac} on this stress set; DropKeep stays eviction interim."
        )

    result: dict[str, Any] = {
        "manifest_id": cfg.get("manifest_id", "w4_snapkv_matched"),
        "rev": cfg.get("rev", 1),
        "model_path": model_path,
        "n": len(examples),
        "keep_frac": keep_frac,
        "snapkv": {
            "compression_ratio": snap_cfg.compression_ratio,
            "window_size": snap_cfg.window_size,
            "kernel_size": snap_cfg.kernel_size,
            "error": snap_error,
        },
        "dropkeep": {
            "sink_tokens": drop_cfg.sink_tokens,
            "recent_tokens": drop_cfg.recent_tokens,
            "keep_tokens": drop_cfg.keep_tokens,
        },
        "fullkv_mean": full_mean,
        "dropkeep_mean": drop_mean,
        "snapkv_mean": snap_mean,
        "delta_snap_minus_drop": (
            None if snap_mean is None else float(snap_mean) - float(drop_mean)
        ),
        "by_category": {
            cat: {
                "n": len(v["full"]),
                "fullkv_mean": _mean(v["full"]),
                "dropkeep_mean": _mean(v["drop"]),
                "snapkv_mean": _mean(v["snap"]) if v["snap"] else None,
            }
            for cat, v in by_cat.items()
        },
        "decision": decision,
        "seconds": {"fullkv": t_full, "dropkeep": t_drop, "snapkv": t_snap},
        "rows": detail,
        "selected_ids": [ex.example_id for ex in examples],
    }

    scratch = os.environ.get("PRIORITYKV_SCRATCH")
    if out_path is None:
        out_dir = (
            Path(scratch) / "runs" / "snapkv_quality"
            if scratch
            else root / "runs" / "snapkv_quality"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{result['manifest_id']}_r1.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["out"] = str(out_path)
    print(
        f"n={result['n']} full={full_mean:.3f} drop={drop_mean:.3f} "
        f"snap={snap_mean if snap_mean is None else f'{snap_mean:.3f}'} "
        f"decision={decision.split('—')[0].strip()} out={out_path}",
        flush=True,
    )
    return result
