"""PriorityBench quality pilot: FullKV vs FP8 (+ INT4) scores (W2/W2c)."""

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
from prioritykv.int4_baseline import run_transformers_int4
from prioritykv.int4_kv import Int4KvConfig


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


def materialize_examples(
    rows: list[dict[str, Any]],
    *,
    data_root: Path | None = None,
):
    """Build PriorityExample list for selected manifest rows.

    Prefers on-disk JSONL (required for buried W3 rows whose template_id ends
    in ``.buried``). Falls back to regenerate-from-seed for W1/W2 pilots.
    """
    from prioritybench.generate import assign_split, load_jsonl
    from prioritybench.schema import PriorityExample as PE
    from prioritykv.baselines.buried_state import bury_short_state_turns

    disk: dict[str, Any] = {}
    if data_root is not None:
        for split in ("calibration", "validation", "test"):
            p = Path(data_root) / split / "examples.jsonl"
            if p.exists():
                for ex in load_jsonl(p):
                    disk[ex.example_id] = ex

    examples = []
    for row in rows:
        eid = row["example_id"]
        if eid in disk:
            examples.append(disk[eid])
            continue
        tid = str(row["template_id"])
        bury = bool(row.get("buried_state")) or tid.endswith(".buried") or eid.endswith(
            "__buried"
        )
        if tid.endswith(".buried"):
            tid = tid[: -len(".buried")]
        tmpl = template_by_id(tid)
        if tmpl is None:
            raise KeyError(row["template_id"])
        ex = generate_one(
            tmpl,
            seed=int(row["seed"]),
            context_length=int(row["context_length"]),
        )
        if bury:
            buried_msgs = bury_short_state_turns(ex.messages, seed=int(row["seed"]))
            want = eid if eid.endswith("__buried") else f"{ex.example_id}__buried"
            ex = PE(
                example_id=want,
                category=ex.category,
                split=assign_split(want),
                context_length=ex.context_length,
                template_id=ex.template_id + ".buried",
                seed=ex.seed,
                messages=buried_msgs,
                scoring=ex.scoring,
                meta={**dict(ex.meta), "buried_state": True, "parent_id": ex.example_id},
            )
        if ex.example_id != eid:
            raise ValueError(f"id mismatch: got {ex.example_id} expected {eid}")
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
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")

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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_triple_pilot(
    config_path: Path,
    out_path: Path | None = None,
    *,
    reuse_path: Path | None = None,
) -> dict[str, Any]:
    """FullKV vs FP8 vs uniform INT4 on the same PriorityBench selection.

    INT4 runs via HF QuantizedCache / fake-groupwise (see int4_baseline).
    Optional ``reuse_path``: JSON from a prior FullKV/FP8 pilot to skip vLLM
    and only score INT4 (saves GPU hours when iterating on INT4).
    """
    root = config_path.resolve().parents[1]
    cfg = _load_yaml(config_path)
    bench_path = root / cfg["bench_manifest"]
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    rows = select_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    fp8 = cfg.get("fp8", {})
    icfg = cfg.get("int4", {})
    modes = str(icfg.get("modes", "all")).lower()
    allow_fake = bool(icfg.get("allow_fake_fallback", True))
    prompts = [PromptRow(id=ex.example_id, messages=list(ex.messages)) for ex in examples]

    full_by_id: dict[str, str] = {}
    fp8_by_id: dict[str, str] = {}
    t_full = t_fp8 = 0.0

    if reuse_path is not None:
        prior = json.loads(Path(reuse_path).read_text(encoding="utf-8"))
        for r in prior.get("rows", []):
            full_by_id[r["example_id"]] = r.get("fullkv_preview_full") or r.get(
                "fullkv_text", r.get("fullkv_preview", "")
            )
            # Prefer full text fields when present
            if "fullkv_text" in r:
                full_by_id[r["example_id"]] = r["fullkv_text"]
            if "fp8_text" in r:
                fp8_by_id[r["example_id"]] = r["fp8_text"]
            elif "fp8_preview" in r and r["example_id"] not in fp8_by_id:
                fp8_by_id[r["example_id"]] = r["fp8_preview"]
        # If prior was w2b-style and only has previews, fall through to regenerate
        if len(full_by_id) != len(examples) or any(
            len(full_by_id[ex.example_id]) < 8 for ex in examples
        ):
            full_by_id.clear()
            fp8_by_id.clear()

    if not full_by_id and modes in ("all", "skip_fp8", "vllm_only"):
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
            full_by_id[ex.example_id] = ft

    if not fp8_by_id and modes in ("all", "vllm_only"):
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
        for ex, (qt, _) in zip(examples, fp8_out, strict=True):
            fp8_by_id[ex.example_id] = qt

    # Checkpoint FullKV/FP8 before the slow INT4 leg so crashes can reuse.
    if full_by_id and modes in ("all", "skip_fp8", "vllm_only"):
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        ck_dir = (
            Path(scratch) / "runs" / "w2c_pb_quality"
            if scratch
            else root / "runs" / "w2c_pb_quality"
        )
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_path = ck_dir / f"{cfg['manifest_id']}_r{cfg['rev']}_vllm_partial.json"
        partial = {
            "manifest_id": cfg["manifest_id"],
            "rev": cfg["rev"],
            "partial": "vllm_full_fp8",
            "rows": [
                {
                    "example_id": ex.example_id,
                    "category": ex.category.value,
                    "fullkv_text": full_by_id.get(ex.example_id, ""),
                    "fp8_text": fp8_by_id.get(ex.example_id, ""),
                    "fullkv_score": float(
                        score_example(ex, full_by_id[ex.example_id])
                    )
                    if ex.example_id in full_by_id and full_by_id[ex.example_id]
                    else None,
                    "fp8_score": float(score_example(ex, fp8_by_id[ex.example_id]))
                    if ex.example_id in fp8_by_id and fp8_by_id[ex.example_id]
                    else None,
                }
                for ex in examples
            ],
        }
        ck_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    int4_by_id: dict[str, str] = {}
    int4_meta: dict[str, Any] = {}
    t_int4 = 0.0
    if modes in ("all", "skip_fp8", "int4_only"):
        int4_cfg = Int4KvConfig(
            nbits=int(icfg.get("nbits", 4)),
            group_size=int(icfg.get("group_size", 32)),
            backend=str(icfg.get("backend", "quanto")),
        )
        t2 = time.time()
        int4_out = run_transformers_int4(
            model_path,
            prompts,
            max_new,
            max_model_len=int(vcfg["max_model_len"]),
            cfg=int4_cfg,
            prefer_quanto=bool(icfg.get("prefer_quanto", True)),
            allow_fake_fallback=allow_fake,
        )
        t_int4 = time.time() - t2
        for ex, (it, _, meta) in zip(examples, int4_out, strict=True):
            int4_by_id[ex.example_id] = it
            int4_meta[ex.example_id] = meta

    detail = []
    by_cat: dict[str, dict[str, list[float]]] = {}
    for ex in examples:
        cat = ex.category.value
        ft = full_by_id.get(ex.example_id, "")
        qt = fp8_by_id.get(ex.example_id, "")
        it = int4_by_id.get(ex.example_id, "")
        sf = float(score_example(ex, ft)) if ft else float("nan")
        sq = float(score_example(ex, qt)) if qt else float("nan")
        si = float(score_example(ex, it)) if it else float("nan")
        bucket = by_cat.setdefault(cat, {"full": [], "fp8": [], "int4": []})
        if ft:
            bucket["full"].append(sf)
        if qt:
            bucket["fp8"].append(sq)
        if it:
            bucket["int4"].append(si)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "fullkv_score": sf,
                "fp8_score": sq,
                "int4_score": si,
                "fullkv_text": ft,
                "fp8_text": qt,
                "int4_text": it,
                "int4_meta": int4_meta.get(ex.example_id),
            }
        )

    cat_summary = {}
    for cat, b in sorted(by_cat.items()):
        cat_summary[cat] = {
            "n": max(len(b["full"]), len(b["fp8"]), len(b["int4"])),
            "fullkv_mean": _mean(b["full"]),
            "fp8_mean": _mean(b["fp8"]),
            "int4_mean": _mean(b["int4"]),
            "delta_fp8_minus_full": _mean(b["fp8"]) - _mean(b["full"]) if b["fp8"] and b["full"] else None,
            "delta_int4_minus_full": _mean(b["int4"]) - _mean(b["full"]) if b["int4"] and b["full"] else None,
        }

    all_full = [d["fullkv_score"] for d in detail if d["fullkv_text"]]
    all_fp8 = [d["fp8_score"] for d in detail if d["fp8_text"]]
    all_int4 = [d["int4_score"] for d in detail if d["int4_text"]]

    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(detail),
        "modes": modes,
        "fullkv_mean": _mean(all_full),
        "fp8_mean": _mean(all_fp8),
        "int4_mean": _mean(all_int4),
        "delta_fp8_minus_full": (_mean(all_fp8) - _mean(all_full)) if all_fp8 and all_full else None,
        "delta_int4_minus_full": (_mean(all_int4) - _mean(all_full)) if all_int4 and all_full else None,
        "by_category": cat_summary,
        "seconds": {"fullkv": t_full, "fp8": t_fp8, "int4": t_int4},
        "selected_ids": [d["example_id"] for d in detail],
        "int4_modes_seen": sorted(
            { (d.get("int4_meta") or {}).get("mode") for d in detail if d.get("int4_meta") }
            - {None}
        ),
        "rows": detail,
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "w2c_pb_quality"
            if scratch
            else root / "runs" / "w2c_pb_quality"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
