"""P1: matched-budget attention baselines vs structure on PriorityBench stress.

Arms: FullKV · structure · SnapKV · H2O (ObservedAttention) · PyramidKV · hybrid
(structure-protected ∪ SnapKV residual). Same keep_frac / compression_ratio.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from prioritybench.pins import qwen3_chat_template_kwargs
from prioritybench.scoring import score_example
from prioritykv.baselines.attn_press import (
    AttnPressConfig,
    compression_ratio_for_keep,
    make_h2o_press,
    make_pyramid_press,
    make_snapkv_press,
    press_status,
)
from prioritykv.baselines.hybrid_press import (
    make_hybrid_press,
    protected_indices_from_roles,
)
from prioritykv.baselines.keep_policy import KeepPolicyConfig, assign_token_roles
from prioritykv.baselines.keep_policy_run import run_transformers_keep_policy
from prioritykv.baselines.kvpress_run import run_transformers_kvpress
from prioritykv.bench_pilot import _mean, materialize_examples
from prioritykv.fp8_baseline import _run_vllm_mode
from prioritykv.fullkv_compare import PromptRow, resolve_model_path
from prioritykv.stress_pilot import select_stress_rows


def _arm_rows(
    examples,
    full_texts: dict[str, str],
    outs: list[tuple[str, list[int], dict[str, Any]]] | None,
    *,
    error: str | None,
) -> dict[str, Any]:
    detail = []
    by_cat: dict[str, dict[str, list[float]]] = {}
    if outs is None:
        return {
            "mean": None,
            "fullkv_mean": _mean(
                [float(score_example(ex, full_texts[ex.example_id])) for ex in examples]
            ),
            "error": error,
            "rows": [],
        }
    for ex, (txt, _, meta) in zip(examples, outs, strict=True):
        cat = ex.category.value
        ft = full_texts[ex.example_id]
        sf = float(score_example(ex, ft))
        sp = float(score_example(ex, txt))
        by_cat.setdefault(cat, {"full": [], "pol": []})
        by_cat[cat]["full"].append(sf)
        by_cat[cat]["pol"].append(sp)
        detail.append(
            {
                "example_id": ex.example_id,
                "category": cat,
                "context_length": ex.context_length,
                "replication_slice": (ex.meta or {}).get("replication_slice"),
                "fullkv_score": sf,
                "policy_score": sp,
                "fullkv_pass": sf >= 1.0,
                "policy_pass": sp >= 1.0,
                "fullkv_text": ft,
                "policy_text": txt,
                "meta": meta,
            }
        )
    return {
        "mean": _mean([d["policy_score"] for d in detail]),
        "fullkv_mean": _mean([d["fullkv_score"] for d in detail]),
        "delta_minus_full": _mean(
            [d["policy_score"] - d["fullkv_score"] for d in detail]
        ),
        "by_category": {
            c: {
                "n": len(v["pol"]),
                "fullkv_mean": _mean(v["full"]),
                "policy_mean": _mean(v["pol"]),
            }
            for c, v in sorted(by_cat.items())
        },
        "error": error,
        "rows": detail,
    }


def run_attn_baselines(
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
    cr = float(cfg.get("compression_ratio", compression_ratio_for_keep(keep_frac)))
    press_cfg = AttnPressConfig(
        keep_frac=keep_frac,
        compression_ratio=cr,
        window_size=int(cfg.get("snapkv_window_size", 64)),
        kernel_size=int(cfg.get("snapkv_kernel_size", 5)),
        h2o_attn_implementation=str(cfg.get("h2o_attn_implementation", "eager")),
    )
    keep_cfg = KeepPolicyConfig(
        keep_frac=keep_frac,
        sink_tokens=int(cfg.get("sink_tokens", 16)),
        force_recent=int(cfg.get("force_recent", 128)),
        seed=int(cfg.get("seed", 0)),
        page_tokens=int(cfg.get("page_tokens", 16)),
        granularity=str(cfg.get("granularity", "token")),
    )
    arms_wanted = list(
        cfg.get(
            "arms",
            ["structure", "snapkv", "h2o", "pyramid", "hybrid"],
        )
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

    arms: dict[str, Any] = {}
    seconds: dict[str, float] = {"fullkv": t_full}
    status = press_status()

    if "structure" in arms_wanted:
        t1 = time.time()
        try:
            outs = run_transformers_keep_policy(
                model_path,
                prompts,
                max_new,
                policy="structure",
                keep_cfg=keep_cfg,
                max_model_len=int(vcfg["max_model_len"]),
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            outs, err = None, f"{type(exc).__name__}: {exc}"
            print(f"[attn_baselines] structure FAIL: {err}", flush=True)
        seconds["structure"] = time.time() - t1
        arms["structure"] = _arm_rows(examples, full_texts, outs, error=err)

    def _run_press(name: str, press, *, attn_impl: str, per_prompt=None) -> None:
        t1 = time.time()
        try:
            if press is None and per_prompt is None:
                raise RuntimeError(f"{name} press unavailable; kvpress installed? {status}")
            outs = run_transformers_kvpress(
                model_path,
                prompts,
                max_new,
                press=press,
                mode=name,
                max_model_len=int(vcfg["max_model_len"]),
                attn_implementation=attn_impl,
                per_prompt_press=per_prompt,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            outs, err = None, f"{type(exc).__name__}: {exc}"
            print(f"[attn_baselines] {name} FAIL: {err}", flush=True)
        seconds[name] = time.time() - t1
        arms[name] = _arm_rows(examples, full_texts, outs, error=err)

    if "snapkv" in arms_wanted:
        _run_press("snapkv", make_snapkv_press(press_cfg), attn_impl="sdpa")

    if "h2o" in arms_wanted:
        # ObservedAttention needs eager full attn maps — OOM on ~16k. Prefer 8k-only.
        h2o_max = int(cfg.get("h2o_max_prompt_tokens", 9000))
        h2o_prompts = []
        h2o_examples = []
        for ex, pr in zip(examples, prompts, strict=True):
            # context_length is the nominal bench length; skip long prompts for eager H2O.
            if int(ex.context_length) <= h2o_max:
                h2o_prompts.append(pr)
                h2o_examples.append(ex)
        t1 = time.time()
        try:
            if not h2o_prompts:
                raise RuntimeError(f"no ≤{h2o_max}-token examples for H2O (eager) arm")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            outs = run_transformers_kvpress(
                model_path,
                h2o_prompts,
                max_new,
                press=make_h2o_press(press_cfg),
                mode="h2o",
                max_model_len=int(vcfg["max_model_len"]),
                attn_implementation=press_cfg.h2o_attn_implementation,
            )
            # Align to full example list: None outs for skipped 16k.
            by_id = {
                ex.example_id: o for ex, o in zip(h2o_examples, outs, strict=True)
            }
            aligned = []
            for ex in examples:
                if ex.example_id in by_id:
                    aligned.append(by_id[ex.example_id])
                else:
                    aligned.append(
                        (
                            "",
                            [],
                            {
                                "mode": "h2o",
                                "skipped": True,
                                "reason": f"context_length={ex.context_length}>{h2o_max}",
                            },
                        )
                    )
            # Score only non-skipped for arm mean; still store rows.
            err = None
            detail = []
            scores = []
            for ex, (txt, _, meta) in zip(examples, aligned, strict=True):
                ft = full_texts[ex.example_id]
                sf = float(score_example(ex, ft))
                if meta.get("skipped"):
                    sp = None
                else:
                    sp = float(score_example(ex, txt))
                    scores.append(sp)
                detail.append(
                    {
                        "example_id": ex.example_id,
                        "category": ex.category.value,
                        "context_length": ex.context_length,
                        "replication_slice": (ex.meta or {}).get("replication_slice"),
                        "fullkv_score": sf,
                        "policy_score": sp,
                        "fullkv_pass": sf >= 1.0,
                        "policy_pass": (sp is not None and sp >= 1.0),
                        "fullkv_text": ft,
                        "policy_text": txt,
                        "meta": meta,
                    }
                )
            arms["h2o"] = {
                "mean": _mean(scores) if scores else None,
                "fullkv_mean": _mean(
                    [float(score_example(ex, full_texts[ex.example_id])) for ex in examples]
                ),
                "n_scored": len(scores),
                "n_skipped": len(examples) - len(scores),
                "delta_minus_full": (
                    _mean(
                        [
                            float(d["policy_score"]) - float(d["fullkv_score"])
                            for d in detail
                            if d["policy_score"] is not None
                        ]
                    )
                    if scores
                    else None
                ),
                "error": err,
                "rows": detail,
            }
        except Exception as exc:  # noqa: BLE001
            outs, err = None, f"{type(exc).__name__}: {exc}"
            print(f"[attn_baselines] h2o FAIL: {err}", flush=True)
            arms["h2o"] = _arm_rows(examples, full_texts, None, error=err)
        seconds["h2o"] = time.time() - t1

    if "pyramid" in arms_wanted:
        _run_press("pyramid", make_pyramid_press(press_cfg), attn_impl="sdpa")

    if "hybrid" in arms_wanted:
        chat_kwargs = dict(qwen3_chat_template_kwargs())

        def _hybrid_press(pr: PromptRow, n: int, tok) -> Any:
            roles = assign_token_roles(tok, pr.messages, chat_kwargs=chat_kwargs)
            if len(roles) > n:
                roles = roles[-n:]
            elif len(roles) < n:
                # Truncation path: pad filler roles at front.
                from prioritykv.page_roles import PageRole

                roles = [PageRole.FILLER] * (n - len(roles)) + list(roles)
            prot = protected_indices_from_roles(
                roles,
                n=n,
                sink_tokens=keep_cfg.sink_tokens,
                force_recent=keep_cfg.force_recent,
            )
            return make_hybrid_press(
                compression_ratio=cr,
                protected=prot,
                window_size=press_cfg.window_size,
                kernel_size=press_cfg.kernel_size,
            )

        _run_press("hybrid", None, attn_impl="sdpa", per_prompt=_hybrid_press)

    # Flat example_rows for stats.
    example_rows: list[dict[str, Any]] = []
    for ex in examples:
        eid = ex.example_id
        ft = full_texts[eid]
        sf = float(score_example(ex, ft))
        row: dict[str, Any] = {
            "example_id": eid,
            "category": ex.category.value,
            "context_length": ex.context_length,
            "replication_slice": (ex.meta or {}).get("replication_slice"),
            "fullkv_score": sf,
            "fullkv_pass": sf >= 1.0,
        }
        for name, arm in arms.items():
            for d in arm.get("rows") or []:
                if d["example_id"] == eid:
                    row[f"{name}_score"] = d["policy_score"]
                    row[f"{name}_pass"] = d["policy_pass"]
                    break
        example_rows.append(row)

    # Headline decision vs structure.
    struct_m = (arms.get("structure") or {}).get("mean")
    comparisons = {}
    for name in ("snapkv", "h2o", "pyramid", "hybrid"):
        m = (arms.get(name) or {}).get("mean")
        if struct_m is None or m is None:
            comparisons[f"{name}_vs_structure"] = None
        else:
            comparisons[f"{name}_vs_structure"] = float(m) - float(struct_m)

    hybrid_m = (arms.get("hybrid") or {}).get("mean")
    snap_m = (arms.get("snapkv") or {}).get("mean")
    if (
        hybrid_m is not None
        and struct_m is not None
        and snap_m is not None
        and hybrid_m > max(struct_m, snap_m) + 1e-9
    ):
        decision = "P1_HYBRID_WINS — hybrid > structure and SnapKV at matched budget"
    elif struct_m is not None and snap_m is not None and struct_m > snap_m + 1e-9:
        decision = "P1_STRUCTURE_BEATS_SNAPKV — structure > SnapKV at matched keep_frac"
    elif snap_m is not None and struct_m is not None and snap_m + 1e-9 >= struct_m:
        decision = "P1_SNAPKV_MATCHES — SnapKV ≥ structure (honest negative / revise claim)"
    else:
        decision = "P1_PARTIAL — incomplete arms or errors; see arms.*.error"

    result: dict[str, Any] = {
        "manifest_id": cfg.get("manifest_id", "p1_attn_baselines"),
        "rev": cfg.get("rev", 1),
        "model_path": model_path,
        "n": len(examples),
        "keep_frac": keep_frac,
        "compression_ratio": cr,
        "selection": cfg.get("selection"),
        "press_status": status,
        "fullkv_mean": _mean(
            [float(score_example(ex, full_texts[ex.example_id])) for ex in examples]
        ),
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in arms.items()
        },
        "arms_detail": arms,
        "example_rows": example_rows,
        "comparisons_mean_delta": comparisons,
        "decision": decision,
        "seconds": seconds,
        "selected_ids": [ex.example_id for ex in examples],
    }

    scratch = os.environ.get("PRIORITYKV_SCRATCH")
    if out_path is None:
        out_dir = (
            Path(scratch) / "runs" / "attn_baselines"
            if scratch
            else root / "runs" / "attn_baselines"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{result['manifest_id']}_r{result['rev']}.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
