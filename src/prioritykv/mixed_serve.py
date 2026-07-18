"""W6 systems job: byte-matched mixed BF16/INT4 KV — uniform vs structure.

FullKV vs uniform-INT4 vs structure-mixed at the SAME INT4 fraction, on the
mid-context stress set. Desired result: structure-mixed >> uniform-INT4 at equal
bytes (role-aware precision keeps tool/supersession/multi-turn reliable).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from prioritybench.scoring import score_example
from prioritykv.baselines.buried_state import (
    bury_short_state_turns,
    relocate_state_to_middle,
)
from prioritykv.bench_pilot import _mean, materialize_examples
from prioritykv.fullkv_compare import PromptRow, resolve_model_path
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.mixed_kv import MixedPlanConfig
from prioritykv.mixed_kv_run import run_transformers_mixed_kv
from prioritykv.stress_pilot import select_stress_rows


def run_mixed_serve(config_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Triple-GPU parent can restrict shard via env (same contract as D4 dual).
    only_ctx = os.environ.get("PRIORITYKV_ONLY_CONTEXT_LENGTH")
    if only_ctx:
        cfg = dict(cfg)
        sel = dict(cfg.get("selection") or {})
        sel["context_lengths"] = [int(only_ctx)]
        cfg["selection"] = sel
        print(f"[mixed_serve] ONLY_CONTEXT_LENGTH={only_ctx}", flush=True)
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    print(
        f"[mixed_serve] selected n={len(rows)} "
        f"contexts={sorted({int(r['context_length']) for r in rows})}",
        flush=True,
    )
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")

    use_buried = bool(cfg.get("buried_state", False))
    use_middle = bool(cfg.get("relocate_middle", False))
    middle_pos = float(cfg.get("relocate_position", 0.5))
    prompts = []
    for ex in examples:
        msgs = list(ex.messages)
        seed = hash(ex.example_id) % 10_000
        if use_buried:
            msgs = bury_short_state_turns(msgs, seed=seed)
        if use_middle:
            msgs = relocate_state_to_middle(msgs, position=middle_pos, seed=seed)
        prompts.append(PromptRow(id=ex.example_id, messages=msgs))

    model_path = resolve_model_path(cfg)
    max_new = int(cfg["decode"]["max_new_tokens"])
    vcfg = cfg["vllm"]
    mcfg = cfg.get("mixed", {})
    risk_path = mcfg.get("risk_fit_path")
    if risk_path:
        rp = Path(str(risk_path))
        risk_path = str(rp if rp.is_absolute() else root / rp)
    plan_cfg = MixedPlanConfig(
        int4_frac=float(mcfg.get("int4_frac", 0.75)),
        sink_tokens=int(mcfg.get("sink_tokens", 16)),
        recent_window=int(mcfg.get("recent_window", 128)),
        risk_fit_path=risk_path,
    )
    int4_cfg = Int4KvConfig(
        nbits=int(mcfg.get("nbits", 4)),
        group_size=int(mcfg.get("group_size", 32)),
    )
    degrade = str(mcfg.get("degrade", "int4")).lower()
    if degrade not in ("int4", "zero"):
        raise ValueError(f"mixed.degrade must be int4|zero, got {degrade}")
    storage = mcfg.get("storage")
    if storage is not None:
        storage = str(storage).lower()
        if storage not in ("packed", "fake"):
            raise ValueError(f"mixed.storage must be packed|fake, got {storage}")
    attn_backend = mcfg.get("attn_backend")
    if attn_backend is not None:
        attn_backend = str(attn_backend).lower()
        if attn_backend not in ("sdpa", "flashinfer"):
            raise ValueError(f"mixed.attn_backend must be sdpa|flashinfer, got {attn_backend}")
    fi_parity_every = int(mcfg.get("fi_parity_every", 1))
    fi_require_pass = bool(mcfg.get("fi_require_pass", True))
    policies = list(mcfg.get("policies", ["full", "uniform", "structure"]))

    full_texts: dict[str, str] = {}
    arms: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for policy in policies:
        t1 = time.time()
        outs = run_transformers_mixed_kv(
            model_path,
            prompts,
            max_new,
            policy=policy,
            plan_cfg=plan_cfg,
            int4_cfg=int4_cfg,
            degrade=degrade,
            storage=storage,
            attn_backend=attn_backend,
            fi_parity_every=fi_parity_every,
            fi_require_pass=fi_require_pass,
            max_model_len=int(vcfg["max_model_len"]),
        )
        seconds[policy] = time.time() - t1
        if policy == "full":
            for ex, (txt, _, _) in zip(examples, outs, strict=True):
                full_texts[ex.example_id] = txt
        detail = []
        by_cat: dict[str, dict[str, list[float]]] = {}
        int4_fracs: list[float] = []
        for ex, (txt, _, meta) in zip(examples, outs, strict=True):
            cat = ex.category.value
            sp = float(score_example(ex, txt))
            int4_fracs.append(float(meta.get("int4_frac_realized", 0.0)))
            by_cat.setdefault(cat, {"pol": []})
            by_cat[cat]["pol"].append(sp)
            detail.append(
                {
                    "example_id": ex.example_id,
                    "category": cat,
                    "context_length": ex.context_length,
                    "policy_score": sp,
                    "policy_text": txt,
                    "meta": meta,
                }
            )
        arms[policy] = {
            "mean": _mean([d["policy_score"] for d in detail]),
            "int4_frac_realized": _mean(int4_fracs),
            "by_category": {
                c: {"n": len(v["pol"]), "policy_mean": _mean(v["pol"])}
                for c, v in sorted(by_cat.items())
            },
            "rows": detail,
        }

    full_mean = _mean(
        [float(score_example(ex, full_texts.get(ex.example_id, ""))) for ex in examples]
    ) if full_texts else None
    for policy in policies:
        if full_mean is not None:
            arms[policy]["delta_minus_full"] = arms[policy]["mean"] - full_mean

    result = {
        "manifest_id": cfg["manifest_id"],
        "rev": cfg["rev"],
        "model_path": model_path,
        "n": len(examples),
        "buried_state": use_buried,
        "relocate_middle": use_middle,
        "mixed": {
            "int4_frac": plan_cfg.int4_frac,
            "sink_tokens": plan_cfg.sink_tokens,
            "recent_window": plan_cfg.recent_window,
            "nbits": int4_cfg.nbits,
            "group_size": int4_cfg.group_size,
            "degrade": degrade,
            "storage": storage,
            "attn_backend": attn_backend,
            "risk_fit_path": risk_path,
        },
        "policies": policies,
        "fullkv_mean": full_mean,
        "arms": {p: {k: v for k, v in arms[p].items() if k != "rows"} for p in policies},
        "arms_detail": arms,
        "seconds": seconds,
        "selected_ids": [ex.example_id for ex in examples],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "mixed_serve"
            if scratch
            else root / "runs" / "mixed_serve"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg['manifest_id']}_r{cfg['rev']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["out_path"] = str(out_path)
    return result
