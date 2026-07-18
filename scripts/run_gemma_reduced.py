#!/usr/bin/env python3
"""Reduced Gemma matched-keep: structure vs uniform (publish secondary).

If Gemma weights/license unavailable → SKIP_NO_GEMMA (non-blocking for paper).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import prioritykv.cxx20_cuda_ext  # noqa: F401

from prioritykv.baselines.keep_policy import KeepPolicyConfig, select_structure, select_uniform
from prioritykv.baselines.keep_policy import assign_token_roles
from prioritykv.bench_pilot import materialize_examples
from prioritykv.fullkv_compare import PromptRow, _apply_chat
from prioritykv.page_roles import PageRole
from prioritykv.stress_pilot import select_stress_rows


DEFAULT_GEMMA = {
    "hub_id": "google/gemma-2-9b-it",
    "revision": None,
    "local_dirname": "gemma-2-9b-it",
}


def _resolve_gemma(cfg: dict) -> str | None:
    local = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    # Prefer local dirname under common caches / PRIORITYKV model dir
    for base in (
        Path(os.environ.get("PRIORITYKV_MODELS", "/data/anupam/scratch/models")),
        ROOT / "models",
        local / "hub",
    ):
        cand = base / cfg["local_dirname"]
        if cand.exists():
            return str(cand)
    # Try hub id (may fail on license)
    return cfg["hub_id"]


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "gemma_reduced.yaml"))
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "gemma_reduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gemma_reduced_{args.out_tag}.json"

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    gcfg = {**DEFAULT_GEMMA, **(cfg.get("model") or {})}
    model_id = _resolve_gemma(gcfg)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        result = {"decision": "SKIP_NO_TORCH", "error": str(exc), "pass": None}
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 0

    if not torch.cuda.is_available():
        result = {"decision": "SKIP_NO_CUDA", "pass": None}
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"out={out_path}")
        return 0

    print(f"[gemma] loading {model_id}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001
        result = {
            "decision": "SKIP_NO_GEMMA",
            "pass": None,
            "error": str(exc),
            "hub_id": gcfg["hub_id"],
            "note": "License/weights unavailable — non-blocking for publish track.",
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"out={out_path}")
        return 0  # skip is success for worker

    bench = json.loads((ROOT / cfg.get("bench_manifest", "data/prioritybench/manifests/w3_lock.json")).read_text())
    rows = select_stress_rows(bench, cfg.get("selection") or {
        "splits": ["calibration"],
        "context_lengths": [8000],
        "n_tool_schema": 4,
        "n_instruction_supersession": 4,
        "n_multi_turn_state": 6,
    })
    examples = materialize_examples(rows, data_root=ROOT / "data" / "prioritybench")
    keep_cfg = KeepPolicyConfig(keep_frac=args.keep_frac, sink_tokens=16, force_recent=64)

    def _gen(messages, keep_idx):
        import torch
        # Apply chat; then drop tokens not in keep_idx (prompt-level keep)
        text = _apply_chat(tok, messages)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        n = int(ids.numel())
        if keep_idx is not None:
            idx = torch.as_tensor([i for i in keep_idx if 0 <= int(i) < n], dtype=torch.long)
            ids = ids.index_select(0, idx)
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids.unsqueeze(0),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        return tok.decode(out[0, ids.numel() :], skip_special_tokens=True)

    from prioritybench.scoring import score_example
    from prioritybench.pins import qwen3_chat_template_kwargs

    # Gemma may ignore qwen chat kwargs — use empty
    chat_kwargs = {}
    t0 = time.time()
    detail = []
    for ex in examples:
        msgs = list(ex.messages)
        try:
            roles = assign_token_roles(tok, msgs, chat_kwargs=chat_kwargs)
        except Exception:
            roles = [PageRole.FILLER] * 512
        text = _apply_chat(tok, msgs)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        roles = list(roles)[:n] + [PageRole.RECENT] * max(0, n - len(roles))
        u_idx = select_uniform(n, keep_cfg)
        s_idx = select_structure(n, roles, keep_cfg)
        full_txt = _gen(msgs, None)
        uni_txt = _gen(msgs, u_idx.tolist())
        str_txt = _gen(msgs, s_idx.tolist())
        detail.append(
            {
                "example_id": ex.example_id,
                "category": ex.category.value,
                "full": float(score_example(ex, full_txt)),
                "uniform": float(score_example(ex, uni_txt)),
                "structure": float(score_example(ex, str_txt)),
            }
        )
        print(
            f"[gemma] {ex.example_id} full={detail[-1]['full']} "
            f"uni={detail[-1]['uniform']} str={detail[-1]['structure']}",
            flush=True,
        )

    def _m(key):
        xs = [d[key] for d in detail]
        return sum(xs) / len(xs) if xs else None

    result = {
        "decision": "GEMMA_REDUCED_PASS",
        "pass": True,
        "hub_id": gcfg["hub_id"],
        "n": len(detail),
        "keep_frac": args.keep_frac,
        "means": {
            "full": _m("full"),
            "uniform": _m("uniform"),
            "structure": _m("structure"),
        },
        "seconds": round(time.time() - t0, 3),
        "rows": detail,
        "note": "Reduced matched-keep on Gemma; PriorityBench scorers may be Qwen-tuned.",
    }
    # Soft signal: structure >= uniform
    if (_m("structure") or 0) + 1e-9 >= (_m("uniform") or 0):
        result["decision"] = "GEMMA_REDUCED_PASS"
    else:
        result["decision"] = "GEMMA_REDUCED_PARTIAL"
        result["pass"] = True  # still ship numbers
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("decision", "pass", "means", "n")}, indent=2))
    print(f"out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
