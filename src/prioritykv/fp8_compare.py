"""W1 FP8 KV path: vLLM FullKV vs kv_cache_dtype=fp8 on the same prompts.

Modes
-----
- uncalibrated (default): base BF16 weights + FP8 KV scales ≈ 1.0
- calibrated: load a checkpoint produced by scripts/cal_fp8.py (llmcompressor)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from prioritykv.fullkv_compare import (
    PromptRow,
    load_prompts,
    resolve_model_path,
    token_agree,
)


@dataclass
class Fp8Row:
    id: str
    full_text: str
    fp8_text: str
    exact: bool
    token_agree: float


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _chat_texts(model_path: str, prompts: list[PromptRow]) -> list[str]:
    from transformers import AutoTokenizer

    from prioritybench.pins import qwen3_chat_template_kwargs

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = dict(qwen3_chat_template_kwargs())
    return [
        tok.apply_chat_template(
            p.messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        for p in prompts
    ]


def _vllm_generate(
    *,
    model_path: str,
    texts: list[str],
    max_new_tokens: int,
    kv_cache_dtype: str | None,
    tp: int,
    gpu_mem: float,
    max_model_len: int,
) -> list[tuple[str, list[int]]]:
    from vllm import LLM, SamplingParams

    kwargs: dict[str, Any] = dict(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    llm = LLM(**kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    outputs = llm.generate(texts, sampling)
    out: list[tuple[str, list[int]]] = []
    for o in outputs:
        seq = o.outputs[0]
        out.append((seq.text, list(seq.token_ids)))
    # Drop engine so the next dtype config can allocate cleanly.
    del llm
    return out


def compare_fp8(
    manifest_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    root = manifest_path.resolve().parents[1]
    prompts = load_prompts(root / manifest["prompts"])
    model_path = os.environ.get("FP8_MODEL_PATH") or resolve_model_path(manifest)
    # Optional calibrated checkpoint overrides base weights path.
    fp8_model = os.environ.get("FP8_CAL_MODEL") or manifest.get("fp8_model") or model_path
    max_new = int(manifest["decode"]["max_new_tokens"])
    vcfg = manifest["vllm"]
    kv_dtype = manifest.get("fp8", {}).get("kv_cache_dtype", "fp8")

    texts = _chat_texts(model_path, prompts)

    t0 = time.time()
    full = _vllm_generate(
        model_path=model_path,
        texts=texts,
        max_new_tokens=max_new,
        kv_cache_dtype=None,
        tp=int(vcfg["tensor_parallel_size"]),
        gpu_mem=float(vcfg["gpu_memory_utilization"]),
        max_model_len=int(vcfg["max_model_len"]),
    )
    t_full = time.time() - t0

    # Rebuild prompt texts from fp8_model tokenizer if different directory.
    texts_fp8 = texts if fp8_model == model_path else _chat_texts(fp8_model, prompts)

    t1 = time.time()
    fp8 = _vllm_generate(
        model_path=fp8_model,
        texts=texts_fp8,
        max_new_tokens=max_new,
        kv_cache_dtype=kv_dtype,
        tp=int(vcfg["tensor_parallel_size"]),
        gpu_mem=float(vcfg["gpu_memory_utilization"]),
        max_model_len=int(vcfg["max_model_len"]),
    )
    t_fp8 = time.time() - t1

    rows: list[Fp8Row] = []
    for p, (ft, fid), (pt, pid) in zip(prompts, full, fp8, strict=True):
        rows.append(
            Fp8Row(
                id=p.id,
                full_text=ft,
                fp8_text=pt,
                exact=(ft == pt),
                token_agree=token_agree(fid, pid),
            )
        )

    exact_rate = sum(1 for r in rows if r.exact) / len(rows)
    mean_agree = sum(r.token_agree for r in rows) / len(rows)
    gate = manifest["gate"]
    # FP8 is allowed to diverge more than Transformers↔vLLM FullKV.
    passed = (
        exact_rate >= float(gate["min_exact_match"])
        and mean_agree >= float(gate["min_mean_token_agree"])
    )

    result = {
        "manifest_id": manifest["manifest_id"],
        "rev": manifest["rev"],
        "model_path": model_path,
        "fp8_model": fp8_model,
        "kv_cache_dtype": kv_dtype,
        "n": len(rows),
        "exact_match_rate": exact_rate,
        "mean_token_agree": mean_agree,
        "passed": passed,
        "seconds": {"fullkv": t_full, "fp8": t_fp8},
        "rows": [asdict(r) for r in rows],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        out_dir = (
            Path(scratch) / "runs" / "w1_fp8"
            if scratch
            else root / "runs" / "w1_fp8"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{manifest['manifest_id']}_r{manifest['rev']}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    result["out_path"] = str(out_path)
    return result
