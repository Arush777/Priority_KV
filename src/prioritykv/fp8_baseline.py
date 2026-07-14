"""FP8 KV baseline (S1) helpers — W1 smoke + optional oneshot calib."""

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
    _apply_chat,
)


@dataclass
class Fp8Row:
    id: str
    full_text: str
    fp8_text: str
    exact: bool
    token_agree: float


def load_fp8_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _run_vllm_mode(
    model_path: str,
    prompts: list[PromptRow],
    *,
    max_new_tokens: int,
    kv_cache_dtype: str,
    calculate_kv_scales: bool,
    tp: int,
    gpu_mem: float,
    max_model_len: int,
) -> list[tuple[str, list[int]]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": gpu_mem,
        "max_model_len": max_model_len,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enforce_eager": True,
    }
    if kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
        kwargs["calculate_kv_scales"] = calculate_kv_scales

    llm = LLM(**kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    texts = [_apply_chat(tok, r.messages) for r in prompts]
    # Soft guard: truncate from the left (keep system + final ask) if over budget.
    budget = max_model_len - max_new_tokens - 8
    trimmed: list[str] = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            trimmed.append(t)
            continue
        # Keep tail (final instruction) + head (schemas/constraints).
        head = ids[: budget // 4]
        tail = ids[-(budget - len(head)) :]
        trimmed.append(tok.decode(head + tail, skip_special_tokens=False))
    outputs = llm.generate(trimmed, sampling)
    out: list[tuple[str, list[int]]] = []
    for o in outputs:
        seq = o.outputs[0]
        out.append((seq.text, list(seq.token_ids)))
    return out


def compare_fullkv_fp8(
    manifest_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Greedy FullKV (auto KV) vs FP8 KV with on-the-fly scale estimation."""
    root = manifest_path.resolve().parents[1]
    # allow either dedicated fp8 yaml or reuse w1_fullkv prompts/model
    manifest = load_fp8_manifest(manifest_path)
    prompts_path = root / manifest["prompts"]
    prompts = load_prompts(prompts_path)
    model_path = resolve_model_path(manifest)
    max_new = int(manifest["decode"]["max_new_tokens"])
    vcfg = manifest["vllm"]
    fp8 = manifest["fp8"]

    t0 = time.time()
    full = _run_vllm_mode(
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
    quant = _run_vllm_mode(
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

    rows: list[Fp8Row] = []
    for p, (ft, fid), (qt, qid) in zip(prompts, full, quant, strict=True):
        rows.append(
            Fp8Row(
                id=p.id,
                full_text=ft,
                fp8_text=qt,
                exact=(ft == qt),
                token_agree=token_agree(fid, qid),
            )
        )

    exact_rate = sum(1 for r in rows if r.exact) / len(rows)
    mean_agree = sum(r.token_agree for r in rows) / len(rows)
    gate = manifest["gate"]
    passed = (
        exact_rate >= float(gate["min_exact_match"])
        and mean_agree >= float(gate["min_mean_token_agree"])
    )

    result = {
        "manifest_id": manifest["manifest_id"],
        "rev": manifest["rev"],
        "mode": "vllm_fp8_calculate_kv_scales",
        "model_path": model_path,
        "n": len(rows),
        "exact_match_rate": exact_rate,
        "mean_token_agree": mean_agree,
        "passed": passed,
        "seconds": {"fullkv": t_full, "fp8": t_fp8},
        "rows": [asdict(r) for r in rows],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        if scratch:
            out_dir = Path(scratch) / "runs" / "w1_fp8"
        else:
            out_dir = root / "runs" / "w1_fp8"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{manifest['manifest_id']}_r{manifest['rev']}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    result["out_path"] = str(out_path)
    return result


def build_local_calib_messages(n: int = 64) -> list[dict[str, Any]]:
    """Offline calib examples (no hub download). Chat-style messages."""
    topics = [
        "scheduler logs",
        "invoice checks",
        "api telemetry",
        "cache pages",
        "tool schemas",
    ]
    out: list[dict[str, Any]] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        out.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise technical assistant.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Summarize {topic} batch {i} in two sentences. "
                            f"Include the id BK-{1000 + i}."
                        ),
                    },
                ]
            }
        )
    return out


def oneshot_calibrate_fp8(
    *,
    model_path: str,
    save_dir: Path,
    n_calib: int = 64,
    max_seq_len: int = 2048,
    strategy: str = "tensor",
) -> Path:
    """llm-compressor oneshot calib; writes a loadable directory under save_dir."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from prioritybench.pins import qwen3_chat_template_kwargs

    save_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def process(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            **qwen3_chat_template_kwargs(),
        )
        return tokenizer(
            text,
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    raw = build_local_calib_messages(n_calib)
    ds = Dataset.from_list(raw)
    ds = ds.map(process, remove_columns=ds.column_names)

    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    # Try Qwen3 attention target; fall back to KV-only if class name differs.
    targets = ["Qwen3Attention", "Qwen2Attention"]
    recipe = None
    last_err: Exception | None = None
    for tgt in targets:
        try:
            recipe = QuantizationModifier(
                config_groups={
                    "attention": QuantizationScheme(
                        targets=[tgt],
                        input_activations=fp8_args,
                    )
                },
                kv_cache_scheme=fp8_args,
            )
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=max_seq_len,
                num_calibration_samples=n_calib,
            )
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 — try next target
            last_err = e
            recipe = None

    if recipe is None:
        # KV-cache scheme only
        recipe = QuantizationModifier(kv_cache_scheme=fp8_args)
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_seq_len,
            num_calibration_samples=n_calib,
        )
        if last_err is not None:
            # kept for debugging in result sidecars
            (save_dir / "calib_fallback.txt").write_text(
                f"attention target failed: {last_err!r}; used kv_cache_scheme only\n"
            )

    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    meta = {
        "source_model": model_path,
        "n_calib": n_calib,
        "max_seq_len": max_seq_len,
        "strategy": strategy,
    }
    (save_dir / "prioritykv_fp8_meta.json").write_text(json.dumps(meta, indent=2))
    return save_dir
