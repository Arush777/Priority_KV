"""FullKV generation compare: Transformers vs vLLM (greedy).

Used for W1 gate G0 — backends must agree on a small prompt set.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptRow:
    id: str
    messages: list[dict[str, str]]


@dataclass
class CompareRow:
    id: str
    hf_text: str
    vllm_text: str
    exact: bool
    token_agree: float
    hf_tokens: list[int]
    vllm_tokens: list[int]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_prompts(path: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(PromptRow(id=obj["id"], messages=obj["messages"]))
    return rows


def resolve_model_path(manifest: dict[str, Any]) -> str:
    env = os.environ.get("MODEL_PATH") or os.environ.get("PRIORITYKV_MODEL")
    if env:
        return env
    scratch = os.environ.get("PRIORITYKV_SCRATCH", "")
    if scratch:
        cand = Path(scratch) / "models" / manifest["model"]["local_dirname"]
        if cand.exists():
            return str(cand)
    # hub id + revision as fallback (needs network)
    return manifest["model"]["hub_id"]


def _apply_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    from prioritybench.pins import qwen3_chat_template_kwargs

    kwargs = dict(qwen3_chat_template_kwargs())
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def run_transformers(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    revision: str | None,
) -> list[tuple[str, list[int]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    out: list[tuple[str, list[int]]] = []
    for row in prompts:
        text = _apply_chat(tok, row.messages)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_ids = gen[0, inputs["input_ids"].shape[-1] :].tolist()
        new_text = tok.decode(new_ids, skip_special_tokens=True)
        out.append((new_text, new_ids))
    del model
    torch.cuda.empty_cache()
    return out


def run_vllm(
    model_path: str,
    prompts: list[PromptRow],
    max_new_tokens: int,
    revision: str | None,
    tp: int,
    gpu_mem: float,
    max_model_len: int,
) -> list[tuple[str, list[int]]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision if not Path(model_path).exists() else None,
        trust_remote_code=True,
    )
    llm = LLM(
        model=model_path,
        revision=revision if not Path(model_path).exists() else None,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    texts = [_apply_chat(tok, r.messages) for r in prompts]
    outputs = llm.generate(texts, sampling)
    out: list[tuple[str, list[int]]] = []
    for o in outputs:
        seq = o.outputs[0]
        out.append((seq.text, list(seq.token_ids)))
    return out


def token_agree(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    hit = sum(1 for i in range(n) if a[i] == b[i])
    # penalize length mismatch lightly
    return hit / max(len(a), len(b))


def compare(
    manifest_path: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    root = manifest_path.resolve().parents[1]
    prompts_path = root / manifest["prompts"]
    prompts = load_prompts(prompts_path)
    model_path = resolve_model_path(manifest)
    revision = manifest["model"].get("revision")
    # local checkout already pins revision; don't pass hub revision to local path
    rev_arg = None if Path(model_path).exists() else revision
    max_new = int(manifest["decode"]["max_new_tokens"])
    vcfg = manifest["vllm"]

    t0 = time.time()
    hf = run_transformers(model_path, prompts, max_new, rev_arg)
    t_hf = time.time() - t0

    t1 = time.time()
    vl = run_vllm(
        model_path,
        prompts,
        max_new,
        rev_arg,
        tp=int(vcfg["tensor_parallel_size"]),
        gpu_mem=float(vcfg["gpu_memory_utilization"]),
        max_model_len=int(vcfg["max_model_len"]),
    )
    t_vllm = time.time() - t1

    rows: list[CompareRow] = []
    for p, (ht, hid), (vt, vid) in zip(prompts, hf, vl, strict=True):
        rows.append(
            CompareRow(
                id=p.id,
                hf_text=ht,
                vllm_text=vt,
                exact=(ht == vt),
                token_agree=token_agree(hid, vid),
                hf_tokens=hid,
                vllm_tokens=vid,
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
        "model_path": model_path,
        "n": len(rows),
        "exact_match_rate": exact_rate,
        "mean_token_agree": mean_agree,
        "passed": passed,
        "seconds": {"transformers": t_hf, "vllm": t_vllm},
        "rows": [asdict(r) for r in rows],
    }

    if out_path is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        if scratch:
            out_dir = Path(scratch) / "runs" / "w1_fullkv"
        else:
            out_dir = root / "runs" / "w1_fullkv"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{manifest['manifest_id']}_r{manifest['rev']}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    result["out_path"] = str(out_path)
    return result
