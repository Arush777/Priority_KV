#!/usr/bin/env python3
"""CPU retention preflight: gold-token spans vs keep policies / sink+recent.

Replays structured_stress transforms + select_keep_indices (no GPU decode).
Reports whether gold state lands in kept indices — gates "ceiling" vs port-artifact
claims for Llama and leakage reads for structure>FullKV.

Usage:
  PYTHONPATH=src python scripts/audit_retention.py \\
    --config configs/w5_stress_s0_kf25_token.yaml \\
    --out jobs/results/audit_retention_qwen_s0_kf25.json

  PYTHONPATH=src python scripts/audit_retention.py \\
    --config configs/p3_llama31_attn_s0_kf25.yaml \\
    --out jobs/results/audit_retention_llama_s0_kf25.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import yaml

from prioritybench.pins import chat_template_kwargs_for_tokenizer
from prioritykv.baselines.buried_state import (
    _is_filler_turn,
    bury_short_state_turns,
    relocate_state_to_middle,
)
from prioritykv.baselines.keep_policy import (
    KeepPolicyConfig,
    assign_token_roles,
    select_keep_indices,
)
from prioritykv.bench_pilot import materialize_examples
from prioritykv.fullkv_compare import resolve_model_path
from prioritykv.stress_pilot import select_stress_rows


def _gold_message_indices(msgs: list[dict[str, str]]) -> list[int]:
    """Same gold partition as relocate_state_to_middle (non-filler body turns)."""
    n = len(msgs)
    lead = 0
    while lead < n and (msgs[lead].get("role") or "").lower() == "system":
        lead += 1
    if lead >= n - 1:
        return []
    body = msgs[lead:-1]
    return [lead + i for i, m in enumerate(body) if not _is_filler_turn(m)]


def _token_span_for_prefix(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    end_idx: int,
    chat_kwargs: dict,
    add_generation_prompt: bool,
) -> int:
    prefix = list(messages[: end_idx + 1])
    text = tokenizer.apply_chat_template(
        prefix,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_kwargs,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def gold_token_indices(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    chat_kwargs: dict,
    gold_markers: set[str] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Map gold messages (or <<STATE>> spans) to token indices in the full chat prompt.

    When ``gold_markers`` is set (buried mode), only matching ``<<STATE>>…<</STATE>>``
    substrings count — excludes system/filler turns that bury() also wraps.
    """
    n_msgs = len(messages)
    # Prefer explicit bury markers when present.
    full_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    n = len(full_ids)

    state_lo = "<<STATE>>"
    state_hi = "<</STATE>>"
    if state_lo in full_text and state_hi in full_text:
        # Approximate: locate marker substrings via encode of sliced text.
        gold: set[int] = set()
        spans: list[tuple[int, int]] = []
        start = 0
        while True:
            a = full_text.find(state_lo, start)
            if a < 0:
                break
            b = full_text.find(state_hi, a)
            if b < 0:
                break
            marked = full_text[a : b + len(state_hi)]
            if gold_markers is not None and marked not in gold_markers:
                start = b + len(state_hi)
                continue
            # Tokenize prefix lengths to get span bounds.
            pre = len(tokenizer(full_text[:a], add_special_tokens=False)["input_ids"])
            post = len(
                tokenizer(full_text[: b + len(state_hi)], add_special_tokens=False)[
                    "input_ids"
                ]
            )
            lo, hi = min(pre, n), min(post, n)
            if hi > lo:
                spans.append((lo, hi))
                gold.update(range(lo, hi))
            start = b + len(state_hi)
        if gold:
            return sorted(gold), spans

    gold_msgs = _gold_message_indices(messages)
    spans = []
    gold_set: set[int] = set()
    prev = 0
    for i, _msg in enumerate(messages):
        plen = _token_span_for_prefix(
            tokenizer,
            messages,
            end_idx=i,
            chat_kwargs=chat_kwargs,
            add_generation_prompt=(i == n_msgs - 1),
        )
        plen = min(max(plen, prev), n)
        if i in gold_msgs:
            spans.append((prev, plen))
            gold_set.update(range(prev, plen))
        prev = plen
    return sorted(gold_set), spans


def sink_recent_set(n: int, sink: int, recent: int) -> set[int]:
    return set(range(min(sink, n))) | set(range(max(0, n - recent), n))


def _mean(xs: list[float]) -> float | None:
    return float(statistics.fmean(xs)) if xs else None


def audit_config(config_path: Path, *, max_examples: int | None = None) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = json.loads((root / cfg["bench_manifest"]).read_text(encoding="utf-8"))
    rows = select_stress_rows(bench, cfg["selection"])
    examples = materialize_examples(rows, data_root=root / "data" / "prioritybench")
    if max_examples is not None:
        examples = examples[: max(0, max_examples)]

    use_buried = bool(cfg.get("buried_state", False))
    use_middle = bool(cfg.get("relocate_middle", False))
    middle_pos = float(cfg.get("relocate_position", 0.5))

    # Keep block: P0 uses cfg["keep"]; P3 flattens keep fields at top level.
    kraw = cfg.get("keep") or cfg
    keep_cfg = KeepPolicyConfig(
        keep_frac=float(kraw.get("keep_frac", 0.25)),
        sink_tokens=int(kraw.get("sink_tokens", 16)),
        force_recent=int(kraw.get("force_recent", 128)),
        seed=int(kraw.get("seed", 0)),
        page_tokens=int(kraw.get("page_tokens", 16)),
        granularity=str(kraw.get("granularity", "token")),
    )
    policies = list(
        (cfg.get("keep") or {}).get("policies")
        or ["uniform", "structure", "random"]
    )
    # Always include uniform+structure for the audit even if config is attn-only.
    for p in ("uniform", "structure"):
        if p not in policies:
            policies.append(p)

    model_path = resolve_model_path(cfg)
    revision = cfg.get("model", {}).get("revision")
    from transformers import AutoTokenizer

    tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if not Path(model_path).exists() and revision:
        tok_kwargs["revision"] = revision
    # Prefer local HF cache snapshot if hub id.
    tok = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    chat_kwargs = dict(chat_template_kwargs_for_tokenizer(tok))

    detail: list[dict[str, Any]] = []
    for ex_i, ex in enumerate(examples):
        msgs = list(ex.messages)
        seed = hash(ex.example_id) % 10_000
        # Capture pre-bury gold contents for marker filtering (Codex).
        gold_markers: set[str] | None = None
        if use_buried:
            gold_markers = {
                f"<<STATE>> {msgs[i].get('content') or ''} <</STATE>>"
                for i in _gold_message_indices(msgs)
            }
            msgs = bury_short_state_turns(msgs, seed=seed)
        if use_middle:
            msgs = relocate_state_to_middle(msgs, position=middle_pos, seed=seed)

        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        n = len(tok(full_text, add_special_tokens=False)["input_ids"])
        gold_ids, spans = gold_token_indices(
            tok, msgs, chat_kwargs=chat_kwargs, gold_markers=gold_markers
        )
        gold_set = set(gold_ids)
        sr = sink_recent_set(n, keep_cfg.sink_tokens, keep_cfg.force_recent)
        roles = assign_token_roles(tok, msgs, chat_kwargs=chat_kwargs)

        row: dict[str, Any] = {
            "example_id": ex.example_id,
            "category": getattr(ex.category, "value", str(ex.category)),
            "n_tokens": n,
            "n_gold_tokens": len(gold_set),
            "gold_token_spans": spans,
            "gold_in_sink_recent_frac": (
                len(gold_set & sr) / len(gold_set) if gold_set else None
            ),
            "gold_fully_in_sink_recent": (
                bool(gold_set) and gold_set.issubset(sr)
            ),
        }
        for pol in policies:
            if pol == "keep_all":
                kept = set(range(n))
            else:
                # Match production keep_policy_run: seed + row_index for random.
                pol_cfg = (
                    KeepPolicyConfig(
                        keep_frac=keep_cfg.keep_frac,
                        sink_tokens=keep_cfg.sink_tokens,
                        force_recent=keep_cfg.force_recent,
                        seed=keep_cfg.seed + ex_i,
                        page_tokens=keep_cfg.page_tokens,
                        granularity=keep_cfg.granularity,
                        risk_fit_path=keep_cfg.risk_fit_path,
                    )
                    if pol == "random"
                    else keep_cfg
                )
                kept = set(
                    select_keep_indices(
                        n, pol_cfg, policy=pol, roles=roles
                    ).tolist()
                )
            inter = gold_set & kept
            row[f"{pol}_gold_kept_frac"] = (
                len(inter) / len(gold_set) if gold_set else None
            )
            row[f"{pol}_gold_fully_evicted"] = bool(gold_set) and len(inter) == 0
        detail.append(row)

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in detail if r.get(key) is not None]

    summary: dict[str, Any] = {
        "config": str(config_path),
        "manifest_id": cfg.get("manifest_id"),
        "model_path": model_path,
        "n": len(detail),
        "buried_state": use_buried,
        "relocate_middle": use_middle,
        "keep_frac": keep_cfg.keep_frac,
        "sink_tokens": keep_cfg.sink_tokens,
        "force_recent": keep_cfg.force_recent,
        "mean_gold_in_sink_recent_frac": _mean(col("gold_in_sink_recent_frac")),
        "frac_examples_gold_fully_in_sink_recent": (
            sum(1 for r in detail if r.get("gold_fully_in_sink_recent")) / len(detail)
            if detail
            else None
        ),
        "mean_n_tokens": _mean([float(r["n_tokens"]) for r in detail]),
        "mean_n_gold_tokens": _mean(
            [float(r["n_gold_tokens"]) for r in detail if r["n_gold_tokens"]]
        ),
    }
    for pol in policies:
        kf = f"{pol}_gold_kept_frac"
        fe = f"{pol}_gold_fully_evicted"
        summary[f"mean_{kf}"] = _mean(col(kf))
        summary[f"frac_{fe}"] = (
            sum(1 for r in detail if r.get(fe)) / len(detail) if detail else None
        )

    # Verdict helpers for docs.
    gsr = summary["mean_gold_in_sink_recent_frac"]
    if gsr is None:
        verdict = "NO_GOLD_SPANS"
    elif gsr >= 0.5:
        verdict = "PORT_ARTIFACT_LIKELY — ≥50% of gold tokens already in sink+recent kept region"
    else:
        verdict = "GOLD_MOSTLY_EVICTABLE — majority of gold outside sink+recent"
    summary["verdict"] = verdict
    summary["rows"] = detail
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()
    # Allow HF hub downloads if local scratch missing.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    result = audit_config(args.config, max_examples=args.max_examples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Write summary without full rows first for humans? Keep rows — needed for checker.
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Also write a slim summary sibling.
    slim = {k: v for k, v in result.items() if k != "rows"}
    slim_path = args.out.with_name(args.out.stem + "_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(slim, indent=2))
    print(f"wrote {args.out} and {slim_path}")


if __name__ == "__main__":
    main()
