"""Generation-free gold-span retention audit on public τ-bench trajectories.

This is **mechanistic evidence only**: it measures whether naturally occurring
schemas, identifiers, tool results, and constraints survive a retention policy.
It is *not* a τ-bench evaluation, it does not measure task success, and it never
runs the user simulator, a tool backend, or any generation.

Extraction rules below are frozen before any policy is applied, and the span
classes are defined by structure (message role, tool-call fields, regex over
imperative policy language) rather than by anything downstream of retention.

Only the deterministic, position/role-based policies (``structure``, ``uniform``,
``random``) can be audited on CPU. SnapKV scores tokens by realised attention and
therefore requires a GPU forward pass; it is deliberately out of scope here and
that limitation must be reported alongside the table.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SPAN_CLASSES: tuple[str, ...] = (
    "tool_name",
    "tool_call_argument",
    "reused_identifier",
    "reused_tool_result_value",
    "explicit_policy",
    "correction",
)

# Structure the policy can *see* from chat roles alone, vs content buried inside
# free-form prose. Drives the visible-vs-buried breakdown.
VISIBLE_CLASSES = frozenset({"tool_name", "tool_call_argument"})
BURIED_CLASSES = frozenset(
    {"reused_identifier", "reused_tool_result_value", "explicit_policy", "correction"}
)

_POLICY_LINE = re.compile(
    r"^\s*(?:[-*\d.)\s]*)?(?=.*\b(?:must|never|always|only|cannot|can't|should not|"
    r"do not|don't|required|prohibited|not allowed|may not)\b)(.{8,300})$",
    re.IGNORECASE | re.MULTILINE,
)

_CORRECTION = re.compile(
    r"\b(?:actually|instead|no,|wait,|sorry,|change that|scratch that|"
    r"never mind|nevermind|on second thought|correction|I meant|rather than|"
    r"cancel that|not that one)\b",
    re.IGNORECASE,
)

# Identifier-shaped literals: snake_case handles with digits, order/reservation
# codes, emails, and long digit runs.
_IDENTIFIER = re.compile(
    r"\b(?:[a-z]+_[a-z]+_\d{2,}|[A-Z]{2,}\d{4,}|#[A-Za-z0-9]{5,}|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\d{7,})\b"
)

_MIN_REUSED_VALUE_LEN = 4


@dataclass(frozen=True)
class Span:
    """One gold span, located by character offsets in the rendered transcript."""

    span_class: str
    text: str
    start: int
    end: int
    source_message_index: int
    source_role: str

    @property
    def is_visible_structure(self) -> bool:
        return self.span_class in VISIBLE_CLASSES


@dataclass
class Trajectory:
    traj_id: str
    task_name: str
    source_model: str
    messages: list[dict[str, Any]]
    is_correct: bool | None = None


@dataclass
class RenderedTrajectory:
    trajectory: Trajectory
    text: str
    # (start, end) char offsets of each message in ``text``
    message_offsets: list[tuple[int, int]]
    spans: list[Span] = field(default_factory=list)


def load_trajectories(
    dataset_dir: str | Path,
    *,
    files: Sequence[str] | None = None,
    limit_per_file: int | None = None,
) -> list[Trajectory]:
    """Load public τ-bench trajectories from a pinned local snapshot."""
    root = Path(dataset_dir)
    paths = (
        [root / f for f in files] if files else sorted(root.glob("*.jsonl"))
    )
    out: list[Trajectory] = []
    for path in paths:
        if not path.is_file():
            continue
        source_model = path.stem
        with path.open() as fh:
            for i, line in enumerate(fh):
                if limit_per_file is not None and i >= limit_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = row.get("meta") or {}
                out.append(
                    Trajectory(
                        traj_id=f"{source_model}::{meta.get('id', i)}",
                        task_name=row.get("task_name", "unknown"),
                        source_model=source_model,
                        messages=row.get("messages") or [],
                        is_correct=meta.get("is_correct"),
                    )
                )
    return out


def _message_text(msg: dict[str, Any]) -> str:
    """Flatten one message to the text a KV cache would actually hold."""
    parts: list[str] = []
    content = msg.get("content")
    if content:
        parts.append(str(content))
    for call in msg.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if name:
            parts.append(f"{name}({args if args is not None else ''})")
    return "\n".join(parts)


def render(trajectory: Trajectory) -> RenderedTrajectory:
    """Concatenate messages into one transcript, tracking per-message offsets."""
    chunks: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for msg in trajectory.messages:
        body = f"{msg.get('role', '?')}: {_message_text(msg)}\n"
        chunks.append(body)
        offsets.append((cursor, cursor + len(body)))
        cursor += len(body)
    return RenderedTrajectory(
        trajectory=trajectory, text="".join(chunks), message_offsets=offsets
    )


def _add(spans: list[Span], cls: str, text: str, start: int, end: int, i: int, role: str):
    """Record a span, trimming surrounding whitespace from *both* text and offsets.

    Offsets must always slice back to exactly ``text``; the token-alignment pass
    and the manual audit both depend on that invariant.
    """
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    start += lead
    end -= trail
    text = text.strip()
    if end > start and text:
        spans.append(Span(cls, text, start, end, i, role))


def _find_in(haystack: str, needle: str, base: int) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    if not needle:
        return hits
    pos = haystack.find(needle)
    while pos != -1:
        hits.append((base + pos, base + pos + len(needle)))
        pos = haystack.find(needle, pos + 1)
    return hits


def extract_spans(rendered: RenderedTrajectory) -> list[Span]:
    """Apply the frozen extraction rules to one rendered trajectory."""
    spans: list[Span] = []
    messages = rendered.trajectory.messages
    text = rendered.text

    tool_result_values: list[tuple[int, str]] = []
    identifier_hits: dict[str, list[tuple[int, int, int, str]]] = {}

    for i, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        start, end = rendered.message_offsets[i]
        body = text[start:end]

        # 1/2. Tool names and call arguments, straight from the structured field.
        for call in msg.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            name = fn.get("name")
            if name:
                for s, e in _find_in(body, str(name), start):
                    _add(spans, "tool_name", str(name), s, e, i, role)
            raw_args = fn.get("arguments")
            if raw_args:
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                values = (
                    [v for v in parsed.values()] if isinstance(parsed, dict) else []
                )
                for v in values:
                    sv = str(v)
                    if len(sv) < _MIN_REUSED_VALUE_LEN:
                        continue
                    for s, e in _find_in(body, sv, start)[:1]:
                        _add(spans, "tool_call_argument", sv, s, e, i, role)

        # 5. Explicit policy lines (system/developer prose).
        if role in ("system", "developer"):
            for m in _POLICY_LINE.finditer(body):
                _add(spans, "explicit_policy", m.group(1).strip(),
                     start + m.start(1), start + m.end(1), i, role)

        # 6. Conversational corrections / superseding constraints.
        if role == "user" and _CORRECTION.search(body):
            _add(spans, "correction", body, start, end, i, role)

        # Collect for the reuse passes.
        if role == "tool" and msg.get("content"):
            tool_result_values.append((i, str(msg["content"])))
        for m in _IDENTIFIER.finditer(body):
            identifier_hits.setdefault(m.group(0), []).append(
                (start + m.start(), start + m.end(), i, role)
            )

    # 3. Identifiers that are actually reused later (≥2 occurrences).
    for ident, hits in identifier_hits.items():
        if len(hits) < 2:
            continue
        for s, e, i, role in hits:
            _add(spans, "reused_identifier", ident, s, e, i, role)

    # 4. Values produced by a tool result and referenced again downstream.
    for msg_index, content in tool_result_values:
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        for value in _flatten_scalars(parsed):
            sv = str(value)
            if len(sv) < _MIN_REUSED_VALUE_LEN:
                continue
            after = rendered.message_offsets[msg_index][1]
            if text.find(sv, after) == -1:
                continue
            s0, e0 = rendered.message_offsets[msg_index]
            for s, e in _find_in(text[s0:e0], sv, s0)[:1]:
                _add(spans, "reused_tool_result_value", sv, s, e, msg_index, "tool")

    return spans


def _flatten_scalars(obj: Any, depth: int = 0) -> Iterable[Any]:
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_scalars(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten_scalars(v, depth + 1)
    elif isinstance(obj, (str, int, float)) and not isinstance(obj, bool):
        yield obj


# --------------------------------------------------------------------------- #
# Token alignment and retention measurement
# --------------------------------------------------------------------------- #


def char_to_token_spans(
    tokenizer, text: str, spans: Sequence[Span]
) -> tuple[list[tuple[Span, int, int]], int]:
    """Map char offsets to token index ranges via fast-tokenizer offsets.

    Returns the located spans and the total token count of ``text``. Uses binary
    search over the offset table: a linear scan per span is O(spans x tokens),
    which is minutes per trajectory at 30k tokens.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    n_tokens = len(offsets)
    if n_tokens == 0:
        return [], 0

    starts = np.array([s for s, _ in offsets], dtype=np.int64)
    ends = np.array([e for _, e in offsets], dtype=np.int64)

    out: list[tuple[Span, int, int]] = []
    for span in spans:
        # First token whose end > span.start, last token whose start < span.end.
        lo = int(np.searchsorted(ends, span.start, side="right"))
        hi = int(np.searchsorted(starts, span.end, side="left"))
        if hi > lo:
            out.append((span, lo, hi))
    return out, n_tokens


@dataclass
class SpanRetention:
    span_class: str
    n_tokens: int
    n_retained: int
    any_retained: bool
    all_retained: bool
    fraction_retained: float
    age_tokens: int
    relative_position: float
    context_tokens: int
    visible_structure: bool


def measure_retention(
    token_spans: Sequence[tuple[Span, int, int]],
    keep_indices: np.ndarray,
    context_tokens: int,
) -> list[SpanRetention]:
    """Per-span retention under one policy's keep mask."""
    kept = np.zeros(context_tokens, dtype=bool)
    idx = np.asarray(keep_indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < context_tokens)]
    kept[idx] = True

    out: list[SpanRetention] = []
    for span, lo, hi in token_spans:
        lo = max(0, min(lo, context_tokens))
        hi = max(lo, min(hi, context_tokens))
        n = hi - lo
        if n == 0:
            continue
        n_ret = int(kept[lo:hi].sum())
        out.append(
            SpanRetention(
                span_class=span.span_class,
                n_tokens=n,
                n_retained=n_ret,
                any_retained=n_ret > 0,
                all_retained=n_ret == n,
                fraction_retained=n_ret / n,
                # How far back the span sits from the decision point.
                age_tokens=context_tokens - hi,
                relative_position=lo / max(1, context_tokens),
                context_tokens=context_tokens,
                visible_structure=span.is_visible_structure,
            )
        )
    return out


def aggregate(retentions: Iterable[SpanRetention]) -> dict[str, Any]:
    """Aggregate retention by span class and by visible-vs-buried."""
    rows = list(retentions)
    if not rows:
        return {"n_spans": 0, "by_class": {}, "by_visibility": {}}

    def _agg(subset: list[SpanRetention]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "any_retained_rate": float(np.mean([r.any_retained for r in subset])),
            "all_retained_rate": float(np.mean([r.all_retained for r in subset])),
            "mean_fraction_retained": float(
                np.mean([r.fraction_retained for r in subset])
            ),
            "median_age_tokens": float(np.median([r.age_tokens for r in subset])),
            "median_context_tokens": float(
                np.median([r.context_tokens for r in subset])
            ),
        }

    by_class = {c: _agg([r for r in rows if r.span_class == c]) for c in SPAN_CLASSES}
    by_visibility = {
        "visible_structure": _agg([r for r in rows if r.visible_structure]),
        "buried": _agg([r for r in rows if not r.visible_structure]),
    }
    return {
        "n_spans": len(rows),
        "overall": _agg(rows),
        "by_class": by_class,
        "by_visibility": by_visibility,
    }


def sample_for_manual_audit(
    all_spans: Sequence[tuple[str, Span]],
    *,
    n: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Uniform random sample of extracted spans for hand-checked precision.

    Deliberately uniform over *all* extracted spans — never filtered to spans
    that happen to favour any policy.
    """
    rng = np.random.default_rng(seed)
    total = len(all_spans)
    if total == 0:
        return []
    pick = rng.choice(total, size=min(n, total), replace=False)
    out = []
    for i in sorted(int(p) for p in pick):
        traj_id, span = all_spans[i]
        out.append(
            {
                "traj_id": traj_id,
                "span_class": span.span_class,
                "text": span.text[:400],
                "source_role": span.source_role,
                "source_message_index": span.source_message_index,
                "char_start": span.start,
                "char_end": span.end,
                "correct_extraction": None,  # filled by a human auditor
                "notes": "",
            }
        )
    return out


def stratified_sample(
    trajectories: Sequence[Trajectory], *, n: int, seed: int = 0
) -> list[Trajectory]:
    """Stratify by (task_name, source_model) so repeats aren't pseudo-independent."""
    strata: dict[tuple[str, str], list[Trajectory]] = {}
    for t in trajectories:
        strata.setdefault((t.task_name, t.source_model), []).append(t)
    keys = sorted(strata)
    rng = np.random.default_rng(seed)
    out: list[Trajectory] = []
    per = max(1, n // max(1, len(keys)))
    for k in keys:
        pool = sorted(strata[k], key=lambda t: t.traj_id)
        take = min(per, len(pool))
        pick = rng.choice(len(pool), size=take, replace=False)
        out.extend(pool[int(p)] for p in sorted(pick))
    return out[:n]
