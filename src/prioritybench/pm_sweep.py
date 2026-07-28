"""Protected-mass sweep generator (SWEEP_PM_V1, post-freeze namespace).

The frozen results measure two endpoints of the structural prior's operating
range: PriorityBench-A stress (~6% protected-role mass, structure wins) and
BFCL multi-turn (~99%, structure saturates). This module fills the interval.

Each sweep example starts from a plain-state stress example (gold state in
short recognizable turns, long ``[filler/…]`` padding) and converts some of its
filler (user, assistant) pairs into *distractor tool-schema* pairs: a short
registry notice plus a JSON dump of deferred tool schemas.  Converted pairs
carry protected roles under the frozen tagger (short user turn → OTHER,
JSON schema dump → TOOL), so the conversion dials the protected-role fraction
up from the ~6% base toward ~95% while:

- the gold state, final ask, and scoring payload stay byte-identical,
- the approximate context length is preserved (dumps are sized to the chars
  they replace), and
- the added mass is *realistic* distraction — exactly the kind of schema bulk
  that dominates BFCL system prompts.

Nothing under FINAL_RUN_MANIFEST.yaml is touched; frozen generators are reused
read-only.  Nominal levels are targets; the manifest records the *measured*
protected fraction per example (pinned tokenizer, same measure as
``scripts/analyze_protected_fraction.py``), and analysis uses the measured value.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence

from prioritybench.generate import _generate_round_robin
from prioritybench.schema import PriorityExample
from prioritybench.templates import (
    INSTRUCTION_SUPERSESSION_TEMPLATES_V2,
    MULTI_TURN_STATE_TEMPLATES_V2,
    TOOL_SCHEMA_TEMPLATES,
)
from prioritybench.templates.base import approx_token_len
from prioritykv.baselines.buried_state import relocate_state_to_middle
from prioritykv.baselines.keep_policy import _message_role_stress
from prioritykv.page_roles import PROTECTED_ROLES, PageRole

SWEEP_PM_SEED = 20270301
# 0.06 realizes at the unconverted base; the rest bracket the three keep-budget
# crossovers (0.10, 0.25, 0.50) from both sides.
PM_LEVELS: tuple[float, ...] = (0.06, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95)
PM_CONTEXT_LENGTH = 8000

# Same structural-role set as scripts/analyze_protected_fraction.py, so the
# sweep's x-axis is commensurable with the paper's 6.0% / 98.8% endpoints.
STRUCTURE_ROLES = frozenset(PROTECTED_ROLES) | {PageRole.OTHER}

# Fixed sink + forced-recent overlay (approximate; the manifest's measured
# fraction uses the token-level tagger instead).
_OVERLAY_TOKENS = 16 + 128

# Distractor tool names deliberately disjoint from every gold tool name used by
# TOOL_SCHEMA_TEMPLATES (search_docs, list_files, echo_debug, read_file,
# write_file, stat_path, sql_query, sql_explain, list_tables, http_get,
# http_post, dns_lookup).
_DISTRACTOR_STEMS: tuple[str, ...] = (
    "rotate_credentials",
    "sync_ledger",
    "export_metrics",
    "archive_snapshot",
    "queue_digest",
    "validate_manifest",
    "refresh_cache_index",
    "annotate_trace",
    "bundle_artifacts",
    "diff_release_config",
)

_DISTRACTOR_PARAMS: tuple[str, ...] = (
    "path",
    "limit",
    "cursor",
    "label",
    "region",
    "dry_run",
    "batch_id",
    "ttl_seconds",
    "output_format",
    "channel",
)

# Short user notice for a converted pair. Kept free of every constraint-hint
# keyword (must / never / only / constraint / forbidden / always answer /
# latest instruction) so it cannot collide with the supersession tagger or
# scorer; under 500 chars it tags as OTHER, mirroring other short state turns.
_REGISTRY_NOTICE = (
    "Automated registry event: additional deferred tool schemas were published "
    "to this session for a later workflow phase. Reference copy follows."
)


def _distractor_schema(rng: random.Random) -> Dict:
    stem = rng.choice(_DISTRACTOR_STEMS)
    props: Dict[str, Dict] = {}
    required: List[str] = []
    for _ in range(rng.randint(2, 5)):
        pname = f"{rng.choice(_DISTRACTOR_PARAMS)}_{rng.randint(1, 99)}"
        ptype = rng.choice(("string", "integer", "boolean"))
        props[pname] = {
            "type": ptype,
            "description": f"{pname} argument for {stem.replace('_', ' ')}",
        }
        if rng.random() < 0.5:
            required.append(pname)
    return {
        "name": f"{stem}_v{rng.randint(1, 9)}",
        "description": f"Deferred-phase operation: {stem.replace('_', ' ')}.",
        "parameters": {
            "type": "object",
            "required": sorted(set(required)),
            "properties": props,
        },
    }


def _schema_dump(rng: random.Random, min_chars: int) -> str:
    """Assistant-side JSON dump of deferred tool schemas, ≥ ``min_chars`` chars.

    Starts with ``{`` and contains ``"name":`` fields, so the frozen tagger
    assigns it PageRole.TOOL — the same role the BFCL schema bulk carries.
    """
    schemas: List[Dict] = [_distractor_schema(rng)]
    payload = {
        "registered_tool_schemas": schemas,
        "schema_format": "json schema",
        "note": (
            "Deferred tools recorded for a later phase of this workflow; "
            "not part of the current request."
        ),
    }
    while len(json.dumps(payload, indent=2)) < min_chars:
        schemas.append(_distractor_schema(rng))
    return json.dumps(payload, indent=2)


def approx_protected_fraction(messages: Sequence[Dict[str, str]]) -> float:
    """Char-approximate protected-role mass (construction-time greedy target).

    The manifest later records the exact token-level measurement; this
    approximation exists so conversion can stop near the nominal level without
    loading a tokenizer inside the generator.
    """
    total = 0
    prot = 0
    for msg in messages:
        t = approx_token_len(msg.get("content", ""))
        total += t
        if _message_role_stress(msg) in STRUCTURE_ROLES:
            prot += t
    total = max(total, 1)
    return min(1.0, (prot + _OVERLAY_TOKENS) / (total + _OVERLAY_TOKENS))


def _filler_pair_indices(messages: Sequence[Dict[str, str]]) -> List[int]:
    """Indices i where (messages[i], messages[i+1]) is a filler user/asst pair."""
    out: List[int] = []
    for i in range(len(messages) - 1):
        if (
            messages[i].get("role") == "user"
            and str(messages[i].get("content", "")).startswith("[filler/")
            and messages[i + 1].get("role") == "assistant"
        ):
            out.append(i)
    return out


def convert_to_level(
    ex: PriorityExample,
    target_frac: float,
    *,
    master_seed: int = SWEEP_PM_SEED,
) -> PriorityExample:
    """Return a copy of ``ex`` whose filler pairs are converted until the
    approximate protected fraction reaches ``target_frac``."""
    level_tag = f"pm{int(round(target_frac * 100)):02d}"
    rng = random.Random(f"{master_seed}:{ex.example_id}:{level_tag}")
    msgs = [dict(m) for m in ex.messages]

    pairs = _filler_pair_indices(msgs)
    order = list(pairs)
    rng.shuffle(order)

    def _needed_chars() -> int:
        """Chars of additional protected mass required to reach the target."""
        total = sum(approx_token_len(m.get("content", "")) for m in msgs)
        prot = sum(
            approx_token_len(m.get("content", ""))
            for m in msgs
            if _message_role_stress(m) in STRUCTURE_ROLES
        )
        deficit_tokens = target_frac * (total + _OVERLAY_TOKENS) - prot - _OVERLAY_TOKENS
        return int(deficit_tokens * 4)

    converted = 0
    partial = 0
    for i in order:
        need = _needed_chars()
        if need <= 0:
            break
        filler_text = msgs[i]["content"]
        ack_text = msgs[i + 1]["content"]
        pair_chars = len(filler_text) + len(ack_text)
        leftover = pair_chars - need
        if leftover >= 600:
            # Partial conversion: schema pair sized to land on the target,
            # followed by a re-inserted filler pair carrying the leftover mass,
            # so the level grid stays fine-grained despite big filler pairs.
            dump = _schema_dump(rng, max(400, need))
            msgs[i]["content"] = _REGISTRY_NOTICE
            msgs[i + 1]["content"] = dump
            msgs.insert(i + 2, {"role": "user", "content": filler_text[:leftover]})
            msgs.insert(i + 3, {"role": "assistant", "content": ack_text})
            converted += 1
            partial = 1
            break
        dump = _schema_dump(rng, max(400, pair_chars - len(_REGISTRY_NOTICE)))
        msgs[i]["content"] = _REGISTRY_NOTICE
        msgs[i + 1]["content"] = dump
        converted += 1

    return PriorityExample(
        example_id=f"{ex.example_id}__{level_tag}",
        category=ex.category,
        split=ex.split,
        context_length=ex.context_length,
        template_id=ex.template_id,
        seed=ex.seed,
        messages=msgs,
        scoring=dict(ex.scoring),
        meta={
            **dict(ex.meta),
            "pm_level": level_tag,
            "pm_target_frac": float(target_frac),
            "pm_approx_frac": approx_protected_fraction(msgs),
            "pm_converted_pairs": converted,
            "pm_partial_pair": partial,
            "pm_filler_pairs_total": len(pairs),
            "pm_base_example_id": ex.example_id,
            "generator": "prioritybench.pm_sweep",
        },
    )


def generate_pm_base_pool(
    *,
    master_seed: int = SWEEP_PM_SEED,
    n_per_category: int = 20,
    context_length: int = PM_CONTEXT_LENGTH,
) -> List[PriorityExample]:
    """Plain-state base pool: ``n_per_category`` × 3 categories at one length.

    Buried-state variants are excluded on purpose: burying tests whether the
    *label* can find the state at all, which is a different failure mechanism
    from the budget oversubscription this sweep isolates.  The exclusion is
    recorded in the manifest.
    """
    overgen = max(n_per_category * 3, n_per_category + 8)
    pools = (
        _generate_round_robin(
            overgen,
            master_seed=master_seed,
            context_lengths=(context_length,),
            templates=TOOL_SCHEMA_TEMPLATES,
        ),
        _generate_round_robin(
            overgen,
            master_seed=master_seed + 10_000,
            context_lengths=(context_length,),
            templates=INSTRUCTION_SUPERSESSION_TEMPLATES_V2,
        ),
        _generate_round_robin(
            overgen,
            master_seed=master_seed + 20_000,
            context_lengths=(context_length,),
            templates=MULTI_TURN_STATE_TEMPLATES_V2,
        ),
    )
    out: List[PriorityExample] = []
    for pool in pools:
        plain = [ex for ex in sorted(pool, key=lambda e: e.example_id)
                 if not bool(ex.meta.get("buried_state"))]
        if len(plain) < n_per_category:
            raise RuntimeError(
                f"only {len(plain)} plain-state examples available in "
                f"{pool[0].category.value}; raise overgen"
            )
        out.extend(plain[:n_per_category])
    return out


def relocate_gold_to_middle(ex: PriorityExample, *, position: float = 0.5) -> PriorityExample:
    """Move the gold-bearing turns into the middle of the filler/schema band.

    Sweep v1 measured no degradation even at 5.5x oversubscription, because
    PriorityBench templates emit gold state in the *prefix* and the evaluated
    structure policy breaks ties by index order: when the protected set does not
    fit, the earliest protected positions survive, which is exactly where gold
    already sits.  Oversubscription therefore could not bite.

    Relocating gold to mid-context removes that accidental rescue, so the budget
    test measures what it claims to measure: whether a binary role label can
    still find protected state once the protected set exceeds the budget.
    """
    msgs = relocate_state_to_middle(ex.messages, position=position)
    return PriorityExample(
        example_id=f"{ex.example_id}__mid",
        category=ex.category,
        split=ex.split,
        context_length=ex.context_length,
        template_id=ex.template_id,
        seed=ex.seed,
        messages=msgs,
        scoring=dict(ex.scoring),
        meta={**dict(ex.meta), "pm_placement": "middle",
              "pm_relocate_position": float(position)},
    )


def generate_pm_sweep(
    *,
    master_seed: int = SWEEP_PM_SEED,
    levels: Sequence[float] = PM_LEVELS,
    n_per_category: int = 20,
    context_length: int = PM_CONTEXT_LENGTH,
    placement: str = "prefix",
) -> List[PriorityExample]:
    """Full sweep: every base example instantiated at every protected-mass level.

    The same gold task appears at every level, so per-level comparisons are
    paired on the base example and the only thing that varies along the x-axis
    is the composition of the surrounding context.

    ``placement`` selects where the gold state sits: ``prefix`` reproduces the
    frozen template layout (v1), ``middle`` relocates gold into the filler band
    after schema conversion, which is the condition under which the
    oversubscription boundary is actually testable.
    """
    if placement not in ("prefix", "middle"):
        raise ValueError(f"unknown placement {placement!r}")
    base = generate_pm_base_pool(
        master_seed=master_seed,
        n_per_category=n_per_category,
        context_length=context_length,
    )
    out: List[PriorityExample] = []
    for level in levels:
        for ex in base:
            # Relocate BEFORE conversion. relocate_state_to_middle classifies a
            # turn as filler only by the generator's `[filler/…]` / "Acknowledged
            # filler" markers, so converted schema pairs read as non-filler and
            # get bundled into the relocated gold block; because the surviving
            # filler list shrinks as conversion rises, that pushed gold back
            # toward the prefix exactly at the high-mass levels (measured: char
            # fraction 0.65 at pm06 but 0.07 at pm95), confounding the x-axis.
            # Relocating first puts gold mid-context in plain filler, and the
            # surrounding filler is then converted around it.
            src = relocate_gold_to_middle(ex) if placement == "middle" else ex
            out.append(convert_to_level(src, level, master_seed=master_seed))
    return out
