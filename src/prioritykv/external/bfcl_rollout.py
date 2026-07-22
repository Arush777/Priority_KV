"""Official BFCL V3 multi-turn rollout, driven through a retention arm.

This reproduces ``BaseHandler.inference_multi_turn_prompting`` from the pinned
Gorilla checkout: same turn/step structure, same miss-func holdout reveal, same
step limit and force-quit semantics, same ``role="tool"`` execution-result
messages. The only substitution is *how* the next token block is produced — a
:class:`~prioritykv.external.arms.Generator` applies the retention policy to the
growing conversation before each generation step.

The unit of analysis is the conversation: the official checker returns one
verdict per task, so turns are never treated as independent samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from prioritykv.external.bfcl_data import LONG_CONTEXT_CATEGORIES, BfclTask

def _reveal_template() -> str:
    """The official miss-func reveal template, read from the pinned checkout.

    Never hardcoded here: the exact wording is part of the prompt under test and
    must track upstream rather than a copy that can silently drift.
    """
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    )

    return DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING


class ContextLimitExceeded(RuntimeError):
    """Raised instead of silently truncating an external prompt."""

    def __init__(self, prompt_tokens: int, limit: int, turn: int, step: int):
        super().__init__(
            f"prompt is {prompt_tokens} tokens, over the {limit}-token ceiling "
            f"at turn {turn} step {step}"
        )
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        self.turn = turn
        self.step = step


@dataclass
class RolloutResult:
    """Everything one conversation-arm rollout produced."""

    task_id: str
    arm: str
    # Shape the official checker consumes: [turn][step][call strings]
    model_result_decoded: list[list[list[str]]]
    raw_outputs: list[list[str]] = field(default_factory=list)
    terminal_status: str = "success"
    error: str | None = None
    force_quit: bool = False
    steps_used: int = 0
    prompt_token_counts: list[int] = field(default_factory=list)
    requested_keep: list[int] = field(default_factory=list)
    realized_keep: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def max_prompt_tokens(self) -> int:
        return max(self.prompt_token_counts) if self.prompt_token_counts else 0


def _reveal_prompt(holdout_docs: list[dict]) -> str:
    return _reveal_template().format(functions=holdout_docs)


def split_reasoning(raw: str) -> tuple[str, str]:
    """Split a reasoning-model response into (reasoning, answer).

    Port of ``QwenHandler._parse_query_response_prompting`` from the pinned
    checkout. Without this, a thinking-mode response like
    ``<think>...</think>\\n\\n[foo(a=1)]`` is handed to the decoder whole, which
    assumes the text *is* the call list -- so a perfectly correct tool call
    decodes to nothing and the turn is scored as empty.
    """
    if "</think>" not in raw:
        return "", raw
    parts = raw.split("</think>")
    reasoning = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
    return reasoning, parts[-1].lstrip("\n")


def run_rollout(
    task: BfclTask,
    generator,
    *,
    system_prompt: str,
    decode_execute: Callable[[str], list[str]],
    execute_calls: Callable[..., tuple[list[str], dict]],
    is_empty_response: Callable[[list], bool],
    max_step_limit: int = 20,
    max_new_tokens: int = 512,
    prompt_token_ceiling: int | None = None,
    execution_model_name: str = "prioritykv",
) -> RolloutResult:
    """Drive one conversation through one retention arm.

    ``execute_calls`` is the official ``execute_multi_turn_func_call``; state is
    namespaced by ``execution_model_name`` so concurrent arms cannot contaminate
    each other's stateful API instances.
    """
    long_context = task.category in LONG_CONTEXT_CATEGORIES
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    all_decoded: list[list[list[str]]] = []
    all_raw: list[list[str]] = []
    prompt_counts: list[int] = []
    requested_keep: list[int] = []
    realized_keep: list[int] = []
    total_gen = 0.0
    total_sel = 0.0
    steps_used = 0
    force_quit = False
    step_extras: list[dict[str, Any]] = []

    t_start = time.perf_counter()

    for turn_idx, turn_messages in enumerate(task.question):
        current = list(turn_messages)
        holdout = task.missed_function.get(str(turn_idx))
        if holdout:
            # Official contract: a holdout turn carries no user message; the
            # withheld tool docs are revealed as a synthetic user turn.
            current = [{"role": "user", "content": _reveal_prompt(holdout)}]
        messages.extend(current)

        turn_decoded: list[list[str]] = []
        turn_raw: list[str] = []
        step = 0
        while True:
            result = generator.generate(messages, max_new_tokens)
            if prompt_token_ceiling is not None and result.prompt_tokens > prompt_token_ceiling:
                raise ContextLimitExceeded(
                    result.prompt_tokens, prompt_token_ceiling, turn_idx, step
                )
            prompt_counts.append(result.prompt_tokens)
            requested_keep.append(result.requested_keep)
            realized_keep.append(result.realized_keep)
            total_gen += result.timings.get("generate_s", 0.0)
            total_sel += result.timings.get("select_s", 0.0)
            steps_used += 1
            if result.extra:
                step_extras.append(dict(result.extra))

            raw = result.text
            turn_raw.append(raw)
            # Match the official handler: the assistant turn carries the answer,
            # with reasoning kept alongside rather than inlined into the history.
            reasoning, answer = split_reasoning(raw)
            assistant_msg = {"role": "assistant", "content": answer}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            try:
                decoded = decode_execute(answer)
            except Exception:  # noqa: BLE001
                # Upstream treats an undecodable response as end-of-turn.
                turn_decoded.append([])
                break

            turn_decoded.append(list(decoded))
            if is_empty_response(decoded):
                break

            exec_results, _ = execute_calls(
                func_call_list=decoded,
                initial_config=task.initial_config,
                involved_classes=task.involved_classes,
                model_name=execution_model_name,
                test_entry_id=task.task_id,
                long_context=long_context,
                is_evaL_run=False,
            )
            for call, res in zip(decoded, exec_results):
                messages.append({"role": "tool", "name": call, "content": str(res)})

            step += 1
            if step >= max_step_limit:
                force_quit = True
                break

        all_decoded.append(turn_decoded)
        all_raw.append(turn_raw)
        if force_quit:
            break

    return RolloutResult(
        task_id=task.task_id,
        arm=getattr(generator, "arm", "unknown"),
        model_result_decoded=all_decoded,
        raw_outputs=all_raw,
        terminal_status="force_quit" if force_quit else "success",
        force_quit=force_quit,
        steps_used=steps_used,
        prompt_token_counts=prompt_counts,
        requested_keep=requested_keep,
        realized_keep=realized_keep,
        timings={
            "generate_s": total_gen,
            "select_s": total_sel,
            "end_to_end_s": time.perf_counter() - t_start,
        },
        extra={
            "turns": len(all_decoded),
            "n_turns_expected": task.n_turns,
            # Per-step press metadata (press class, compression ratio, and for
            # ADAPT the alpha actually used) so the policy is auditable.
            "step_extras": step_extras,
        },
    )


def pad_decoded_to_turns(decoded: list[list[list[str]]], n_turns: int) -> list[list[list[str]]]:
    """Pad a short (force-quit) rollout so the official checker can consume it.

    A truncated rollout must be *scored as a failure*, not skipped, so missing
    turns are filled with empty step lists rather than dropped.
    """
    out = [list(t) for t in decoded]
    while len(out) < n_turns:
        out.append([])
    return out[:n_turns]


def score_rollout(
    task: BfclTask,
    rollout: RolloutResult,
    *,
    multi_turn_checker: Callable[..., dict],
    model_name: str,
) -> dict[str, Any]:
    """Score with the unmodified official ``multi_turn_checker``."""
    decoded = pad_decoded_to_turns(rollout.model_result_decoded, task.n_turns)
    try:
        verdict = multi_turn_checker(
            decoded,
            task.ground_truth,
            task.as_test_entry(),
            task.category,
            model_name,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "scorer_error": f"{type(exc).__name__}: {exc}",
            "error_type": "scorer_exception",
        }
    return dict(verdict)


def summarise_turn_lengths(task: BfclTask) -> dict[str, int]:
    return {
        "n_turns": task.n_turns,
        "n_ground_truth_turns": len(task.ground_truth),
        "n_functions": len(task.function),
        "n_involved_classes": len(task.involved_classes),
    }


def messages_for_turn(
    task: BfclTask, system_prompt: str, turn_idx: int
) -> list[dict[str, str]]:
    """Prefix through ``turn_idx`` user turns, for context-length accounting."""
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for i, turn in enumerate(task.question[: turn_idx + 1]):
        holdout = task.missed_function.get(str(i))
        if holdout:
            msgs.append({"role": "user", "content": _reveal_prompt(holdout)})
        else:
            msgs.extend(list(turn))
    return msgs


def all_user_messages(task: BfclTask) -> Sequence[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in task.question:
        out.extend(turn)
    return out
