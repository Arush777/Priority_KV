"""Pinned model / chat-template IDs for PriorityBench + S2 page tagging.

Locked in docs/decisions.md. Change only via a new Decided line.
"""

from __future__ import annotations

# Primary eval / tagging model (plan §3.3, §4).
QWEN3_8B_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# Agent traces: disable thinking mode so tool-call spans stay in the
# assistant channel the page tagger expects (not inside <think>).
QWEN3_ENABLE_THINKING = False


def qwen3_tokenizer_kwargs() -> dict:
    """kwargs for AutoTokenizer.from_pretrained(...)."""
    return {
        "pretrained_model_name_or_path": QWEN3_8B_MODEL_ID,
        "revision": QWEN3_8B_REVISION,
        "trust_remote_code": True,
    }


def qwen3_chat_template_kwargs() -> dict:
    """Extra kwargs for tokenizer.apply_chat_template(...)."""
    return {"enable_thinking": QWEN3_ENABLE_THINKING}
