"""Bridge to the *official* BFCL implementation at a pinned Gorilla checkout.

Nothing in this module reimplements BFCL logic. It only puts the pinned
``berkeley-function-call-leaderboard`` package on ``sys.path`` and re-exports the
official prompt constants, execution driver, and scorer, so the rest of the
harness cannot silently drift from upstream semantics.

The checkout is pinned by commit in ``configs/external_bfcl_prajna_v1.yaml``;
``assert_pinned_revision`` refuses to run against any other commit.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path


class BfclUnavailableError(RuntimeError):
    """Raised when the pinned official BFCL checkout cannot be used."""


def bfcl_package_root(gorilla_root: str | Path) -> Path:
    return Path(gorilla_root) / "berkeley-function-call-leaderboard"


def install_path(gorilla_root: str | Path) -> Path:
    """Prepend the pinned BFCL package to ``sys.path`` (idempotent)."""
    pkg = bfcl_package_root(gorilla_root)
    if not (pkg / "bfcl_eval").is_dir():
        raise BfclUnavailableError(f"no bfcl_eval package under {pkg}")
    p = str(pkg)
    if p not in sys.path:
        sys.path.insert(0, p)
    return pkg


def head_revision(gorilla_root: str | Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(gorilla_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def assert_pinned_revision(gorilla_root: str | Path, expected: str) -> str:
    """Hard-fail unless the checkout sits exactly on the frozen commit."""
    got = head_revision(gorilla_root)
    if got != expected:
        raise BfclUnavailableError(
            f"gorilla checkout is at {got}, frozen revision is {expected}; "
            "refusing to score against a different BFCL implementation"
        )
    return got


@functools.lru_cache(maxsize=4)
def load_official(gorilla_root: str) -> dict:
    """Import and return the official symbols this harness is allowed to use."""
    install_path(gorilla_root)
    from bfcl_eval.constants.category_mapping import (  # noqa: E402
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from bfcl_eval.constants.default_prompts import (  # noqa: E402
        DEFAULT_SYSTEM_PROMPT,
        MAXIMUM_STEP_LIMIT,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # noqa: E402
        multi_turn_checker,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )

    return {
        "DEFAULT_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT,
        "MAXIMUM_STEP_LIMIT": MAXIMUM_STEP_LIMIT,
        "MULTI_TURN_FUNC_DOC_FILE_MAPPING": MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        "multi_turn_checker": multi_turn_checker,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
        "is_empty_execute_response": is_empty_execute_response,
    }


def reset_execution_instances() -> None:
    """Drop cached stateful API instances held in ``multi_turn_utils`` globals.

    The official executor memoises one instance per
    ``{model_name}_{test_entry_id}_{class}`` key in module globals and never
    evicts. A long-lived shard would otherwise both leak memory and risk state
    bleeding between tasks, so shards call this between work units.
    """
    # Nothing to reset if the official package was never imported; treat that as
    # a no-op so callers can reset unconditionally.
    module = sys.modules.get("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    if module is None:
        return

    g = vars(module)
    for key in [k for k in g if k.endswith("_instance")]:
        del g[key]
