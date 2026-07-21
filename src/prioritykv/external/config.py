"""Loader for the frozen EXTERNAL_BFCL_PRAJNA_V1 config."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in env:
                raise KeyError(f"config references ${{{name}}} but it is unset")
            return env[name]
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    return value


def load_config(path: str | Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Load the frozen config, expanding ``${VAR}`` from the environment."""
    env = dict(os.environ if env is None else env)
    raw = yaml.safe_load(Path(path).read_text())
    return _expand(raw, env)


def keep_policy_config(cfg: dict[str, Any], *, keep_frac: float | None = None, seed: int = 0):
    """Build a ``KeepPolicyConfig`` from the frozen arm settings."""
    from prioritykv.baselines.keep_policy import KeepPolicyConfig

    kp = cfg["arms"]["keep_policy"]
    return KeepPolicyConfig(
        keep_frac=float(cfg["arms"]["keep_frac"] if keep_frac is None else keep_frac),
        sink_tokens=int(kp["sink_tokens"]),
        force_recent=int(kp["force_recent"]),
        seed=int(seed),
        granularity=str(kp.get("granularity", "token")),
    )


def harness_revision(repo_root: str | Path) -> str:
    """Git commit of this harness; part of every work unit's identity."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def uv_lock_hash(repo_root: str | Path) -> str:
    import hashlib

    p = Path(repo_root) / "uv.lock"
    if not p.is_file():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()
