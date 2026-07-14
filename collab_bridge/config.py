from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    agent_id: str
    telegram_bot_token: str
    telegram_chat_id: str
    cursor_api_key: str
    cursor_model: str
    repo_root: Path
    github_repo: str
    github_default_branch: str
    tick_interval_sec: int = 3600
    max_messages_per_tick: int = 40
    agent_timeout_sec: int = 1800
    dry_run: bool = False
    require_scope_ack: bool = True
    stop_keywords: list[str] = field(default_factory=list)
    max_commits_per_tick: int = 5
    branch_prefix: str = ""
    state_dir: Path = field(default_factory=lambda: _repo_root_default() / "state")
    telegram_bootstrap_offset: int | None = None
    memory_window: int = 50
    use_agent_resume: bool = True

    @property
    def branch_ns(self) -> str:
        prefix = self.branch_prefix.strip()
        if prefix:
            return prefix.rstrip("/") + "/"
        return f"agent/{self.agent_id}/"

    @property
    def tag(self) -> str:
        return f"[agent:{self.agent_id}]"

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Settings":
        root = _repo_root_default()
        # override=True so a stale exported GITHUB_REPO/TELEGRAM_* in the shell
        # cannot silently win over the project .env
        load_dotenv(env_file or (root / ".env"), override=True)

        agent_id = (os.getenv("AGENT_ID") or "").strip().lower()
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
        api_key = (os.getenv("CURSOR_API_KEY") or "").strip()
        model = (os.getenv("CURSOR_MODEL") or "auto").strip()

        repo_root_env = (os.getenv("REPO_ROOT") or "").strip()
        repo_root = Path(repo_root_env) if repo_root_env else root

        stops = [
            s.strip()
            for s in (os.getenv("STOP_KEYWORDS") or "STOP_BRIDGE,HALT_AGENTS").split(",")
            if s.strip()
        ]

        bootstrap = os.getenv("TELEGRAM_BOOTSTRAP_OFFSET")
        branch_prefix = (os.getenv("BRANCH_PREFIX") or "").strip()

        return cls(
            agent_id=agent_id or "unnamed",
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
            cursor_api_key=api_key,
            cursor_model=model,
            repo_root=repo_root.resolve(),
            github_repo=(os.getenv("GITHUB_REPO") or "Arush777/Priority_KV").strip(),
            github_default_branch=(os.getenv("GITHUB_DEFAULT_BRANCH") or "main").strip(),
            tick_interval_sec=int(os.getenv("TICK_INTERVAL_SEC") or "3600"),
            max_messages_per_tick=int(os.getenv("MAX_MESSAGES_PER_TICK") or "40"),
            agent_timeout_sec=int(os.getenv("AGENT_TIMEOUT_SEC") or "1800"),
            dry_run=(os.getenv("DRY_RUN") or "0").strip() in {"1", "true", "True", "yes"},
            require_scope_ack=(os.getenv("REQUIRE_SCOPE_ACK") or "1").strip()
            in {"1", "true", "True", "yes"},
            stop_keywords=stops,
            max_commits_per_tick=int(os.getenv("MAX_COMMITS_PER_TICK") or "5"),
            branch_prefix=branch_prefix,
            state_dir=(repo_root / "state").resolve(),
            telegram_bootstrap_offset=int(bootstrap) if bootstrap else None,
            memory_window=int(os.getenv("MEMORY_WINDOW") or "50"),
            use_agent_resume=(os.getenv("USE_AGENT_RESUME") or "1").strip()
            in {"1", "true", "True", "yes"},
        )

    def validate_for_telegram(self) -> list[str]:
        errs: list[str] = []
        if not self.telegram_bot_token:
            errs.append("TELEGRAM_BOT_TOKEN is missing")
        if not self.telegram_chat_id:
            errs.append("TELEGRAM_CHAT_ID is missing")
        if not self.agent_id or self.agent_id == "unnamed":
            errs.append("AGENT_ID must be set (e.g. arush or friend)")
        return errs

    def validate_for_agent(self) -> list[str]:
        errs = self.validate_for_telegram()
        if not self.dry_run and not self.cursor_api_key:
            errs.append("CURSOR_API_KEY is missing (or set DRY_RUN=1)")
        if not self.repo_root.exists():
            errs.append(f"REPO_ROOT does not exist: {self.repo_root}")
        return errs
