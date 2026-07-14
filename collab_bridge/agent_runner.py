from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .protocol import build_agent_prompt


@dataclass
class AgentRunResult:
    ok: bool
    mode: str
    summary: str
    status_file: Path | None = None
    error: str | None = None
    agent_run_id: str | None = None


def run_cursor_agent(settings: Settings, prompt: str) -> AgentRunResult:
    """Invoke Cursor SDK local agent against REPO_ROOT, or dry-run."""
    status_path = settings.state_dir / f"last_status_{settings.agent_id}.txt"
    settings.state_dir.mkdir(parents=True, exist_ok=True)

    if settings.dry_run:
        text = (
            f"[agent:{settings.agent_id}] TICK\n"
            "SUMMARY: DRY_RUN=1 — no Cursor SDK call. Bridge polled Telegram successfully.\n"
            "ACTIONS: parsed messages; skipped coding\n"
            "CLAIM: none\n"
            "DONE: none\n"
            "BLOCKED: none\n"
            f"ASK: @agent:peer confirm you received this dry-run\n"
            "PROPOSE_SCOPE: no\n"
            "NEXT: set DRY_RUN=0 and CURSOR_API_KEY to enable live ticks\n"
        )
        status_path.write_text(text, encoding="utf-8")
        return AgentRunResult(
            ok=True,
            mode="dry_run",
            summary="dry-run tick",
            status_file=status_path,
        )

    try:
        from cursor_sdk import Agent, LocalAgentOptions  # type: ignore
    except ImportError:
        # Fallback: try alternate import name if package differs
        try:
            from cursor_sdk import Agent  # type: ignore
            from cursor_sdk import LocalAgentOptions  # type: ignore
        except ImportError as exc:
            err = (
                "cursor-sdk is not installed. Run: pip install cursor-sdk\n"
                f"Import error: {exc}"
            )
            status_path.write_text(
                f"[agent:{settings.agent_id}] TICK\n"
                f"SUMMARY: FAILED — {err}\n"
                "ACTIONS: none\nCLAIM: none\nDONE: none\nBLOCKED: setup\n"
                "ASK: none\nPROPOSE_SCOPE: no\nNEXT: install cursor-sdk\n",
                encoding="utf-8",
            )
            return AgentRunResult(
                ok=False,
                mode="missing_sdk",
                summary=err,
                status_file=status_path,
                error=err,
            )

    try:
        # Persist prompt for debugging
        (settings.state_dir / f"last_prompt_{settings.agent_id}.txt").write_text(
            prompt, encoding="utf-8"
        )

        with Agent.create(
            model=settings.cursor_model,
            api_key=settings.cursor_api_key,
            local=LocalAgentOptions(cwd=str(settings.repo_root)),
        ) as agent:
            run = agent.send(prompt)
            result = run.wait()
            status = getattr(result, "status", None)
            run_id = getattr(result, "id", None) or getattr(run, "id", None)

            if status == "error":
                err = f"Cursor run failed: id={run_id}"
                if not status_path.exists():
                    status_path.write_text(
                        f"[agent:{settings.agent_id}] TICK\n"
                        f"SUMMARY: agent run error ({run_id})\n"
                        "ACTIONS: none reliable\nCLAIM: none\nDONE: none\n"
                        "BLOCKED: agent_error\nASK: @agent:peer please continue if you can\n"
                        "PROPOSE_SCOPE: no\nNEXT: retry next tick\n",
                        encoding="utf-8",
                    )
                return AgentRunResult(
                    ok=False,
                    mode="cursor_sdk",
                    summary=err,
                    status_file=status_path if status_path.exists() else None,
                    error=err,
                    agent_run_id=str(run_id) if run_id else None,
                )

            # Prefer agent-written status file; else synthesize from result text
            if not status_path.exists():
                text = ""
                for getter in ("text", "result"):
                    if hasattr(result, getter):
                        try:
                            val = getattr(result, getter)
                            text = val() if callable(val) else str(val or "")
                            if text:
                                break
                        except Exception:
                            pass
                status_path.write_text(
                    f"[agent:{settings.agent_id}] TICK\n"
                    f"SUMMARY: completed run {run_id}\n"
                    f"ACTIONS: see agent transcript\n"
                    f"CLAIM: none\nDONE: none\nBLOCKED: none\n"
                    f"ASK: @agent:peer review latest commits/PR\n"
                    f"PROPOSE_SCOPE: no\nNEXT: continue IR scope work\n"
                    f"\n--- raw ---\n{text[:3500]}\n",
                    encoding="utf-8",
                )

            return AgentRunResult(
                ok=True,
                mode="cursor_sdk",
                summary=f"cursor run ok id={run_id}",
                status_file=status_path,
                agent_run_id=str(run_id) if run_id else None,
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status_path.write_text(
            f"[agent:{settings.agent_id}] TICK\n"
            f"SUMMARY: exception during agent run\n"
            f"ACTIONS: none\nCLAIM: none\nDONE: none\nBLOCKED: exception\n"
            f"ASK: @agent:peer hold — I hit an error\nPROPOSE_SCOPE: no\n"
            f"NEXT: debug bridge logs\n\n{err[:3000]}\n",
            encoding="utf-8",
        )
        return AgentRunResult(
            ok=False,
            mode="cursor_sdk",
            summary=str(exc),
            status_file=status_path,
            error=err,
        )


def compose_prompt(settings: Settings, transcript: str) -> str:
    peer = "friend" if settings.agent_id != "friend" else "arush"
    return build_agent_prompt(
        agent_id=settings.agent_id,
        peer_hint=peer,
        github_repo=settings.github_repo,
        default_branch=settings.github_default_branch,
        branch_ns=settings.branch_ns,
        scope_path="scopes/PROJECT_SCOPE.md",
        collab_path="COLLAB.md",
        transcript=transcript,
        require_scope_ack=settings.require_scope_ack,
        max_commits=settings.max_commits_per_tick,
    )
