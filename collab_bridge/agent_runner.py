from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .memory import CollabMemory
from .protocol import build_agent_prompt


@dataclass
class AgentRunResult:
    ok: bool
    mode: str
    summary: str
    status_file: Path | None = None
    error: str | None = None
    agent_run_id: str | None = None
    cursor_agent_id: str | None = None
    resumed: bool = False


def run_cursor_agent(
    settings: Settings,
    prompt: str,
    *,
    memory: CollabMemory | None = None,
) -> AgentRunResult:
    """Invoke Cursor SDK local agent; resume prior agent id when configured."""
    status_path = settings.state_dir / f"last_status_{settings.agent_id}.txt"
    settings.state_dir.mkdir(parents=True, exist_ok=True)

    if settings.dry_run:
        text = (
            f"[agent:{settings.agent_id}] TICK\n"
            "SUMMARY: DRY_RUN=1 — no Cursor SDK call. Memory ring would update.\n"
            "ACTIONS: parsed messages; skipped coding\n"
            "CLAIM: none\nDONE: none\nBLOCKED: none\n"
            "ASK: @agent:peer confirm dry-run\n"
            "PROPOSE_SCOPE: no\nDECISIONS_WRITTEN: no\n"
            "NEXT: set DRY_RUN=0 for live ticks\n"
        )
        status_path.write_text(text, encoding="utf-8")
        return AgentRunResult(
            ok=True,
            mode="dry_run",
            summary="dry-run tick",
            status_file=status_path,
            resumed=False,
        )

    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions  # type: ignore
    except ImportError:
        try:
            from cursor_sdk import Agent, LocalAgentOptions  # type: ignore

            AgentOptions = None  # type: ignore
        except ImportError as exc:
            err = (
                "cursor-sdk is not installed. Run: pip install cursor-sdk\n"
                f"Import error: {exc}"
            )
            status_path.write_text(
                f"[agent:{settings.agent_id}] TICK\n"
                f"SUMMARY: FAILED — {err}\n"
                "ACTIONS: none\nCLAIM: none\nDONE: none\nBLOCKED: setup\n"
                "ASK: none\nPROPOSE_SCOPE: no\nDECISIONS_WRITTEN: no\n"
                "NEXT: install cursor-sdk\n",
                encoding="utf-8",
            )
            return AgentRunResult(
                ok=False,
                mode="missing_sdk",
                summary=err,
                status_file=status_path,
                error=err,
            )

    prior_id = memory.get_cursor_agent_id() if memory else None
    resumed = False
    agent_cm = None
    cursor_agent_id = prior_id

    try:
        (settings.state_dir / f"last_prompt_{settings.agent_id}.txt").write_text(
            prompt, encoding="utf-8"
        )

        if settings.use_agent_resume and prior_id:
            try:
                if AgentOptions is not None:
                    agent_cm = Agent.resume(
                        prior_id,
                        AgentOptions(
                            api_key=settings.cursor_api_key,
                            model=settings.cursor_model,
                            local=LocalAgentOptions(cwd=str(settings.repo_root)),
                        ),
                    )
                else:
                    agent_cm = Agent.resume(
                        prior_id,
                        api_key=settings.cursor_api_key,
                    )
                resumed = True
            except Exception as resume_exc:
                # Fall back to create; clear stale id
                resumed = False
                if memory:
                    memory.set_cursor_agent_id(None)
                (settings.state_dir / f"last_resume_error_{settings.agent_id}.txt").write_text(
                    f"{type(resume_exc).__name__}: {resume_exc}\n",
                    encoding="utf-8",
                )
                agent_cm = Agent.create(
                    model=settings.cursor_model,
                    api_key=settings.cursor_api_key,
                    local=LocalAgentOptions(cwd=str(settings.repo_root)),
                )
        else:
            agent_cm = Agent.create(
                model=settings.cursor_model,
                api_key=settings.cursor_api_key,
                local=LocalAgentOptions(cwd=str(settings.repo_root)),
            )

        with agent_cm as agent:
            # Persist durable agent id for next tick
            for attr in ("agent_id", "agentId", "id"):
                if hasattr(agent, attr):
                    val = getattr(agent, attr)
                    if val:
                        cursor_agent_id = str(val)
                        break
            if memory and cursor_agent_id:
                memory.set_cursor_agent_id(cursor_agent_id)

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
                        "BLOCKED: agent_error\nASK: @agent:peer please continue\n"
                        "PROPOSE_SCOPE: no\nDECISIONS_WRITTEN: no\n"
                        "NEXT: retry next tick\n",
                        encoding="utf-8",
                    )
                return AgentRunResult(
                    ok=False,
                    mode="cursor_sdk_resume" if resumed else "cursor_sdk",
                    summary=err,
                    status_file=status_path if status_path.exists() else None,
                    error=err,
                    agent_run_id=str(run_id) if run_id else None,
                    cursor_agent_id=cursor_agent_id,
                    resumed=resumed,
                )

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
                    f"SUMMARY: completed run {run_id} resumed={resumed}\n"
                    f"ACTIONS: see agent transcript\n"
                    f"CLAIM: none\nDONE: none\nBLOCKED: none\n"
                    f"ASK: @agent:peer review latest commits/PR\n"
                    f"PROPOSE_SCOPE: no\nDECISIONS_WRITTEN: no\n"
                    f"NEXT: continue Priority_KV work\n"
                    f"\n--- raw ---\n{text[:3500]}\n",
                    encoding="utf-8",
                )

            return AgentRunResult(
                ok=True,
                mode="cursor_sdk_resume" if resumed else "cursor_sdk",
                summary=f"cursor run ok id={run_id} resumed={resumed}",
                status_file=status_path,
                agent_run_id=str(run_id) if run_id else None,
                cursor_agent_id=cursor_agent_id,
                resumed=resumed,
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status_path.write_text(
            f"[agent:{settings.agent_id}] TICK\n"
            f"SUMMARY: exception during agent run\n"
            f"ACTIONS: none\nCLAIM: none\nDONE: none\nBLOCKED: exception\n"
            f"ASK: @agent:peer hold — I hit an error\nPROPOSE_SCOPE: no\n"
            f"DECISIONS_WRITTEN: no\nNEXT: debug bridge logs\n\n{err[:3000]}\n",
            encoding="utf-8",
        )
        return AgentRunResult(
            ok=False,
            mode="cursor_sdk",
            summary=str(exc),
            status_file=status_path,
            error=err,
            cursor_agent_id=cursor_agent_id,
            resumed=resumed,
        )


def compose_prompt(
    settings: Settings,
    transcript: str,
    *,
    memory_blob: str,
    resumed: bool,
) -> str:
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
        memory_blob=memory_blob,
        require_scope_ack=settings.require_scope_ack,
        max_commits=settings.max_commits_per_tick,
        resumed=resumed,
    )
