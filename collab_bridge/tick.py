from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .agent_runner import compose_prompt, run_cursor_agent
from .config import Settings
from .protocol import ControlAction, detect_stop, format_transcript, parse_message
from .state_store import StateStore
from .telegram_client import TelegramClient


@dataclass
class TickResult:
    skipped: bool
    reason: str | None
    messages_seen: int
    posted: bool
    details: dict[str, Any]


def run_tick(settings: Settings) -> TickResult:
    errs = settings.validate_for_agent()
    if errs:
        raise RuntimeError("Config invalid:\n- " + "\n- ".join(errs))

    store = StateStore(settings.state_dir, settings.agent_id)
    if not store.acquire_lock():
        return TickResult(
            skipped=True,
            reason="lock held (another tick running)",
            messages_seen=0,
            posted=False,
            details={},
        )

    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    posted = False
    try:
        offset = store.telegram_offset
        if offset is None and settings.telegram_bootstrap_offset is not None:
            offset = settings.telegram_bootstrap_offset

        messages, next_offset = tg.fetch_new_messages(
            offset, limit=settings.max_messages_per_tick
        )
        parsed = [parse_message(m) for m in messages]

        for p in parsed:
            stop_kw = detect_stop(p.msg.text, settings.stop_keywords)
            if stop_kw and p.is_human:
                store.pause(f"keyword {stop_kw} in message {p.msg.message_id}")
                tg.send_message(
                    f"{settings.tag} PAUSED after `{stop_kw}`. "
                    "Human: send RESUME_BRIDGE to continue."
                )
                posted = True
                if next_offset is not None:
                    store.telegram_offset = next_offset
                store.record_tick(
                    {
                        "ts": time.time(),
                        "paused": True,
                        "stop_kw": stop_kw,
                        "messages": len(messages),
                    }
                )
                return TickResult(
                    skipped=True,
                    reason=f"paused by {stop_kw}",
                    messages_seen=len(messages),
                    posted=posted,
                    details={"stop_kw": stop_kw},
                )
            if p.control == ControlAction.RESUME and p.is_human:
                store.resume()
                tg.send_message(f"{settings.tag} RESUMED by human.")
                posted = True

        if store.paused:
            if next_offset is not None:
                store.telegram_offset = next_offset
            return TickResult(
                skipped=True,
                reason=f"paused: {store._data.get('pause_reason')}",
                messages_seen=len(messages),
                posted=posted,
                details={},
            )

        relevant = [p for p in parsed if p.author_agent != settings.agent_id]
        heartbeat = not relevant
        if heartbeat:
            transcript = (
                "(no new peer/human messages)\n"
                "Hourly heartbeat: review repo status, pick one small unowned "
                "improvement aligned with scopes/PROJECT_SCOPE.md, or ask the peer "
                "a concrete clarifying question about the Information Retrieval idea.\n"
            )
        else:
            transcript = format_transcript(relevant, settings.agent_id)

        prompt = compose_prompt(settings, transcript)
        result = run_cursor_agent(settings, prompt)

        status_body = ""
        if result.status_file and result.status_file.exists():
            status_body = result.status_file.read_text(encoding="utf-8").strip()
        if not status_body:
            status_body = (
                f"{settings.tag} TICK\n"
                f"SUMMARY: {result.summary}\n"
                f"ACTIONS: runner mode={result.mode} ok={result.ok}\n"
            )
        if not status_body.lstrip().startswith("["):
            status_body = f"{settings.tag}\n{status_body}"

        tg.send_message(status_body)
        posted = True

        if next_offset is not None:
            store.telegram_offset = next_offset

        details = {
            "ts": time.time(),
            "messages": len(messages),
            "relevant": len(relevant),
            "heartbeat": heartbeat,
            "ok": result.ok,
            "mode": result.mode,
            "agent_run_id": result.agent_run_id,
            "error": result.error,
        }
        store.record_tick(details)
        return TickResult(
            skipped=False,
            reason=None,
            messages_seen=len(messages),
            posted=posted,
            details=details,
        )
    finally:
        store.release_lock()
