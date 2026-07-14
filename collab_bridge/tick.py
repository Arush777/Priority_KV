from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .agent_runner import compose_prompt, run_cursor_agent
from .config import Settings
from .memory import CollabMemory, format_new_and_ring
from .protocol import ControlAction, detect_stop, parse_message
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
    memory = CollabMemory(
        settings.state_dir,
        settings.repo_root,
        settings.agent_id,
        window=settings.memory_window,
    )
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

        # Sticky ring: keep last N including own posts (for continuity)
        memory.append_telegram_messages(messages)

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
        ring_tx = memory.ring_as_transcript(settings.agent_id)
        if heartbeat:
            new_part = (
                "(no brand-new peer/human messages)\n"
                "Heartbeat: use sticky memory + unanswered ASKs; either answer a peer "
                "ASK, advance a CLAIMed workstream, or ask one concrete question.\n"
            )
            transcript = (
                "## Brand-new since last poll\n"
                f"{new_part}\n"
                f"## Last N Telegram messages (sticky ring)\n{ring_tx}\n"
            )
        else:
            transcript = format_new_and_ring(relevant, ring_tx, settings.agent_id)

        prior = memory.get_cursor_agent_id()
        will_try_resume = bool(settings.use_agent_resume and prior)
        memory_blob = memory.read_summary()
        prompt = compose_prompt(
            settings,
            transcript,
            memory_blob=memory_blob,
            resumed=will_try_resume,
        )
        result = run_cursor_agent(settings, prompt, memory=memory)

        status_body = ""
        if result.status_file and result.status_file.exists():
            status_body = result.status_file.read_text(encoding="utf-8").strip()
        if not status_body:
            status_body = (
                f"{settings.tag} TICK\n"
                f"SUMMARY: {result.summary}\n"
                f"ACTIONS: runner mode={result.mode} ok={result.ok} "
                f"resumed={result.resumed}\n"
            )
        if not status_body.lstrip().startswith("["):
            status_body = f"{settings.tag}\n{status_body}"

        memory.update_after_tick(
            status_body,
            new_msgs=len(messages),
            resumed=result.resumed,
        )

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
            "resumed": result.resumed,
            "cursor_agent_id": result.cursor_agent_id,
            "agent_run_id": result.agent_run_id,
            "error": result.error,
            "memory_window": settings.memory_window,
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
