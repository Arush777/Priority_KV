from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Settings
from .state_store import StateStore
from .telegram_client import TelegramClient
from .tick import run_tick


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_check(settings: Settings) -> int:
    errs = settings.validate_for_telegram()
    if errs:
        _print("FAIL telegram config:")
        for e in errs:
            _print(f"  - {e}")
        return 1
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    me = tg.get_me()
    _print(f"OK bot=@{me.get('username')} id={me.get('id')}")
    _print(f"OK agent_id={settings.agent_id} chat_id={settings.telegram_chat_id}")
    _print(f"OK repo_root={settings.repo_root}")
    agent_errs = settings.validate_for_agent()
    if agent_errs:
        _print("WARN agent config (tick will fail until fixed):")
        for e in agent_errs:
            _print(f"  - {e}")
    else:
        _print("OK agent config")
    return 0


def cmd_ping(settings: Settings) -> int:
    errs = settings.validate_for_telegram()
    if errs:
        _print("FAIL: " + "; ".join(errs))
        return 1
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    tg.ping(settings.tag)
    _print("posted ping")
    return 0


def cmd_tick(settings: Settings) -> int:
    result = run_tick(settings)
    _print(
        f"tick skipped={result.skipped} reason={result.reason} "
        f"messages={result.messages_seen} posted={result.posted} details={result.details}"
    )
    return 0 if not result.details.get("error") else 2


def cmd_daemon(settings: Settings) -> int:
    interval = max(60, settings.tick_interval_sec)
    _print(f"daemon start agent={settings.agent_id} interval={interval}s")
    while True:
        try:
            result = run_tick(settings)
            _print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"skipped={result.skipped} reason={result.reason} "
                f"msgs={result.messages_seen} posted={result.posted}"
            )
        except Exception as exc:
            _print(f"tick error: {exc}")
            try:
                tg = TelegramClient(
                    settings.telegram_bot_token, settings.telegram_chat_id
                )
                tg.send_message(f"{settings.tag} tick ERROR: {exc}")
            except Exception:
                pass
        time.sleep(interval)


def cmd_resume(settings: Settings) -> int:
    store = StateStore(settings.state_dir, settings.agent_id)
    store.resume()
    _print("local pause cleared")
    if not settings.validate_for_telegram():
        tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
        tg.send_message(f"{settings.tag} local RESUME (CLI).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collab_bridge",
        description="Telegram ↔ Cursor collaboration bridge for Information_Retrieval",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help="Path to .env (default: repo .env)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Validate config + Telegram bot identity")
    sub.add_parser("ping", help="Post an online message to the group")
    sub.add_parser("tick", help="Run one collaboration tick")
    sub.add_parser("daemon", help="Loop ticks forever (hourly by default)")
    sub.add_parser("resume", help="Clear local pause flag")

    args = parser.parse_args(argv)
    settings = Settings.load(args.env)

    if args.cmd == "check":
        return cmd_check(settings)
    if args.cmd == "ping":
        return cmd_ping(settings)
    if args.cmd == "tick":
        return cmd_tick(settings)
    if args.cmd == "daemon":
        return cmd_daemon(settings)
    if args.cmd == "resume":
        return cmd_resume(settings)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
