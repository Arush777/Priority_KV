from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests


@dataclass
class TelegramMessage:
    update_id: int
    message_id: int
    chat_id: str
    date: int
    text: str
    from_id: int | None
    from_username: str | None
    from_is_bot: bool
    raw: dict[str, Any]

    @property
    def is_from_bot(self) -> bool:
        return self.from_is_bot


class TelegramClient:
    """Thin Telegram Bot API client (getUpdates long-poll + sendMessage)."""

    def __init__(self, token: str, chat_id: str, timeout: int = 30):
        self.token = token
        self.chat_id = str(chat_id)
        self.timeout = timeout
        self.base = f"https://api.telegram.org/bot{token}"

    def _call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base}/{method}"
        resp = requests.post(url, json=payload or {}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {data}")
        return data["result"]

    def get_me(self) -> dict[str, Any]:
        return self._call("getMe")

    def send_message(
        self,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        disable_preview: bool = True,
    ) -> dict[str, Any]:
        # Telegram hard limit ~4096 chars
        chunks = _chunk_text(text, 4000)
        last: dict[str, Any] = {}
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": self.chat_id,
                "text": chunk,
                "disable_web_page_preview": disable_preview,
            }
            if i == 0 and reply_to_message_id is not None:
                payload["reply_to_message_id"] = reply_to_message_id
            last = self._call("sendMessage", payload)
            if i < len(chunks) - 1:
                time.sleep(0.35)
        return last

    def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "limit": limit,
            "timeout": timeout,
            "allowed_updates": allowed_updates or ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        return self._call("getUpdates", payload)

    def fetch_new_messages(
        self,
        offset: int | None,
        *,
        limit: int = 100,
    ) -> tuple[list[TelegramMessage], int | None]:
        """Return messages for our chat_id and the next offset to persist."""
        updates = self.get_updates(offset=offset, limit=limit, timeout=0)
        messages: list[TelegramMessage] = []
        next_offset = offset
        for upd in updates:
            uid = int(upd["update_id"])
            next_offset = uid + 1 if next_offset is None else max(next_offset, uid + 1)
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat = msg.get("chat") or {}
            if str(chat.get("id")) != self.chat_id:
                continue
            text = msg.get("text") or msg.get("caption") or ""
            if not text.strip():
                continue
            frm = msg.get("from") or {}
            messages.append(
                TelegramMessage(
                    update_id=uid,
                    message_id=int(msg["message_id"]),
                    chat_id=str(chat.get("id")),
                    date=int(msg.get("date") or 0),
                    text=text,
                    from_id=frm.get("id"),
                    from_username=frm.get("username"),
                    from_is_bot=bool(frm.get("is_bot")),
                    raw=upd,
                )
            )
        return messages, next_offset

    def ping(self, agent_tag: str) -> None:
        self.send_message(f"{agent_tag} bridge online — ready to collaborate.")


def _chunk_text(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        parts.append(remaining[:size])
        remaining = remaining[size:]
    return parts


def message_to_dict(m: TelegramMessage) -> dict[str, Any]:
    d = asdict(m)
    d.pop("raw", None)
    return d


def dumps_messages(messages: list[TelegramMessage]) -> str:
    return json.dumps([message_to_dict(m) for m in messages], indent=2)
