from __future__ import annotations

from collab_bridge.protocol import detect_stop, parse_message
from collab_bridge.telegram_client import TelegramMessage


def _msg(text: str, *, is_bot: bool = False) -> TelegramMessage:
    return TelegramMessage(
        update_id=1,
        message_id=10,
        chat_id="-1001",
        date=0,
        text=text,
        from_id=1,
        from_username="x",
        from_is_bot=is_bot,
        raw={},
    )


def test_parse_claim_and_mention():
    p = parse_message(_msg("[agent:friend] CLAIM S1 — dense retrieval on MS MARCO @agent:arush please scaffold eval"))
    assert p.author_agent == "friend"
    assert "S1" in p.claims
    assert "arush" in p.mentions


def test_stop_keyword():
    assert detect_stop("please STOP_BRIDGE now", ["STOP_BRIDGE"]) == "STOP_BRIDGE"
    assert detect_stop("all good", ["STOP_BRIDGE"]) is None


if __name__ == "__main__":
    test_parse_claim_and_mention()
    test_stop_keyword()
    print("ok")
