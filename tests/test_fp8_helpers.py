"""CPU tests for FP8 baseline helpers."""

from prioritykv.fp8_baseline import build_local_calib_messages


def test_local_calib_count():
    rows = build_local_calib_messages(10)
    assert len(rows) == 10
    assert "messages" in rows[0]
    assert rows[0]["messages"][0]["role"] == "system"
