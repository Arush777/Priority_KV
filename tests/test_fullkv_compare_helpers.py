"""CPU tests for FullKV compare helpers."""

from prioritykv.fullkv_compare import token_agree


def test_token_agree_identical():
    assert token_agree([1, 2, 3], [1, 2, 3]) == 1.0


def test_token_agree_partial():
    assert token_agree([1, 2, 3], [1, 2, 9]) == 2 / 3


def test_token_agree_empty():
    assert token_agree([], []) == 1.0
    assert token_agree([1], []) == 0.0
