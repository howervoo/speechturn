"""UTF-8 and prompt/answer alignment contracts."""

import pytest

from speechturn.tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["", "hello", "你好，深圳", "é é", "😀\n\t", "\x00"])
def test_utf8_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text
