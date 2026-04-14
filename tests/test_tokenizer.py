"""UTF-8 and prompt/answer alignment contracts."""

import pytest

from speechturn.tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["", "hello", "你好，深圳", "é é", "😀\n\t", "\x00"])
def test_utf8_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_full_byte_vocabulary():
    tokenizer = ByteTokenizer()
    for code in range(128):
        assert tokenizer.encode(chr(code)) == [code + 4]


def test_prompt_labels_are_masked():
    tokenizer = ByteTokenizer()
    inputs, labels = tokenizer.training_pair("Q", "yes")
    prefix = tokenizer.prompt("Q")
    assert inputs[: len(prefix)] == prefix
    assert labels[: len(prefix) - 1] == [-100] * (len(prefix) - 1)
    assert labels[len(prefix) - 1 :] == tokenizer.encode("yes") + [2]


def test_empty_answer_still_supervises_eos():
    _, labels = ByteTokenizer().training_pair("listen", "")
    assert [label for label in labels if label != -100] == [2]
