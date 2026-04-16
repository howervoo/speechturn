"""Independent edit-distance examples and whole-corpus weighting."""

from speechturn.metrics import character_error_rate, word_error_rate


def test_substitution_deletion_insertion_counts():
    assert word_error_rate("a b", "a c").substitutions == 1
    assert word_error_rate("a b", "a").deletions == 1
    assert word_error_rate("a", "a b").insertions == 1


def test_empty_reference_and_hypothesis():
    assert word_error_rate("", "").rate == 0
    assert word_error_rate("", "a b").rate == 2
    assert word_error_rate("a b", "").rate == 1


def test_chinese_character_error():
    assert character_error_rate("你好 深圳", "你好广州").rate == 0.5
