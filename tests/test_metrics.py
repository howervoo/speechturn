"""Independent edit-distance examples and whole-corpus weighting."""

import itertools

from speechturn.metrics import (
    aggregate_counts,
    character_error_rate,
    edit_counts,
    normalize_text,
    word_error_rate,
)


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


def test_normalization_is_explicit():
    assert normalize_text("Ａ， B!", lowercase=True, remove_punctuation=True) == "a b"
    assert word_error_rate("A", "a").rate == 1


def test_corpus_rate_weights_reference_lengths():
    score = aggregate_counts([word_error_rate("a b c", "a b c"), word_error_rate("x", "z")])
    assert score.rate == 0.25


def test_exhaustive_small_edit_distances():

    def distance(a, b):
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            table[i][0] = i
        for j in range(len(b) + 1):
            table[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                table[i][j] = min(
                    table[i - 1][j] + 1,
                    table[i][j - 1] + 1,
                    table[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                )
        return table[-1][-1]

    strings = [
        "".join(value) for size in range(4) for value in itertools.product("ab", repeat=size)
    ]
    for a, b in itertools.product(strings, repeat=2):
        counts = edit_counts(a, b)
        assert counts.errors == distance(a, b)
        assert len(a) - counts.deletions + counts.insertions == len(b)


def test_metric_serialization_complete():
    assert word_error_rate("a", "b").to_dict() == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
        "reference_length": 1,
        "hypothesis_length": 1,
        "errors": 1,
        "rate": 1.0,
    }
