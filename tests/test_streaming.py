"""Chunk preservation, event ordering and exhaustive state transitions."""

import itertools

import numpy as np
import pytest

from speechturn.streaming import TRANSITIONS, AudioChunker


def test_every_small_chunk_partition_is_lossless():
    samples = np.arange(8, dtype=np.float32)
    for cuts in itertools.product([False, True], repeat=7):
        boundaries = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [8]
        chunker = AudioChunker(3)
        chunks = []
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            chunks.extend(chunker.feed(samples[left:right]))
        chunks.extend(chunker.flush())
        assert [len(chunk) for chunk in chunks] == [3, 3, 2]
        assert np.array_equal(np.concatenate(chunks), samples)


def test_empty_chunker_flush():
    chunker = AudioChunker(4)
    assert chunker.feed(np.array([])) == []
    assert chunker.flush() == []


def test_closed_chunker_rejects_both_operations():
    chunker = AudioChunker(4)
    chunker.flush()
    with pytest.raises(ValueError):
        chunker.feed(np.ones(2))
    with pytest.raises(ValueError):
        chunker.flush()


def test_input_and_output_do_not_alias_buffer():
    samples = np.arange(5, dtype=np.float32)
    chunker = AudioChunker(3)
    output = chunker.feed(samples)
    samples[:] = -1
    output[0][:] = -2
    assert chunker.flush()[0].tolist() == [3, 4]


def test_transition_table_golden():
    assert TRANSITIONS == {
        "idle": {"start": "open"},
        "open": {"audio": "open", "text": "open", "final": "closed", "error": "failed"},
        "closed": {},
        "failed": {},
    }
