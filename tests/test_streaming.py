"""Chunk preservation, event ordering and exhaustive state transitions."""

import itertools

import numpy as np

from speechturn.streaming import AudioChunker


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
