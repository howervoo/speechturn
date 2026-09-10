"""Regression: prepare_example wraps errors with example ID context."""

from pathlib import Path

import pytest

from speechturn.batching import prepare_example
from speechturn.data import AudioExample


def test_prepare_example_missing_file_includes_id(tmp_path: Path):
    """Missing audio file error should include the example ID."""
    record = AudioExample(
        id="sample-42",
        audio="nonexistent.wav",
        instruction="transcribe",
        text="hello",
        speaker="spk1",
    )
    with pytest.raises(ValueError, match="sample-42"):
        prepare_example(record, tmp_path)


def test_prepare_example_invalid_audio_includes_id(tmp_path: Path):
    """Invalid audio file error should include the example ID."""
    # Create an invalid "audio" file
    bad_file = tmp_path / "bad.wav"
    bad_file.write_text("not audio data")

    record = AudioExample(
        id="sample-99",
        audio="bad.wav",
        instruction="transcribe",
        text="world",
        speaker="spk2",
    )
    with pytest.raises(ValueError, match="sample-99"):
        prepare_example(record, tmp_path)
