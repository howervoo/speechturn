"""Regression: train validates checkpoint path before starting."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from speechturn.batching import PreparedExample
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.training import train
import numpy as np


def _dummy_example():
    """Create a minimal valid example for training."""
    features = np.zeros((10, 40), dtype=np.float32)
    return PreparedExample(
        id="test-1",
        features=features,
        instruction="transcribe",
        text="hello",
    )


def test_train_creates_checkpoint_parent_dirs(tmp_path: Path):
    """train should create missing parent directories for checkpoint."""
    nested = tmp_path / "deep" / "nested" / "model.pt"
    example = _dummy_example()
    config = SpeechConfig()
    train_config = TrainConfig(steps=1, batch_size=1)

    result = train([example], nested, config, train_config)
    assert nested.exists()
    assert result.step == 1


def test_train_rejects_invalid_checkpoint_path():
    """train should fail early if checkpoint path is invalid."""
    example = _dummy_example()
    config = SpeechConfig()
    train_config = TrainConfig(steps=1, batch_size=1)

    # Try to write to a path that looks like a file (not a directory)
    # This should fail when trying to create parent directories
    with pytest.raises(ValueError, match="cannot create checkpoint directory"):
        train([example], "/dev/null/impossible/path/model.pt", config, train_config)
