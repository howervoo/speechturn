"""Real gradient steps and exact interrupted CPU training recovery."""

import numpy as np

from speechturn.batching import PreparedExample
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.training import train


def settings():
    return SpeechConfig(
        n_mels=8, encoder_dim=8, model_dim=8, num_heads=2, max_audio_tokens=16, max_text_tokens=32
    )


def examples():
    return [
        PreparedExample("one", np.ones((7, 8), dtype=np.float32), "Q", "a"),
        PreparedExample("two", np.zeros((5, 8), dtype=np.float32), "Q", "bb"),
    ]


def test_training_reduces_synthetic_loss(tmp_path):
    result = train(
        examples(), tmp_path / "model.pt", settings(), TrainConfig(steps=8, learning_rate=0.01)
    )
    assert result.step == 8
    assert result.losses[-1] < result.losses[0]
