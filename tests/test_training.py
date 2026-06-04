"""Real gradient steps and exact interrupted CPU training recovery."""

import random

import numpy as np
import pytest
import torch

from speechturn.batching import PreparedExample
from speechturn.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.model import SpeechLanguageModel
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


def test_resume_matches_uninterrupted_training(tmp_path):
    cfg = settings()
    full = train(
        examples(), tmp_path / "full.pt", cfg, TrainConfig(steps=4, gradient_accumulation=2)
    )
    train(examples(), tmp_path / "part.pt", cfg, TrainConfig(steps=2, gradient_accumulation=2))
    resumed = train(
        examples(),
        tmp_path / "resume.pt",
        cfg,
        TrainConfig(steps=4, gradient_accumulation=2),
        tmp_path / "part.pt",
    )
    assert full.losses[2:] == resumed.losses
    assert all(
        (
            torch.equal(value, resumed.model.state_dict()[name])
            for name, value in full.model.state_dict().items()
        )
    )


def test_checkpoint_preserves_all_model_tensors(tmp_path):
    model = SpeechLanguageModel(settings())
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, step=3)
    restored, optimizer, step = load_checkpoint(path)
    assert optimizer is None and step == 3
    assert restored.config == model.config
    assert all(
        (
            torch.equal(value, restored.state_dict()[name])
            for name, value in model.state_dict().items()
        )
    )


def test_checkpoint_schema_is_exhaustive(tmp_path):
    path = tmp_path / "state.pt"
    save_checkpoint(path, SpeechLanguageModel(settings()))
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"format", "config", "model", "optimizer", "step", "rng"}
    assert payload["format"] == CHECKPOINT_FORMAT
    assert set(payload["rng"]) == {"python", "numpy", "torch"}


def test_checkpoint_restores_three_rngs(tmp_path):
    seed_everything(11)
    path = tmp_path / "rng.pt"
    save_checkpoint(path, SpeechLanguageModel(settings()))
    expected = (random.random(), np.random.random(), torch.rand(2))
    load_checkpoint(path, restore_rng=True)
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])


def test_checkpoint_rejects_unknown_version(tmp_path):
    path = tmp_path / "bad.pt"
    save_checkpoint(path, SpeechLanguageModel(settings()))
    payload = torch.load(path, weights_only=True)
    payload["format"] = "future"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported"):
        load_checkpoint(path)
