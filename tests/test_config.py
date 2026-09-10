"""Configuration serialization and cross-field invariants."""

from dataclasses import fields

import pytest

from speechturn.config import SpeechConfig, TrainConfig


def test_model_config_round_trip():
    settings = SpeechConfig(projection="linear", freeze_encoder=True, dropout=0.2)
    assert SpeechConfig.from_dict(settings.to_dict()) == settings


def test_train_config_has_complete_serialization():
    settings = TrainConfig()
    assert set(settings.to_dict()) == {field.name for field in fields(TrainConfig)}


def test_configuration_rejects_unknown_fields():
    with pytest.raises(ValueError):
        SpeechConfig.from_dict({"unknown": 1})


@pytest.mark.parametrize(
    "values",
    [
        {"hop_length": 401},
        {"win_length": 401},
        {"n_mels": 202},
        {"model_dim": 49},
        {"encoder_dim": 33},
        {"vocab_size": 259},
        {"max_text_tokens": 1},
    ],
)
def test_cross_field_constraints(values):
    with pytest.raises(ValueError):
        SpeechConfig(**values)


def test_configuration_is_immutable():
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        SpeechConfig().n_mels = 80


def test_train_config_allows_zero_save_every():
    """save_every=0 disables intermediate checkpoints and must be accepted."""
    settings = TrainConfig(save_every=0)
    assert settings.save_every == 0


def test_train_config_allows_zero_seed():
    """seed=0 is a valid deterministic seed value."""
    settings = TrainConfig(seed=0)
    assert settings.seed == 0


def test_train_config_rejects_zero_learning_rate():
    """learning_rate=0 would prevent training progress."""
    with pytest.raises(ValueError, match="learning_rate and max_grad_norm must be positive"):
        TrainConfig(learning_rate=0)


def test_train_config_rejects_zero_max_grad_norm():
    """max_grad_norm=0 would zero out all gradients."""
    with pytest.raises(ValueError, match="learning_rate and max_grad_norm must be positive"):
        TrainConfig(max_grad_norm=0)


def test_train_config_is_immutable():
    """TrainConfig must be frozen like SpeechConfig."""
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        TrainConfig().steps = 100
