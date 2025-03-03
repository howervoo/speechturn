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
