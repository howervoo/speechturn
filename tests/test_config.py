"""Configuration serialization and cross-field invariants."""

from speechturn.config import SpeechConfig


def test_model_config_round_trip():
    settings = SpeechConfig(projection="linear", freeze_encoder=True, dropout=0.2)
    assert SpeechConfig.from_dict(settings.to_dict()) == settings
