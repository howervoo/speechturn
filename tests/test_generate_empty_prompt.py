"""Regression: generate() rejects empty or all-pad prompt_ids."""

import pytest

torch = pytest.importorskip("torch")

from speechturn.config import SpeechConfig
from speechturn.model import SpeechLanguageModel


@pytest.fixture()
def model():
    return SpeechLanguageModel(SpeechConfig())


def _dummy_audio(batch: int, frames: int, config: SpeechConfig):
    features = torch.zeros((batch, frames, config.n_mels), dtype=torch.float32)
    lengths = torch.full((batch,), frames, dtype=torch.long)
    return features, lengths


def test_generate_rejects_empty_prompt(model):
    features, lengths = _dummy_audio(1, 8, model.config)
    empty = torch.zeros((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="at least one token"):
        model.generate(features, lengths, prompt_ids=empty)


def test_generate_rejects_all_pad_prompt(model):
    features, lengths = _dummy_audio(1, 8, model.config)
    pads = torch.zeros((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="non-pad token"):
        model.generate(features, lengths, prompt_ids=pads)


def test_generate_accepts_valid_prompt(model):
    features, lengths = _dummy_audio(1, 8, model.config)
    valid = torch.ones((1, 1), dtype=torch.long)
    tokens = model.generate(features, lengths, prompt_ids=valid, max_new_tokens=2)
    assert tokens.shape == (1, 2)
