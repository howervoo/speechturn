"""Regression: save_checkpoint stores CPU tensors for portability."""

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from speechturn.checkpoint import load_checkpoint, save_checkpoint
from speechturn.config import SpeechConfig
from speechturn.model import SpeechLanguageModel


def test_checkpoint_tensors_are_cpu(tmp_path: Path):
    """Saved checkpoint should contain CPU tensors regardless of model device."""
    model = SpeechLanguageModel(SpeechConfig())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model)

    # Load without map_location to verify tensors are CPU
    payload = torch.load(checkpoint, weights_only=True)
    for name, tensor in payload["model"].items():
        assert tensor.device.type == "cpu", f"tensor {name} is on {tensor.device}"


def test_checkpoint_roundtrip_cpu(tmp_path: Path):
    """Verify checkpoint can be saved and loaded on CPU."""
    config = SpeechConfig()
    model = SpeechLanguageModel(config)
    checkpoint = tmp_path / "model.pt"

    save_checkpoint(checkpoint, model, step=10)
    loaded, _, step = load_checkpoint(checkpoint)

    assert step == 10
    assert loaded.config == config

    # Verify weights match
    for (name1, p1), (name2, p2) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert name1 == name2
        assert torch.equal(p1.cpu(), p2.cpu())
