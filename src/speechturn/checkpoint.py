"""Atomic tensor checkpoints with explicit schema and reproducible RNG state."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from speechturn.config import SpeechConfig, positive_integer
from speechturn.model import SpeechLanguageModel

CHECKPOINT_FORMAT = "speechturn.checkpoint.v1"


def seed_everything(seed: int) -> None:
    positive_integer(seed, "seed", 0)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def save_checkpoint(
    path: str | Path,
    model: SpeechLanguageModel,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    """Store tensors and primitive metadata without serializing model objects."""
    positive_integer(step, "step", 0)
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise RuntimeError("NumPy legacy RNG state must be a tuple")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "model": {key: tensor.cpu() for key, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": [
                numpy_state[0],
                numpy_state[1].tolist(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            ],
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    restore_rng: bool = False,
) -> tuple[SpeechLanguageModel, dict[str, Any] | None, int]:
    """Load tensor-only state on CPU and optionally restore all recorded RNGs."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "format",
        "config",
        "model",
        "optimizer",
        "step",
        "rng",
    }:
        raise ValueError("invalid checkpoint schema")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    positive_integer(payload["step"], "step", 0)
    for tensor in payload["model"].values():
        if not isinstance(tensor, torch.Tensor) or (
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
        ):
            raise ValueError("checkpoint weights must be finite tensors")
    model = SpeechLanguageModel(SpeechConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"])
        name, keys, position, gaussian, cached = payload["rng"]["numpy"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    return model, payload["optimizer"], payload["step"]


__all__ = ["CHECKPOINT_FORMAT", "load_checkpoint", "save_checkpoint", "seed_everything"]
