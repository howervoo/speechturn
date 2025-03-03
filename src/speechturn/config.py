"""Validated configurations for the small speech-language reference model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    """Validate an integer without accepting booleans or lossy conversions."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite_number(value: Any, name: str, minimum: float = 0.0) -> float:
    """Validate a finite numerical setting without coercing text or booleans."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class SpeechConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    encoder_dim: int = 32
    model_dim: int = 48
    num_heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 1
    downsample: int = 2
    vocab_size: int = 260
    max_audio_tokens: int = 512
    max_text_tokens: int = 256
    dropout: float = 0.0
    projection: str = "mlp"
    freeze_encoder: bool = False
    freeze_decoder: bool = False

    def __post_init__(self) -> None:
        for name in (
            "sample_rate",
            "n_fft",
            "win_length",
            "hop_length",
            "n_mels",
            "encoder_dim",
            "model_dim",
            "num_heads",
            "encoder_layers",
            "decoder_layers",
            "downsample",
            "vocab_size",
            "max_audio_tokens",
            "max_text_tokens",
        ):
            positive_integer(getattr(self, name), name)
        finite_number(self.dropout, "dropout")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1")
        if self.n_fft < self.win_length or self.hop_length > self.win_length:
            raise ValueError("require hop_length <= win_length <= n_fft")
        if self.n_mels > self.n_fft // 2 + 1:
            raise ValueError("n_mels must not exceed the available frequency bins")
        if self.encoder_dim % self.num_heads or self.model_dim % self.num_heads:
            raise ValueError("attention dimensions must be divisible by num_heads")
        if self.vocab_size < 260 or self.max_text_tokens < 2:
            raise ValueError("the byte tokenizer needs 260 tokens and at least two text positions")
        if self.projection not in ("linear", "mlp"):
            raise ValueError("projection must be linear or mlp")
        if type(self.freeze_encoder) is not bool or type(self.freeze_decoder) is not bool:
            raise ValueError("freeze settings must be booleans")

    def to_dict(self) -> dict[str, Any]:
        """Return every configuration field for a reproducible checkpoint."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SpeechConfig:
        """Load declared fields and reject unknown configuration keys."""
        if not isinstance(values, dict):
            raise ValueError("configuration must be an object")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError("unknown or invalid configuration field") from error


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    batch_size: int = 2
    steps: int = 20
    gradient_accumulation: int = 1
    save_every: int = 0
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        positive_integer(self.seed, "seed", 0)
        positive_integer(self.save_every, "save_every", 0)
        for name in ("batch_size", "steps", "gradient_accumulation"):
            positive_integer(getattr(self, name), name)
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            finite_number(getattr(self, name), name)
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["SpeechConfig", "TrainConfig", "finite_number", "positive_integer"]
