"""Audio-instruction preparation and loss-aware padded batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from speechturn.audio import crop_audio, log_mel, read_audio
from speechturn.config import SpeechConfig
from speechturn.data import AudioExample
from speechturn.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class PreparedExample:
    id: str
    features: np.ndarray
    instruction: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("prepared examples need an identifier")
        if (
            self.features.ndim != 2
            or min(self.features.shape) < 1
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("features must be a non-empty finite matrix")
        ByteTokenizer().training_pair(self.instruction, self.text)


@dataclass(frozen=True)
class SpeechBatch:
    features: torch.Tensor
    feature_lengths: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    ids: list[str]

    def to(self, device: str | torch.device) -> SpeechBatch:
        """Move every tensor while retaining example identity and integer dtypes."""
        return SpeechBatch(
            self.features.to(device),
            self.feature_lengths.to(device),
            self.input_ids.to(device),
            self.labels.to(device),
            self.attention_mask.to(device),
            list(self.ids),
        )


def prepare_example(
    record: AudioExample, root: str | Path, config: SpeechConfig | None = None
) -> PreparedExample:
    settings = config or SpeechConfig()
    try:
        audio = read_audio(Path(root) / record.audio, settings.sample_rate)
        cropped = crop_audio(audio, settings.sample_rate, record.start, record.duration)
    except (OSError, ValueError) as error:
        raise ValueError(f"example {record.id!r}: {error}") from error
    return PreparedExample(record.id, log_mel(cropped, settings), record.instruction, record.text)


def collate_examples(
    examples: Sequence[PreparedExample], config: SpeechConfig | None = None
) -> SpeechBatch:
    """Pad features and teacher-forced tokens with masks matching the decoder contract."""
    settings = config or SpeechConfig()
    if not examples:
        raise ValueError("cannot collate an empty batch")
    if any(example.features.shape[1] != settings.n_mels for example in examples):
        raise ValueError("feature bands do not match n_mels")
    tokenizer = ByteTokenizer()
    pairs = [tokenizer.training_pair(example.instruction, example.text) for example in examples]
    frames = max(len(example.features) for example in examples)
    tokens = max(len(pair[0]) for pair in pairs)
    if (frames + settings.downsample - 1) // settings.downsample > settings.max_audio_tokens:
        raise ValueError("batch audio exceeds max_audio_tokens")
    if tokens > settings.max_text_tokens:
        raise ValueError("batch text exceeds max_text_tokens")
    features = torch.zeros((len(examples), frames, settings.n_mels), dtype=torch.float32)
    input_ids = torch.zeros((len(examples), tokens), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    lengths = []
    for index, (example, (inputs, targets)) in enumerate(zip(examples, pairs, strict=True)):
        length = len(example.features)
        lengths.append(length)
        features[index, :length] = torch.from_numpy(
            np.array(example.features, dtype=np.float32, order="C", copy=True)
        )
        input_ids[index, : len(inputs)] = torch.tensor(inputs, dtype=torch.long)
        labels[index, : len(targets)] = torch.tensor(targets, dtype=torch.long)
    return SpeechBatch(
        features,
        torch.tensor(lengths, dtype=torch.long),
        input_ids,
        labels,
        input_ids.ne(0),
        [example.id for example in examples],
    )


__all__ = ["PreparedExample", "SpeechBatch", "collate_examples", "prepare_example"]
