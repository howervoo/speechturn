"""Validated audio-instruction manifests and speaker-disjoint dataset splits."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from speechturn.config import finite_number, positive_integer


@dataclass(frozen=True)
class AudioExample:
    id: str
    audio: str
    instruction: str
    text: str
    speaker: str = "unknown"
    language: str = "und"
    start: float = 0.0
    duration: float | None = None

    def __post_init__(self) -> None:
        for name in ("id", "audio", "instruction", "speaker", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be non-empty text without NUL characters")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string, including an empty transcript for silence")
        finite_number(self.start, "start")
        if self.duration is not None:
            if finite_number(self.duration, "duration") == 0:
                raise ValueError("duration must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AudioExample:
        if not isinstance(values, dict):
            raise ValueError("manifest records must be objects")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError("missing or unknown manifest fields") from error


def load_manifest(path: str | Path, check_audio: bool = False) -> list[AudioExample]:
    """Read JSONL in order and reject duplicate IDs with line-local diagnostics."""
    source = Path(path)
    records = []
    seen = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            example = AudioExample.from_dict(json.loads(line))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{source.name}:{number}: invalid audio example: {error}") from error
        if example.id in seen:
            raise ValueError(f"{source.name}:{number}: duplicate example ID {example.id!r}")
        if check_audio and not (source.parent / example.audio).is_file():
            raise ValueError(f"{source.name}:{number}: audio file does not exist")
        seen.add(example.id)
        records.append(example)
    if not records:
        raise ValueError("manifest contains no examples")
    return records


def save_manifest(records: Sequence[AudioExample], path: str | Path) -> None:
    """Write all declared fields as UTF-8 JSONL without silently losing duplicate IDs."""
    if not records or len({record.id for record in records}) != len(records):
        raise ValueError("manifest must be non-empty and have unique IDs")
    text = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    Path(path).write_text(text, encoding="utf-8")


def split_by_speaker(
    records: Sequence[AudioExample],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 7,
) -> dict[str, list[AudioExample]]:
    """Split complete speaker groups, retaining all examples exactly once."""
    positive_integer(seed, "seed", 0)
    if len(fractions) != 3 or not records:
        raise ValueError("three split fractions and a non-empty dataset are required")
    ratios = [finite_number(value, "fraction") for value in fractions]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    if len({record.id for record in records}) != len(records):
        raise ValueError("example IDs must be unique")
    speakers = sorted({record.speaker for record in records})
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(speakers) < len(active):
        raise ValueError("not enough speakers for the requested non-empty splits")
    random.Random(seed).shuffle(speakers)
    counts = [int(len(speakers) * ratio) for ratio in ratios]
    residual = len(speakers) - sum(counts)
    priority = sorted(
        active, key=lambda index: (-(len(speakers) * ratios[index] - counts[index]), index)
    )
    for index in priority[:residual]:
        counts[index] += 1
    for index in active:
        if counts[index] == 0:
            donor = max(active, key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[index] += 1
    ownership = {}
    cursor = 0
    names = ("train", "validation", "test")
    for name, count in zip(names, counts, strict=True):
        ownership.update({speaker: name for speaker in speakers[cursor : cursor + count]})
        cursor += count
    result: dict[str, list[AudioExample]] = {name: [] for name in names}
    for record in records:
        result[ownership[record.speaker]].append(record)
    return result


__all__ = ["AudioExample", "load_manifest", "save_manifest", "split_by_speaker"]
