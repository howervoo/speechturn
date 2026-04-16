"""Edit counts and corpus-weighted speech transcription error rates."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0
    hypothesis_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """For empty references, report insertions rather than divide by zero."""
        return self.errors / max(1, self.reference_length)

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "errors": self.errors, "rate": self.rate}


def normalize_text(text: str, lowercase: bool = False, remove_punctuation: bool = False) -> str:
    """Normalize Unicode and whitespace using an explicit evaluation policy."""
    if not isinstance(text, str):
        raise ValueError("transcripts must be text")
    result = unicodedata.normalize("NFKC", text)
    if lowercase:
        result = result.casefold()
    if remove_punctuation:
        result = "".join(char for char in result if not unicodedata.category(char).startswith("P"))
    return " ".join(result.split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Count minimum edits in linear working memory, preferring substitutions on ties."""
    # Each state stores cost, substitutions, deletions, insertions.
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for row, expected in enumerate(reference, 1):
        current = [(row, 0, row, 0)]
        for column, actual in enumerate(hypothesis, 1):
            if expected == actual:
                current.append(previous[column - 1])
                continue
            diagonal = previous[column - 1]
            above = previous[column]
            left = current[-1]
            choices = [
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            ]
            current.append(min(choices, key=lambda value: value[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference), len(hypothesis))


def word_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    """Evaluate whitespace-separated words with case and punctuation preserved."""
    return edit_counts(normalize_text(reference).split(), normalize_text(hypothesis).split())


def character_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    """Evaluate normalized Unicode characters, excluding whitespace."""
    return edit_counts(
        list(normalize_text(reference).replace(" ", "")),
        list(normalize_text(hypothesis).replace(" ", "")),
    )


def aggregate_counts(results: Sequence[ErrorCounts]) -> ErrorCounts:
    """Aggregate edit numerators and reference lengths before dividing."""
    return ErrorCounts(
        sum(item.substitutions for item in results),
        sum(item.deletions for item in results),
        sum(item.insertions for item in results),
        sum(item.reference_length for item in results),
        sum(item.hypothesis_length for item in results),
    )


__all__ = [
    "ErrorCounts",
    "aggregate_counts",
    "character_error_rate",
    "edit_counts",
    "normalize_text",
    "word_error_rate",
]
