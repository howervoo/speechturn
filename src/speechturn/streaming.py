"""Lossless audio chunking and an explicit streaming event protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from speechturn.config import finite_number, positive_integer

TRANSITIONS = {
    "idle": {"start": "open"},
    "open": {"audio": "open", "text": "open", "final": "closed", "error": "failed"},
    "closed": {},
    "failed": {},
}


class AudioChunker:
    """Buffer arbitrary mono chunks; flush emits the remaining samples once."""

    def __init__(self, chunk_samples: int) -> None:
        self.chunk_samples = positive_integer(chunk_samples, "chunk_samples")
        self.buffer = np.empty(0, dtype=np.float32)
        self.closed = False

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        if self.closed:
            raise ValueError("the chunker is closed")
        values = np.asarray(samples)
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
        ):
            raise ValueError("chunks must be mono real numeric arrays")
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("chunks must contain finite samples")
        combined = np.concatenate((self.buffer, values))
        count = len(combined) // self.chunk_samples
        result = [
            combined[index * self.chunk_samples : (index + 1) * self.chunk_samples].copy()
            for index in range(count)
        ]
        self.buffer = combined[count * self.chunk_samples :].copy()
        return result

    def flush(self) -> list[np.ndarray]:
        if self.closed:
            raise ValueError("the chunker is closed")
        self.closed = True
        result = [self.buffer.copy()] if self.buffer.size else []
        self.buffer = np.empty(0, dtype=np.float32)
        return result


@dataclass(frozen=True)
class StreamEvent:
    sequence: int
    kind: str
    timestamp: float
    payload: str = ""

    def __post_init__(self) -> None:
        positive_integer(self.sequence, "sequence", 0)
        finite_number(self.timestamp, "timestamp")
        if not isinstance(self.kind, str) or self.kind not in {
            "start",
            "audio",
            "text",
            "final",
            "error",
        }:
            raise ValueError("unknown event kind")
        if not isinstance(self.payload, str):
            raise ValueError("payload must be text")
        if self.kind == "error" and not self.payload.strip():
            raise ValueError("error events require a non-empty payload")


class StreamSession:
    """Validate ordering before mutating state; timestamps use a monotonic clock."""

    def __init__(self) -> None:
        self.state = "idle"
        self.events: list[StreamEvent] = []

    def accept(self, event: StreamEvent) -> None:
        if event.sequence != len(self.events):
            raise ValueError("event sequence must be contiguous and start at zero")
        if self.events and event.timestamp < self.events[-1].timestamp:
            raise ValueError("timestamps must be nondecreasing")
        if event.kind not in TRANSITIONS[self.state]:
            raise ValueError(f"cannot accept {event.kind} in state {self.state}")
        self.state = TRANSITIONS[self.state][event.kind]
        self.events.append(event)

    def statistics(self, audio_seconds: float) -> dict[str, Any]:
        """Report wall-clock latency and processing time / input audio duration."""
        finite_number(audio_seconds, "audio_seconds")
        if not self.events:
            raise ValueError("session has not started")
        start = self.events[0].timestamp
        texts = [event.timestamp - start for event in self.events if event.kind == "text"]
        elapsed = self.events[-1].timestamp - start
        return {
            "state": self.state,
            "events": len(self.events),
            "elapsed_seconds": elapsed,
            "first_text_seconds": texts[0] if texts else None,
            "final_seconds": elapsed if self.state == "closed" else None,
            "real_time_factor": elapsed / audio_seconds if audio_seconds > 0 else None,
        }


__all__ = ["AudioChunker", "StreamEvent", "StreamSession", "TRANSITIONS"]
