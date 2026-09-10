"""Regression: error stream events require non-empty payload."""

import pytest

from speechturn.streaming import StreamEvent


def test_error_event_rejects_empty_payload():
    """Error events must carry a meaningful error message."""
    with pytest.raises(ValueError, match="non-empty payload"):
        StreamEvent(sequence=0, kind="error", timestamp=1.0, payload="")


def test_error_event_rejects_whitespace_payload():
    """Error events with only whitespace are also rejected."""
    with pytest.raises(ValueError, match="non-empty payload"):
        StreamEvent(sequence=0, kind="error", timestamp=1.0, payload="   ")


def test_error_event_accepts_valid_payload():
    """Error events with actual content are accepted."""
    event = StreamEvent(
        sequence=0, kind="error", timestamp=1.0, payload="connection timeout"
    )
    assert event.payload == "connection timeout"


def test_other_events_accept_empty_payload():
    """Non-error events can have empty payloads."""
    start = StreamEvent(sequence=0, kind="start", timestamp=0.0)
    assert start.payload == ""

    final = StreamEvent(sequence=1, kind="final", timestamp=2.0)
    assert final.payload == ""
