"""Reviewed invalid-input vectors cover every public serialized setting."""

import json
from pathlib import Path

import pytest

from speechturn.config import SpeechConfig, TrainConfig
from speechturn.data import AudioExample
from speechturn.streaming import StreamEvent

CASES = json.loads((Path(__file__).parent / "fixtures/input-contracts.json").read_text())
CONSTRUCTORS = {
    "SpeechConfig": SpeechConfig,
    "TrainConfig": TrainConfig,
    "AudioExample": AudioExample,
    "StreamEvent": StreamEvent,
}
DEFAULTS = {
    "SpeechConfig": {},
    "TrainConfig": {},
    "AudioExample": {"id": "id", "audio": "a.wav", "instruction": "Q", "text": "a"},
    "StreamEvent": {"sequence": 0, "kind": "start", "timestamp": 0.0},
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_reject_invalid_serialized_input(case):
    value = case["value"]
    if isinstance(value, dict) and set(value) == {"float"}:
        value = float(value["float"])
    values = {**DEFAULTS[case["class"]], case["field"]: value}
    with pytest.raises(ValueError):
        CONSTRUCTORS[case["class"]](**values)
