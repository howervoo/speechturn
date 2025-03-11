"""Manifest diagnostics and leakage-free complete dataset partitions."""

import json
from dataclasses import fields

from speechturn.data import AudioExample, load_manifest, save_manifest


def records():
    return [
        AudioExample(f"id-{i}", "sample.wav", "Q", str(i), speaker=f"s-{i // 2}") for i in range(20)
    ]


def test_manifest_full_field_round_trip(tmp_path):
    record = AudioExample("中文", "audio.wav", "转写", "你好", "speaker", "zh", 1.5, 2.5)
    path = tmp_path / "data.jsonl"
    save_manifest([record], path)
    assert load_manifest(path) == [record]
    assert set(json.loads(path.read_text())) == {field.name for field in fields(AudioExample)}
