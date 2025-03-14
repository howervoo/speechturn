"""Manifest diagnostics and leakage-free complete dataset partitions."""

import json
from dataclasses import fields

import pytest

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


def test_duplicate_ids_rejected_on_write(tmp_path):
    record = records()[0]
    with pytest.raises(ValueError):
        save_manifest([record, record], tmp_path / "bad.jsonl")


def test_duplicate_ids_rejected_on_read(tmp_path):
    path = tmp_path / "data.jsonl"
    save_manifest([records()[0]], path)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match=":2: duplicate"):
        load_manifest(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("\n{bad}\n")
    with pytest.raises(ValueError, match="broken.jsonl:2:"):
        load_manifest(path)
