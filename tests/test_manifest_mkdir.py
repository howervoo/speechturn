"""Regression: save_manifest creates parent directories."""

import json
from pathlib import Path

import pytest

from speechturn.data import AudioExample, save_manifest


def test_save_manifest_creates_parent_dirs(tmp_path: Path):
    """save_manifest should create missing parent directories."""
    nested = tmp_path / "deep" / "nested" / "dir" / "manifest.jsonl"
    records = [
        AudioExample(
            id="test-1",
            audio="audio.wav",
            instruction="transcribe",
            text="hello",
            speaker="spk1",
        )
    ]
    save_manifest(records, nested)
    assert nested.exists()
    content = json.loads(nested.read_text().strip())
    assert content["id"] == "test-1"


def test_save_manifest_existing_dir(tmp_path: Path):
    """save_manifest works when parent already exists."""
    target = tmp_path / "manifest.jsonl"
    records = [
        AudioExample(
            id="test-2",
            audio="audio.wav",
            instruction="transcribe",
            text="world",
            speaker="spk2",
        )
    ]
    save_manifest(records, target)
    assert target.exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_manifest_does_not_create_directories(tmp_path: Path, duplicate):
    record = AudioExample("duplicate", "audio.wav", "Q", "a")
    records = [record, record] if duplicate else []
    target = tmp_path / "missing" / "nested" / "manifest.jsonl"
    with pytest.raises(ValueError, match="non-empty and have unique IDs"):
        save_manifest(records, target)
    assert not (tmp_path / "missing").exists()


def test_save_manifest_preserves_conflicting_parent_file(tmp_path: Path):
    parent = tmp_path / "parent"
    parent.write_text("existing data", encoding="utf-8")
    record = AudioExample("test", "audio.wav", "Q", "a")
    with pytest.raises(OSError):
        save_manifest([record], parent / "manifest.jsonl")
    assert parent.read_text(encoding="utf-8") == "existing data"


def test_save_manifest_preserves_conflicting_output_directory(tmp_path: Path):
    target = tmp_path / "manifest.jsonl"
    target.mkdir()
    sentinel = target / "existing.txt"
    sentinel.write_text("existing data", encoding="utf-8")
    record = AudioExample("test", "audio.wav", "Q", "a")
    with pytest.raises(OSError):
        save_manifest([record], target)
    assert sentinel.read_text(encoding="utf-8") == "existing data"
