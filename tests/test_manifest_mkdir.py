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
