"""Installed command behavior and real audio-to-checkpoint round trips."""

import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from speechturn.cli import main
from speechturn.data import AudioExample, save_manifest


def test_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "speechturn", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "validate" in result.stdout


def test_cli_evaluation_is_structured(capsys):
    assert main(["evaluate", "one two", "one three"]) == 0
    assert json.loads(capsys.readouterr().out)["wer"]["rate"] == 0.5


def test_cli_invalid_manifest_returns_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["validate", str(tmp_path / "missing.jsonl")])
    assert error.value.code == 2
    assert "speechturn:" in capsys.readouterr().err


def test_cli_split_paths_remain_valid(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "tone.wav", np.zeros(400), 16000)
    records = [AudioExample(str(i), "tone.wav", "Q", "a", speaker=str(i)) for i in range(3)]
    manifest = data / "input.jsonl"
    save_manifest(records, manifest)
    output = tmp_path / "splits"
    assert main(["split", str(manifest), str(output)]) == 0
    capsys.readouterr()
    for path in output.glob("*.jsonl"):
        assert main(["validate", str(path)]) == 0
