"""Installed command behavior and real audio-to-checkpoint round trips."""

import json
import subprocess
import sys

import pytest

from speechturn.cli import main


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
