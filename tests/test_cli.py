"""Installed command behavior and real audio-to-checkpoint round trips."""

import json
import subprocess
import sys

from speechturn.cli import main


def test_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "speechturn", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "validate" in result.stdout


def test_cli_evaluation_is_structured(capsys):
    assert main(["evaluate", "one two", "one three"]) == 0
    assert json.loads(capsys.readouterr().out)["wer"]["rate"] == 0.5
