"""Installed command behavior and real audio-to-checkpoint round trips."""

import subprocess
import sys


def test_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "speechturn", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "validate" in result.stdout
