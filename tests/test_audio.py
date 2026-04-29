"""Audio shape, anti-aliasing, channel and crop behavior."""

import numpy as np
import pytest

from speechturn.audio import (
    validate_waveform,
)


@pytest.mark.parametrize("samples", [[], [[1]], [float("nan")], [float("inf")], [1j], ["1"]])
def test_invalid_waveforms(samples):
    with pytest.raises(ValueError):
        validate_waveform(np.array(samples))
