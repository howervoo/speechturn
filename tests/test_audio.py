"""Audio shape, anti-aliasing, channel and crop behavior."""

import numpy as np
import pytest
import soundfile as sf

from speechturn.audio import (
    read_audio,
    resample_audio,
    validate_waveform,
)


@pytest.mark.parametrize("samples", [[], [[1]], [float("nan")], [float("inf")], [1j], ["1"]])
def test_invalid_waveforms(samples):
    with pytest.raises(ValueError):
        validate_waveform(np.array(samples))


def test_resampling_length_and_identity():
    values = np.arange(81, dtype=np.float32)
    result = resample_audio(values, 8000, 16000)
    assert result.shape == (162,)
    copy = resample_audio(values, 8000, 8000)
    assert np.array_equal(values, copy)
    assert not np.shares_memory(values, copy)


def test_downsampling_suppresses_out_of_band_tones():
    time = np.arange(16000) / 16000
    signal = np.sin(2 * np.pi * 6000 * time)
    result = resample_audio(signal, 16000, 8000)
    assert np.sqrt(np.mean(result[100:-100] ** 2)) < 0.01


def test_stereo_mix_and_target_rate(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((np.ones(80) * 0.5, np.ones(80) * -0.5)), 8000, subtype="FLOAT")
    result = read_audio(path, 16000)
    assert result.shape == (160,)
    assert np.all(result == 0)
