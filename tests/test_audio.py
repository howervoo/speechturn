"""Audio shape, anti-aliasing, channel and crop behavior."""

import numpy as np
import pytest
import soundfile as sf

from speechturn.audio import (
    crop_audio,
    log_mel,
    mel_filterbank,
    read_audio,
    resample_audio,
    validate_waveform,
)
from speechturn.config import SpeechConfig


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


def test_silence_features_are_finite():
    features = log_mel(np.zeros(10))
    assert features.shape == (1, 40)
    assert np.isfinite(features).all()
    assert np.allclose(features, np.log(1e-10))


def test_mel_frequency_locality():
    settings = SpeechConfig()
    bank = mel_filterbank(settings)
    assert bank.shape == (40, 201)
    assert np.all(bank >= 0)
    assert np.all(np.diff(bank.argmax(axis=1)) >= 0)


def test_crop_does_not_modify_original():
    values = np.arange(100, dtype=np.float32)
    output = crop_audio(values, 10, start=2, duration=3)
    assert np.array_equal(output, np.arange(20, 50))
    output[:] = 0
    assert values[20] == 20
