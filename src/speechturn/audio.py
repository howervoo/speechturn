"""Audio loading, anti-aliased resampling and HTK log-mel features."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from speechturn.config import SpeechConfig, positive_integer


def validate_waveform(samples: np.ndarray) -> np.ndarray:
    """Return a finite, non-empty mono float32 waveform."""
    values = np.asarray(samples)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("waveform must contain real numeric samples")
    values = values.astype(np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be a non-empty finite mono vector")
    return values


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Use rational polyphase filtering to suppress downsampling aliases."""
    positive_integer(source_rate, "source_rate")
    positive_integer(target_rate, "target_rate")
    values = validate_waveform(samples)
    if source_rate == target_rate:
        return values.copy()
    divisor = gcd(source_rate, target_rate)
    result = resample_poly(values, target_rate // divisor, source_rate // divisor)
    return np.asarray(result, dtype=np.float32)


def read_audio(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Read supported audio containers, mix channels and resample explicitly."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[1] < 1:
        raise ValueError("audio must contain at least one channel")
    return resample_audio(samples.mean(axis=1), int(rate), target_rate)


def mel_filterbank(config: SpeechConfig) -> np.ndarray:
    """Construct continuous triangular filters on the HTK mel scale."""
    maximum = 2595.0 * np.log10(1.0 + config.sample_rate / 1400.0)
    points = 700.0 * (10.0 ** (np.linspace(0.0, maximum, config.n_mels + 2) / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    rising = (frequencies[None, :] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - frequencies[None, :]) / (points[2:] - points[1:-1])[:, None]
    return np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)


def log_mel(samples: np.ndarray, config: SpeechConfig | None = None) -> np.ndarray:
    """Return time-major log-mel features; short utterances receive zero padding."""
    settings = config or SpeechConfig()
    values = validate_waveform(samples)
    if values.size < settings.win_length:
        values = np.pad(values, (0, settings.win_length - values.size))
    frames = np.lib.stride_tricks.sliding_window_view(values, settings.win_length)[
        :: settings.hop_length
    ]
    window = np.hanning(settings.win_length)
    spectrum = np.fft.rfft(frames * window, n=settings.n_fft, axis=-1)
    power = np.abs(spectrum) ** 2
    features = power @ mel_filterbank(settings).T
    return np.log(np.maximum(features, 1e-10)).astype(np.float32)


def crop_audio(
    samples: np.ndarray, sample_rate: int, start: float = 0.0, duration: float | None = None
) -> np.ndarray:
    """Crop in seconds and reject empty or out-of-range selections."""
    from speechturn.config import finite_number

    values = validate_waveform(samples)
    positive_integer(sample_rate, "sample_rate")
    offset = finite_number(start, "start")
    first = int(round(offset * sample_rate))
    if duration is None:
        last = len(values)
    else:
        length = finite_number(duration, "duration")
        if length == 0:
            raise ValueError("duration must be positive")
        last = min(len(values), first + int(round(length * sample_rate)))
    if first >= len(values) or last <= first:
        raise ValueError("audio crop contains no samples")
    return values[first:last].copy()


__all__ = [
    "crop_audio",
    "log_mel",
    "mel_filterbank",
    "read_audio",
    "resample_audio",
    "validate_waveform",
]
