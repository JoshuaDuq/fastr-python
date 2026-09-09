"""Estimate power at exact acquisition harmonics on a sampled time grid."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from scipy.signal import ZoomFFT, get_window


def _validate_harmonic_data(
    data: np.ndarray,
    sampling_rate: float,
    fundamental_hz: float,
) -> None:
    """Validate signal and frequency assumptions shared by harmonic estimates."""
    if (
        data.ndim != 2
        or data.size == 0
        or not np.issubdtype(data.dtype, np.number)
        or np.iscomplexobj(data)
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("harmonic data must be a nonempty finite real channel matrix")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in (sampling_rate, fundamental_hz)
    ):
        raise ValueError(
            "harmonic frequencies and sampling rate must be finite and positive"
        )


def _make_harmonic_transform(
    segment_samples: int,
    sampling_rate: float,
    fundamental_hz: float,
    harmonic_count: int,
) -> ZoomFFT:
    """Build a transform covering strictly positive harmonics below Nyquist."""
    if (
        isinstance(harmonic_count, bool)
        or not isinstance(harmonic_count, Integral)
        or harmonic_count < 1
        or harmonic_count * fundamental_hz >= sampling_rate / 2
    ):
        raise ValueError("harmonic count must select positive harmonics below Nyquist")
    return ZoomFFT(
        segment_samples,
        (fundamental_hz, (harmonic_count + 1) * fundamental_hz),
        m=harmonic_count,
        fs=sampling_rate,
    )


def mean_harmonic_psd(
    data: np.ndarray,
    *,
    sampling_rate: float,
    fundamental_hz: float,
    harmonic_count: int,
    segment_samples: int,
) -> np.ndarray:
    """Average Hann periodograms at exact harmonics with Welch normalization.

    Segment lengths live on the sample grid; harmonic frequencies retain the
    declared acquisition period. ZoomFFT evaluates those frequencies directly,
    without rounding the period or interpolating PSD bins.
    """
    _validate_harmonic_data(data, sampling_rate, fundamental_hz)
    if (
        isinstance(segment_samples, bool)
        or not isinstance(segment_samples, Integral)
        or not 2 <= segment_samples <= data.shape[1]
    ):
        raise ValueError("spectral segment must contain at least two available samples")
    window = get_window("hann", segment_samples)
    transform = _make_harmonic_transform(
        segment_samples, sampling_rate, fundamental_hz, harmonic_count
    )
    power = np.zeros((data.shape[0], harmonic_count))
    step = segment_samples - segment_samples // 2
    starts = range(0, data.shape[1] - segment_samples + 1, step)
    for start in starts:
        segment = data[:, start : start + segment_samples]
        centered = segment - segment.mean(axis=1, keepdims=True)
        power += np.abs(transform(centered * window, axis=1)) ** 2
    return power * (2.0 / (len(starts) * sampling_rate * np.sum(window**2)))


def block_coherent_harmonic_rms(
    data: np.ndarray,
    *,
    sampling_rate: float,
    fundamental_hz: float,
    harmonic_orders: np.ndarray,
    block_seconds: float,
) -> np.ndarray:
    """Measure coherent harmonic RMS in each complete non-overlapping block.

    Rectangular windows over complete volumes retain Fourier-series amplitudes
    without spreading a harmonic into its neighbors, even for one-volume blocks.
    This is acquisition-locked signal power, including possible neural signal;
    it is not an estimate of artifact alone or a background-subtracted metric.
    """
    _validate_harmonic_data(data, sampling_rate, fundamental_hz)
    if not math.isfinite(block_seconds) or block_seconds <= 0:
        raise ValueError("harmonic block duration must be finite and positive")
    block_samples = round(block_seconds * sampling_rate)
    if block_samples < 2:
        raise ValueError("harmonic blocks must contain at least two samples")
    if (
        harmonic_orders.ndim != 1
        or not np.issubdtype(harmonic_orders.dtype, np.integer)
        or np.any(harmonic_orders < 1)
        or np.any(np.diff(harmonic_orders) <= 0)
        or np.any(harmonic_orders * fundamental_hz >= sampling_rate / 2)
    ):
        raise ValueError(
            "harmonic orders must be increasing positive integers below Nyquist"
        )
    block_count = data.shape[1] // block_samples
    amplitudes = np.zeros((data.shape[0], block_count))
    if harmonic_orders.size == 0 or block_count == 0:
        return amplitudes
    transform = _make_harmonic_transform(
        block_samples, sampling_rate, fundamental_hz, int(harmonic_orders[-1])
    )
    for block in range(block_count):
        start = block * block_samples
        segment = data[:, start : start + block_samples]
        centered = segment - segment.mean(axis=1, keepdims=True)
        power = np.abs(transform(centered, axis=1)) ** 2
        amplitudes[:, block] = (
            np.sqrt(2 * power[:, harmonic_orders - 1].sum(axis=1)) / block_samples
        )
    return amplitudes
