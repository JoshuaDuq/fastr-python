"""Measure residual scanner-gradient artifact in the corrected output.

The measurement is reported in microvolts rather than as a ratio against local
background. A ratio is unusable as a threshold: on this cohort it reported a
1.75 uV residual as 310x background in a quiet block and a 0.05 uV residual as
13.9x, which says more about the background than about the artifact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

_DEFAULT_MAINS_HZ = 60.0
_DEFAULT_EXCLUSION_HZ = 1.0
_DEFAULT_BLOCK_SECONDS = 30.0
# Widths in units of the Welch frequency resolution. A Blackman-Harris main lobe
# spans four bins either side of the peak, so a narrower band would miss leaked
# power and under-report the residual.
_PEAK_HALF_WIDTH_BINS = 4.0
_BACKGROUND_INNER_BINS = 6.0
_BACKGROUND_OUTER_BINS = 20.0
_WELCH_SECONDS = 8.0


class ResidualQcError(ValueError):
    """Raised when residual artifact cannot be measured from the given inputs."""


def slice_harmonics(
    *,
    groups_per_volume: int,
    repetition_time_seconds: float,
    nyquist_hz: float,
    mains_hz: float = _DEFAULT_MAINS_HZ,
    exclusion_hz: float = _DEFAULT_EXCLUSION_HZ,
) -> tuple[float, ...]:
    """Derive the acquisition-locked harmonics worth measuring.

    Harmonics that collide with the mains frequency are excluded, because power
    there cannot be attributed to the gradient artifact. This is not hypothetical
    for a 0.9 s TR with 18 acquisition slots: the slice rate is 20 Hz and its
    third harmonic is 60 Hz exactly.
    """
    if isinstance(groups_per_volume, bool) or not isinstance(groups_per_volume, int):
        raise ResidualQcError("groups per volume must be a positive integer")
    if groups_per_volume < 1:
        raise ResidualQcError("groups per volume must be a positive integer")
    for name, value in (
        ("repetition time", repetition_time_seconds),
        ("nyquist frequency", nyquist_hz),
        ("mains frequency", mains_hz),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ResidualQcError(f"{name} must be a finite positive number")
    if not math.isfinite(exclusion_hz) or exclusion_hz < 0.0:
        raise ResidualQcError("exclusion width must be a finite non-negative number")

    slice_rate = groups_per_volume / repetition_time_seconds
    harmonics = []
    for order in range(1, int(nyquist_hz // slice_rate) + 1):
        frequency = slice_rate * order
        if frequency >= nyquist_hz:
            break
        nearest_mains = round(frequency / mains_hz) * mains_hz
        if nearest_mains > 0.0 and abs(frequency - nearest_mains) <= exclusion_hz:
            continue
        harmonics.append(float(frequency))
    return tuple(harmonics)


def block_residual_uv(
    data: npt.ArrayLike,
    *,
    sampling_rate: float,
    harmonics: Sequence[float],
    block_seconds: float = _DEFAULT_BLOCK_SECONDS,
) -> np.ndarray:
    """Measure per-channel, per-block residual artifact amplitude in microvolts.

    For each harmonic the power in a narrow band around it is compared with the
    median of two flanking bands, and the excess is integrated. The returned
    value is the root of the summed excess across harmonics, so it is an
    amplitude directly comparable between channels, blocks and runs.
    """
    recording = np.asarray(data, dtype=np.float64)
    if recording.ndim != 2:
        raise ResidualQcError("data must be two-dimensional (channels, samples)")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ResidualQcError("sampling rate must be a finite positive number")
    if not math.isfinite(block_seconds) or block_seconds <= 0.0:
        raise ResidualQcError("block length must be a finite positive number")

    block_samples = round(block_seconds * sampling_rate)
    block_count = recording.shape[1] // block_samples if block_samples else 0
    if block_count < 1:
        return np.empty((recording.shape[0], 0), dtype=np.float64)

    segment_samples = min(round(_WELCH_SECONDS * sampling_rate), block_samples)
    residuals = np.zeros((recording.shape[0], block_count), dtype=np.float64)
    for block in range(block_count):
        start = block * block_samples
        frequencies, power = welch(
            recording[:, start : start + block_samples],
            sampling_rate,
            nperseg=segment_samples,
            window="blackmanharris",
        )
        residuals[:, block] = _excess_amplitude(
            frequencies,
            power,
            harmonics,
            resolution_hz=sampling_rate / segment_samples,
        )
    return residuals


def _excess_amplitude(
    frequencies: np.ndarray,
    power: np.ndarray,
    harmonics: Sequence[float],
    *,
    resolution_hz: float,
) -> np.ndarray:
    peak_half_width = _PEAK_HALF_WIDTH_BINS * resolution_hz
    inner = _BACKGROUND_INNER_BINS * resolution_hz
    outer = _BACKGROUND_OUTER_BINS * resolution_hz
    total = np.zeros(power.shape[0], dtype=np.float64)
    for frequency in harmonics:
        offset = np.abs(frequencies - frequency)
        peak = offset <= peak_half_width
        background = (offset > inner) & (offset <= outer)
        if not peak.any() or not background.any():
            continue
        floor = np.median(power[:, background], axis=1, keepdims=True)
        excess = np.maximum(power[:, peak] - floor, 0.0)
        total += np.trapezoid(excess, frequencies[peak], axis=1)
    return np.sqrt(total)
