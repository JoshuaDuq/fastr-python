"""Measure residual scanner-gradient artifact in the corrected output.

The measurement is reported in microvolts rather than as a ratio against local
background. A ratio is unusable as a threshold: on this cohort it reported a
1.75 uV residual as 310x background in a quiet block and a 0.05 uV residual as
13.9x, which says more about the background than about the artifact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
_VOLUME_SPECTRUM_SECONDS = 90.0
_LOCAL_PEAK_HALF_WIDTH_HZ = 0.15
# A block is flagged relative to the recording's own distribution rather than
# against an absolute microvolt cut, which cannot be set without knowing how
# much artifact was there before correction.
_DEFAULT_MAD_MULTIPLIER = 6.0
_DEFAULT_MINIMUM_CHANNELS = 4
_DEFAULT_FLOOR_UV = 1.0
_MINIMUM_CALIBRATION_BLOCKS = 3
_MAD_TO_SIGMA = 1.4826


class residual_qc_defaults:
    """Defaults shared with the pipeline and configuration layers."""

    MAD_MULTIPLIER = _DEFAULT_MAD_MULTIPLIER
    MINIMUM_CHANNELS = _DEFAULT_MINIMUM_CHANNELS
    FLOOR_UV = _DEFAULT_FLOOR_UV


class ResidualQcError(ValueError):
    """Raised when residual artifact cannot be measured from the given inputs."""


@dataclass(frozen=True, slots=True)
class VolumeHarmonicSpectrum:
    """Exact-bin and nearby peak power for one volume harmonic."""

    order: int
    frequency_hz: float
    exact_power_db: float
    local_peak_frequency_hz: float
    local_peak_power_db: float
    mains_collision: bool


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


def flag_blocks(
    residuals: npt.ArrayLike,
    *,
    mad_multiplier: float = _DEFAULT_MAD_MULTIPLIER,
    minimum_channels: int = _DEFAULT_MINIMUM_CHANNELS,
    floor_uv: float = _DEFAULT_FLOOR_UV,
) -> np.ndarray:
    """Mark blocks whose residual stands out from the recording's own baseline.

    Two conditions must hold together. A channel-block must exceed its own
    channel's median by ``mad_multiplier`` robust sigma, which makes the test
    self-calibrating: a recording that is uniformly noisy has a high baseline
    and is not flagged for it. And it must clear ``floor_uv``, so a very clean
    recording does not flag its own measurement noise.

    A block is flagged only when at least ``minimum_channels`` channels meet
    both conditions. Residual gradient artifact is an induced EMF with coherent
    spatial structure, so a genuine failure appears on many channels at once;
    requiring that keeps one bad electrode from condemning every channel.
    """
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2:
        raise ResidualQcError("residuals must be two-dimensional (channels, blocks)")
    if not math.isfinite(mad_multiplier) or mad_multiplier < 0.0:
        raise ResidualQcError("MAD multiplier must be a finite non-negative number")
    if not math.isfinite(floor_uv) or floor_uv < 0.0:
        raise ResidualQcError("floor must be a finite non-negative number")
    if isinstance(minimum_channels, bool) or not isinstance(minimum_channels, int):
        raise ResidualQcError("minimum channel count must be a positive integer")
    if minimum_channels < 1:
        raise ResidualQcError("minimum channel count must be a positive integer")

    block_count = values.shape[1]
    if block_count < _MINIMUM_CALIBRATION_BLOCKS:
        # Too few blocks to establish what this recording's baseline is. The
        # per-block measurements still reach the sidecar unflagged.
        return np.zeros(block_count, dtype=bool)

    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    elevated = (values - median > mad_multiplier * mad * _MAD_TO_SIGMA) & (
        values > floor_uv
    )
    return elevated.sum(axis=0) >= minimum_channels


def volume_harmonic_spectrum(
    data: npt.ArrayLike,
    *,
    sampling_rate: float,
    repetition_time_seconds: float,
    maximum_frequency_hz: float,
    mains_frequency_hz: float = _DEFAULT_MAINS_HZ,
    mains_exclusion_hz: float = _DEFAULT_EXCLUSION_HZ,
) -> tuple[VolumeHarmonicSpectrum, ...]:
    """Measure exact volume harmonics separately from adjacent sidebands."""
    recording = np.asarray(data, dtype=np.float64)
    if recording.ndim != 2 or recording.shape[1] == 0:
        raise ResidualQcError("data must be two-dimensional (channels, samples)")
    for name, value in (
        ("sampling rate", sampling_rate),
        ("repetition time", repetition_time_seconds),
        ("maximum frequency", maximum_frequency_hz),
        ("mains frequency", mains_frequency_hz),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ResidualQcError(f"{name} must be a finite positive number")
    if not math.isfinite(mains_exclusion_hz) or mains_exclusion_hz < 0.0:
        raise ResidualQcError(
            "mains exclusion width must be a finite non-negative number"
        )
    nyquist = 0.5 * sampling_rate
    if maximum_frequency_hz >= nyquist:
        raise ResidualQcError("maximum frequency must stay below Nyquist")

    samples_per_volume_float = repetition_time_seconds * sampling_rate
    samples_per_volume = round(samples_per_volume_float)
    if not math.isclose(
        samples_per_volume_float,
        samples_per_volume,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ResidualQcError(
            "repetition time must span an integer number of samples"
        )
    available_volumes = recording.shape[1] // samples_per_volume
    target_volumes = max(
        2,
        round(_VOLUME_SPECTRUM_SECONDS / repetition_time_seconds),
    )
    segment_volumes = min(available_volumes, target_volumes)
    if segment_volumes < 2:
        raise ResidualQcError(
            "recording must contain at least two complete volumes"
        )
    segment_samples = segment_volumes * samples_per_volume
    frequencies, power = welch(
        recording,
        sampling_rate,
        nperseg=segment_samples,
        noverlap=segment_samples // 2,
        window="hann",
        axis=1,
    )
    median_power = np.median(power, axis=0)
    fundamental_hz = 1.0 / repetition_time_seconds
    maximum_order = int(maximum_frequency_hz // fundamental_hz)
    profile: list[VolumeHarmonicSpectrum] = []
    for order in range(1, maximum_order + 1):
        frequency_hz = order * fundamental_hz
        exact_index = int(np.argmin(np.abs(frequencies - frequency_hz)))
        local_indices = np.flatnonzero(
            np.abs(frequencies - frequency_hz) <= _LOCAL_PEAK_HALF_WIDTH_HZ
        )
        peak_index = int(
            local_indices[np.argmax(median_power[local_indices])]
        )
        nearest_mains = round(frequency_hz / mains_frequency_hz) * mains_frequency_hz
        profile.append(
            VolumeHarmonicSpectrum(
                order=order,
                frequency_hz=float(frequency_hz),
                exact_power_db=_power_db(median_power[exact_index]),
                local_peak_frequency_hz=float(frequencies[peak_index]),
                local_peak_power_db=_power_db(median_power[peak_index]),
                mains_collision=(
                    nearest_mains > 0.0
                    and abs(frequency_hz - nearest_mains) <= mains_exclusion_hz
                ),
            )
        )
    return tuple(profile)


def _power_db(power: float) -> float:
    return float(10.0 * np.log10(max(power, np.finfo(np.float64).tiny)))


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
