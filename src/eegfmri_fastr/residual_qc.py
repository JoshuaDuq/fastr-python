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
# A local retry has to buy something measurable, not merely something. It must
# fix at least one failed block and pull the worst residual down by 15%, which
# is well outside the block-to-block spread the measurement itself carries.
_LOCAL_RETRY_MAXIMUM_RATIO = 0.85
# And a channel is only recommended as bad when it failed repeatedly: two
# blocks at minimum, and a tenth of a long recording, so a single bad moment
# stays a bad moment.
_BAD_CHANNEL_MINIMUM_BLOCKS = 2
_BAD_CHANNEL_MINIMUM_FRACTION = 0.10


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
    values = _validate_flag_inputs(
        residuals,
        mad_multiplier=mad_multiplier,
        floor_uv=floor_uv,
    )
    if isinstance(minimum_channels, bool) or not isinstance(minimum_channels, int):
        raise ResidualQcError("minimum channel count must be a positive integer")
    if minimum_channels < 1:
        raise ResidualQcError("minimum channel count must be a positive integer")

    elevated = _elevated_channel_blocks(
        values,
        mad_multiplier=mad_multiplier,
        floor_uv=floor_uv,
    )
    return elevated.sum(axis=0) >= minimum_channels


def flag_channel_blocks(
    residuals: npt.ArrayLike,
    *,
    mad_multiplier: float = _DEFAULT_MAD_MULTIPLIER,
    floor_uv: float = _DEFAULT_FLOOR_UV,
) -> np.ndarray:
    """Mark isolated channel-block residuals relative to each channel baseline."""
    values = _validate_flag_inputs(
        residuals,
        mad_multiplier=mad_multiplier,
        floor_uv=floor_uv,
    )
    return _elevated_channel_blocks(
        values,
        mad_multiplier=mad_multiplier,
        floor_uv=floor_uv,
    )


@dataclass(frozen=True, slots=True)
class SpatialChannelFlags:
    """Per-block failure thresholds, and the channel-blocks that cleared them."""

    thresholds_uv: np.ndarray
    flags: np.ndarray


def flag_spatial_channel_blocks(
    residuals: npt.ArrayLike,
    *,
    eeg_channels: Sequence[int],
    absolute_floor_uv: float,
    mad_multiplier: float,
) -> SpatialChannelFlags:
    """Mark channel-blocks that stand out from the other channels beside them.

    ``flag_channel_blocks`` compares a channel against its own past, which
    cannot see an electrode that was bad for the whole run. This compares it
    against the other EEG channels measured in the same block instead, so a
    persistent failure is still visible. The two conditions of ``flag_blocks``
    are kept: the excess must be robust against the spatial distribution, and
    it must clear an absolute floor, because a spatial comparison on its own
    would happily nominate the loudest channel of a clean recording.

    Non-EEG rows are excluded from both the statistic and the result: their
    physiological signal is not evidence about anyone's gradient residual.
    """
    values = _validated_residual_matrix(residuals)
    eeg = _validated_channel_indices(eeg_channels, values.shape[0])
    if not eeg:
        # The threshold is a statistic over the EEG channels of a block, so an
        # empty ensemble has no threshold to report rather than an infinite one.
        raise ResidualQcError("at least one EEG channel is required")
    if not math.isfinite(mad_multiplier) or mad_multiplier < 0.0:
        raise ResidualQcError("MAD multiplier must be a finite non-negative number")
    if not math.isfinite(absolute_floor_uv) or absolute_floor_uv <= 0.0:
        raise ResidualQcError("absolute floor must be a finite positive number")
    if values.shape[1] < _MINIMUM_CALIBRATION_BLOCKS:
        return SpatialChannelFlags(
            thresholds_uv=np.full(values.shape[1], absolute_floor_uv),
            flags=np.zeros(values.shape, dtype=bool),
        )
    eeg_values = values[eeg]
    median = np.median(eeg_values, axis=0)
    mad = np.median(np.abs(eeg_values - median), axis=0)
    thresholds = np.maximum(
        absolute_floor_uv,
        median + mad_multiplier * _MAD_TO_SIGMA * mad,
    )
    flags = np.zeros(values.shape, dtype=bool)
    flags[eeg] = eeg_values > thresholds
    return SpatialChannelFlags(thresholds_uv=thresholds, flags=flags)


@dataclass(frozen=True, slots=True)
class LocalRetryEvaluation:
    """Whether one channel's local retry beat its wide correction, and by how much."""

    accepted: bool
    reason: str
    wide_failed_blocks: np.ndarray
    local_failed_blocks: np.ndarray
    wide_maximum_uv: float
    local_maximum_uv: float


def evaluate_local_retry(
    wide_residuals_uv: npt.ArrayLike,
    local_residuals_uv: npt.ArrayLike,
    thresholds_uv: npt.ArrayLike,
) -> LocalRetryEvaluation:
    """Decide whether to install a local retry of one channel over its wide result.

    Both conditions are required, and both are compared against thresholds
    frozen from the wide pass. Counting failed blocks alone would install a
    retry that merely moved a residual from just above the threshold to just
    below it; comparing maxima alone would install one that halved a single
    spike while leaving every failed block failed.
    """
    wide, local, thresholds = _validated_retry_vectors(
        wide_residuals_uv,
        local_residuals_uv,
        thresholds_uv,
    )
    wide_failed = np.flatnonzero(wide > thresholds)
    local_failed = np.flatnonzero(local > thresholds)
    wide_maximum = float(wide.max(initial=0.0))
    local_maximum = float(local.max(initial=0.0))
    fewer_failures = local_failed.size < wide_failed.size
    lower_maximum = local_maximum <= _LOCAL_RETRY_MAXIMUM_RATIO * wide_maximum
    accepted = bool(fewer_failures and lower_maximum)
    reason = (
        "fewer_failed_blocks_and_lower_maximum"
        if accepted
        else "failed_block_count_not_reduced"
        if not fewer_failures
        else "maximum_residual_not_reduced_by_fifteen_percent"
    )
    return LocalRetryEvaluation(
        accepted=accepted,
        reason=reason,
        wide_failed_blocks=wide_failed,
        local_failed_blocks=local_failed,
        wide_maximum_uv=wide_maximum,
        local_maximum_uv=local_maximum,
    )


def recommend_persistent_bad_channels(flags: npt.ArrayLike) -> np.ndarray:
    """Mark channels whose residual failure survived correction across the run.

    This is a recommendation and nothing more. Nothing here interpolates or
    drops a channel: a scanner-residual failure is one bad-electrode condition
    among several, and which of them justify replacing data is a decision for
    whoever knows the study.
    """
    values = _validated_boolean_matrix(flags)
    if values.shape[1] < _MINIMUM_CALIBRATION_BLOCKS:
        return np.zeros(values.shape[0], dtype=bool)
    required = max(
        _BAD_CHANNEL_MINIMUM_BLOCKS,
        math.ceil(_BAD_CHANNEL_MINIMUM_FRACTION * values.shape[1]),
    )
    return values.sum(axis=1) >= required


def _validated_retry_vectors(
    wide_residuals_uv: npt.ArrayLike,
    local_residuals_uv: npt.ArrayLike,
    thresholds_uv: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (wide_residuals_uv, local_residuals_uv, thresholds_uv)
    )
    if any(values.ndim != 1 for values in vectors):
        raise ResidualQcError("retry residuals must be one-dimensional (blocks,)")
    if len({values.shape[0] for values in vectors}) != 1:
        raise ResidualQcError(
            "retry residuals and thresholds must have the same length"
        )
    if any(not np.all(np.isfinite(values)) for values in vectors):
        raise ResidualQcError("retry residuals and thresholds must be finite")
    return vectors


def _validated_boolean_matrix(flags: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(flags)
    if values.ndim != 2:
        raise ResidualQcError("flags must be two-dimensional (channels, blocks)")
    if values.dtype != bool:
        raise ResidualQcError("flags must be boolean")
    return values


def _validated_residual_matrix(residuals: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2:
        raise ResidualQcError("residuals must be two-dimensional (channels, blocks)")
    if not np.all(np.isfinite(values)):
        raise ResidualQcError("residuals must be finite")
    return values


def _validated_channel_indices(
    channels: Sequence[int],
    channel_count: int,
) -> list[int]:
    indices = list(channels)
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise ResidualQcError("EEG channel indices must be integers")
    if len(set(indices)) != len(indices):
        raise ResidualQcError("EEG channel indices must be unique")
    if any(not 0 <= int(index) < channel_count for index in indices):
        raise ResidualQcError("EEG channel indices must be within the recording")
    return [int(index) for index in indices]


def _validate_flag_inputs(
    residuals: npt.ArrayLike,
    *,
    mad_multiplier: float,
    floor_uv: float,
) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2:
        raise ResidualQcError("residuals must be two-dimensional (channels, blocks)")
    if not math.isfinite(mad_multiplier) or mad_multiplier < 0.0:
        raise ResidualQcError("MAD multiplier must be a finite non-negative number")
    if not math.isfinite(floor_uv) or floor_uv < 0.0:
        raise ResidualQcError("floor must be a finite non-negative number")
    return values


def _elevated_channel_blocks(
    values: np.ndarray,
    *,
    mad_multiplier: float,
    floor_uv: float,
) -> np.ndarray:
    if values.shape[1] < _MINIMUM_CALIBRATION_BLOCKS:
        return np.zeros(values.shape, dtype=bool)
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    return (values - median > mad_multiplier * mad * _MAD_TO_SIGMA) & (
        values > floor_uv
    )


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
