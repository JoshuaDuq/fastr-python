"""Bounded cardiac-artifact correction for FASTR recordings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import mne
import numpy as np
import numpy.typing as npt


class BcgInputError(ValueError):
    """Raised when BCG correction inputs are invalid or insufficient."""


@dataclass(frozen=True, slots=True)
class BcgCorrectionConfig:
    """Validated settings for one bounded BCG correction."""

    method: str
    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in {
            "aas",
            "pca_obs",
        }:
            raise BcgInputError("method must be 'aas' or 'pca_obs'")
        if (
            not isinstance(self.window_seconds, tuple)
            or len(self.window_seconds) != 2
            or not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.window_seconds
            )
            or self.window_seconds[0] >= self.window_seconds[1]
        ):
            raise BcgInputError(
                "window_seconds must be a finite increasing pair"
            )
        if (
            isinstance(self.ecg_to_bcg_delay_seconds, bool)
            or not isinstance(self.ecg_to_bcg_delay_seconds, Real)
            or not math.isfinite(float(self.ecg_to_bcg_delay_seconds))
        ):
            raise BcgInputError(
                "ecg_to_bcg_delay_seconds must be finite"
            )
        if (
            isinstance(self.aas_neighbor_count, bool)
            or not isinstance(self.aas_neighbor_count, Integral)
            or self.aas_neighbor_count < 1
        ):
            raise BcgInputError(
                "aas_neighbor_count must be a positive integer"
            )
        if (
            isinstance(self.pca_obs_components, bool)
            or not isinstance(self.pca_obs_components, Integral)
            or self.pca_obs_components < 1
        ):
            raise BcgInputError(
                "pca_obs_components must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class BcgCorrectionResult:
    """Corrected data and the exact samples changed by the splice."""

    data_volts: npt.NDArray[np.float64]
    corrected_samples: npt.NDArray[np.int64]
    method: str


def correct_bcg(
    data_volts: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    channel_names: Sequence[str],
    eeg_picks: npt.ArrayLike,
    ecg_channel_index: int,
    config: BcgCorrectionConfig,
) -> BcgCorrectionResult:
    """Apply bounded AAS or MNE PCA-OBS around explicit BCG artifact anchors."""
    data = _validate_data(data_volts)
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    peaks = _validate_peak_samples(peak_samples, data.shape[1])
    names = _validate_channel_names(channel_names, data.shape[0])
    eeg_indices = _validate_eeg_picks(eeg_picks, data.shape[0])
    ecg_index = _validate_ecg_index(ecg_channel_index, data.shape[0])
    if ecg_index in eeg_indices:
        raise BcgInputError("ecg_channel_index cannot be corrected as EEG")

    anchor_samples = _artifact_anchor_samples(
        peaks,
        sampling_rate,
        config.ecg_to_bcg_delay_seconds,
        data.shape[1],
    )
    window_start, window_stop = _window_samples(
        config.window_seconds,
        sampling_rate,
    )
    window_bounds = _validate_complete_windows(
        anchor_samples,
        window_start,
        window_stop,
        data.shape[1],
    )
    corrected_samples = _window_union(
        window_bounds,
        sample_count=data.shape[1],
    )

    if config.method == "aas":
        corrected = _correct_aas(
            data,
            eeg_indices,
            window_bounds,
            config.aas_neighbor_count,
        )
    else:
        corrected = _correct_pca_obs(
            data,
            names,
            eeg_indices,
            anchor_samples,
            corrected_samples,
            sampling_rate,
            config.pca_obs_components,
        )
    corrected[ecg_index] = data[ecg_index]
    return BcgCorrectionResult(
        data_volts=corrected,
        corrected_samples=corrected_samples,
        method=config.method,
    )


def _validate_data(data_volts: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(data_volts)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise BcgInputError("data_volts must have shape (channels, samples)")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.number,
    ):
        raise BcgInputError("data_volts must contain finite numeric values")
    values = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise BcgInputError("data_volts must contain finite numeric values")
    return values


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    if (
        isinstance(sampling_rate_hz, bool)
        or not isinstance(sampling_rate_hz, Real)
        or not math.isfinite(float(sampling_rate_hz))
        or sampling_rate_hz <= 0.0
    ):
        raise BcgInputError("sampling_rate_hz must be finite and positive")
    return float(sampling_rate_hz)


def _validate_peak_samples(
    peak_samples: npt.ArrayLike,
    sample_count: int,
) -> np.ndarray:
    values = np.asarray(peak_samples)
    if values.ndim != 1 or values.size < 2:
        raise BcgInputError("peak_samples must contain at least two events")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise BcgInputError("peak_samples must contain integer samples")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= sample_count):
        raise BcgInputError("peak_samples contain positions outside the recording")
    if np.any(np.diff(values) <= 0):
        raise BcgInputError("peak_samples must be strictly increasing")
    return values


def _validate_channel_names(
    channel_names: Sequence[str],
    channel_count: int,
) -> tuple[str, ...]:
    names = tuple(channel_names)
    if len(names) != channel_count or not all(
        isinstance(name, str) and name for name in names
    ):
        raise BcgInputError(
            "channel_names must contain one nonempty name per channel"
        )
    if len(set(names)) != len(names):
        raise BcgInputError("channel_names must be unique")
    return names


def _validate_eeg_picks(eeg_picks: npt.ArrayLike, channel_count: int) -> np.ndarray:
    values = np.asarray(eeg_picks)
    if values.ndim != 1 or values.size == 0:
        raise BcgInputError("eeg_picks must contain at least one channel")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise BcgInputError("eeg_picks must contain integer channel indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= channel_count):
        raise BcgInputError("eeg_picks contain an invalid channel index")
    if np.unique(values).size != values.size:
        raise BcgInputError("eeg_picks cannot contain duplicates")
    return values


def _validate_ecg_index(ecg_channel_index: int, channel_count: int) -> int:
    if (
        isinstance(ecg_channel_index, bool)
        or not isinstance(ecg_channel_index, Integral)
        or ecg_channel_index < 0
        or ecg_channel_index >= channel_count
    ):
        raise BcgInputError("ecg_channel_index is outside the recording")
    return int(ecg_channel_index)


def _artifact_anchor_samples(
    peak_samples: np.ndarray,
    sampling_rate: float,
    delay_seconds: float,
    sample_count: int,
) -> np.ndarray:
    delay_samples = round(delay_seconds * sampling_rate)
    anchors = peak_samples + delay_samples
    if np.any(anchors < 0) or np.any(anchors >= sample_count):
        raise BcgInputError(
            "ECG-to-BCG delay places an artifact anchor outside the recording"
        )
    if np.any(np.diff(anchors) <= 0):
        raise BcgInputError("artifact anchors must be strictly increasing")
    return anchors.astype(np.int64, copy=False)


def _window_samples(
    window_seconds: tuple[float, float],
    sampling_rate: float,
) -> tuple[int, int]:
    return (
        round(window_seconds[0] * sampling_rate),
        round(window_seconds[1] * sampling_rate),
    )


def _validate_complete_windows(
    anchors: np.ndarray,
    window_start: int,
    window_stop: int,
    sample_count: int,
) -> tuple[tuple[int, int], ...]:
    bounds = tuple(
        (int(anchor + window_start), int(anchor + window_stop))
        for anchor in anchors
    )
    if any(
        start < 0 or stop > sample_count or start >= stop
        for start, stop in bounds
    ):
        raise BcgInputError(
            "the configured BCG window is incomplete at one or more anchors"
        )
    return bounds


def _window_union(
    bounds: tuple[tuple[int, int], ...],
    *,
    sample_count: int,
) -> np.ndarray:
    coverage = np.zeros(sample_count, dtype=bool)
    for start, stop in bounds:
        coverage[start:stop] = True
    return np.flatnonzero(coverage).astype(np.int64, copy=False)


def _correct_aas(
    data: np.ndarray,
    eeg_indices: np.ndarray,
    window_bounds: tuple[tuple[int, int], ...],
    neighbor_count: int,
) -> np.ndarray:
    if len(window_bounds) - 1 < neighbor_count:
        raise BcgInputError(
            "AAS requires at least aas_neighbor_count + 1 complete beats"
        )
    correction_sum = np.zeros_like(data[eeg_indices])
    correction_count = np.zeros(data.shape[1], dtype=np.int64)
    anchors = np.asarray(
        [(start + stop) // 2 for start, stop in window_bounds],
        dtype=np.int64,
    )
    for event_index, (start, stop) in enumerate(window_bounds):
        distances = np.abs(anchors - anchors[event_index])
        distances[event_index] = np.iinfo(np.int64).max
        neighbor_indices = np.argsort(distances, kind="stable")[:neighbor_count]
        template = np.mean(
            np.stack(
                [
                    data[eeg_indices, window_bounds[index][0] : window_bounds[index][1]]
                    for index in neighbor_indices
                ],
                axis=0,
            ),
            axis=0,
        )
        correction_sum[:, start:stop] += template
        correction_count[start:stop] += 1

    corrected = data.copy()
    corrected_eeg = corrected[eeg_indices]
    covered = correction_count > 0
    corrected_eeg[:, covered] -= (
        correction_sum[:, covered] / correction_count[covered]
    )
    corrected[eeg_indices] = corrected_eeg
    return corrected


def _correct_pca_obs(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    eeg_indices: np.ndarray,
    anchor_samples: np.ndarray,
    corrected_samples: np.ndarray,
    sampling_rate: float,
    n_components: int,
) -> np.ndarray:
    effective_anchors = _effective_pca_obs_anchors(
        anchor_samples,
        data.shape[1],
    )
    if effective_anchors.size < n_components + 1:
        raise BcgInputError(
            "PCA-OBS requires at least n_components + 1 effective beats"
        )
    peak_range = round(np.median(np.diff(effective_anchors)) / 2.0)
    if n_components > 2 * peak_range + 1:
        raise BcgInputError(
            "pca_obs_components exceeds the effective heartbeat window"
        )

    eeg_data = data[eeg_indices]
    eeg_names = [channel_names[int(index)] for index in eeg_indices]
    raw = mne.io.RawArray(
        eeg_data,
        mne.create_info(
            ch_names=eeg_names,
            sfreq=sampling_rate,
            ch_types=["eeg"] * len(eeg_names),
        ),
        verbose="ERROR",
    )
    corrected_raw = mne.preprocessing.apply_pca_obs(
        raw,
        picks=eeg_names,
        qrs_times=effective_anchors.astype(np.float64) / sampling_rate,
        n_components=n_components,
        copy=True,
        verbose="ERROR",
    )
    try:
        corrected_eeg = corrected_raw.get_data()
    finally:
        raw.close()
        corrected_raw.close()

    channel_means = np.mean(eeg_data, axis=1, keepdims=True)
    corrected_eeg += channel_means
    corrected = data.copy()
    corrected_eeg_splice = corrected_eeg[:, corrected_samples]
    corrected[eeg_indices[:, None], corrected_samples] = corrected_eeg_splice
    return corrected


def _effective_pca_obs_anchors(
    anchor_samples: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    peak_range = round(np.median(np.diff(anchor_samples)) / 2.0)
    effective_count = anchor_samples.size
    while (
        effective_count > 0
        and anchor_samples[effective_count - 1] + peak_range > sample_count
    ):
        effective_count -= 1
    if effective_count < 2:
        raise BcgInputError(
            "PCA-OBS requires at least two effective heartbeat anchors"
        )
    return anchor_samples[:effective_count]
