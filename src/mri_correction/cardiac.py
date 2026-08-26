"""Independent ECG R-peak detection for recordings acquired during fMRI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from .bcg_config import DetectorConfig


class CardiacInputError(ValueError):
    """Raised when an ECG cannot be processed by the detector."""


@dataclass(frozen=True, slots=True)
class CardiacDetectionQuality:
    """Quality summary for one independent ECG detection."""

    candidate_count: int
    accepted_count: int
    rejected_count: int
    median_rr_seconds: float
    rr_iqr_seconds: float
    minimum_rr_seconds: float
    maximum_rr_seconds: float
    implied_rate_bpm: float
    template_correlation_median: float
    rejected_low_correlation: int
    rejected_double_mark: int
    rejected_interval: int
    status: str


@dataclass(frozen=True, slots=True)
class CardiacDetection:
    """Independent ECG R-peak samples and their quality summary."""

    peak_samples: npt.NDArray[np.int64]
    quality: CardiacDetectionQuality


@dataclass(frozen=True, slots=True)
class _CandidateSet:
    peaks: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float64]
    rejected_double_mark: int


@dataclass(frozen=True, slots=True)
class _SelectionResult:
    peaks: npt.NDArray[np.int64]
    correlations: npt.NDArray[np.float64]
    rejected_low_correlation: int
    rejected_interval: int
    rejected_double_mark: int


def detect_r_peaks(
    ecg: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    config: DetectorConfig,
) -> CardiacDetection:
    """Detect ECG R peaks using only the supplied ECG samples.

    The detector follows the MRI-specific FMRIB sequence: QRS-enhancing
    conditioning, nonnegative k-Teager energy, adaptive thresholding, and a
    morphology-and-rhythm refinement stage. Returned samples always refer to
    the original ECG coordinate system.
    """
    original = _validate_ecg(ecg)
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    _validate_record_length(original.size, sampling_rate, config)

    centered = original - np.median(original)
    scale = _robust_scale(centered)
    standardized = centered / scale
    conditioned = _bandpass(standardized, sampling_rate, config)
    energy = _teager_energy(conditioned, sampling_rate, config)
    candidates, candidate_scores = _generate_candidates(
        energy,
        conditioned,
        sampling_rate,
        config,
    )
    if candidates.size < 3:
        raise CardiacInputError(
            "ECG detector found fewer than three candidate cardiac events"
        )

    period = _estimate_period(candidates, sampling_rate, config)
    initial = _consolidate_candidates(
        candidates,
        candidate_scores,
        period=period,
        sampling_rate=sampling_rate,
    )
    if initial.peaks.size < 3:
        raise CardiacInputError(
            "ECG detector could not construct a cardiac-event seed train"
        )

    morphology_signal = np.abs(centered)
    template = _build_template(
        morphology_signal,
        initial.peaks,
        sampling_rate,
        config,
    )
    selection = _select_events(
        initial.peaks,
        morphology_signal,
        template,
        period=period,
        sampling_rate=sampling_rate,
        config=config,
    )
    peaks = selection.peaks
    correlations = selection.correlations
    rejected_low_correlation = selection.rejected_low_correlation
    rejected_interval = selection.rejected_interval
    rejected_double_mark = (
        initial.rejected_double_mark + selection.rejected_double_mark
    )

    for _ in range(config.refinement_iterations):
        peaks = _align_events(peaks, morphology_signal, sampling_rate, config)
        selection = _select_events(
            peaks,
            morphology_signal,
            template,
            period=period,
            sampling_rate=sampling_rate,
            config=config,
        )
        peaks = selection.peaks
        correlations = selection.correlations
        rejected_low_correlation += selection.rejected_low_correlation
        rejected_interval += selection.rejected_interval
        rejected_double_mark += selection.rejected_double_mark
        template = _build_template(
            morphology_signal,
            peaks,
            sampling_rate,
            config,
        )

    peaks = _recover_missing_events(
        peaks,
        morphology_signal,
        template,
        period=period,
        sampling_rate=sampling_rate,
        config=config,
    )
    peaks = _align_events(peaks, morphology_signal, sampling_rate, config)
    final_selection = _select_events(
        peaks,
        morphology_signal,
        template,
        period=period,
        sampling_rate=sampling_rate,
        config=config,
    )
    peaks = final_selection.peaks
    correlations = final_selection.correlations
    rejected_low_correlation += final_selection.rejected_low_correlation
    rejected_interval += final_selection.rejected_interval
    rejected_double_mark += final_selection.rejected_double_mark
    if peaks.size < 3:
        raise CardiacInputError(
            "ECG detector rejected too many events to form a cardiac train"
        )

    quality = _quality_summary(
        candidate_count=int(candidates.size),
        peak_samples=peaks,
        correlations=correlations,
        sampling_rate=sampling_rate,
        rejected_low_correlation=rejected_low_correlation,
        rejected_double_mark=rejected_double_mark,
        rejected_interval=rejected_interval,
        config=config,
    )
    return CardiacDetection(peak_samples=peaks, quality=quality)


def _validate_ecg(ecg: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(ecg)
    if values.ndim != 1 or values.size < 2:
        raise CardiacInputError("ECG must be a one-dimensional signal")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.number,
    ):
        raise CardiacInputError("ECG must contain finite numeric samples")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise CardiacInputError("ECG must contain finite numeric samples")
    return values


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    if (
        isinstance(sampling_rate_hz, bool)
        or not isinstance(sampling_rate_hz, Real)
        or not math.isfinite(float(sampling_rate_hz))
        or sampling_rate_hz <= 0.0
    ):
        raise CardiacInputError("sampling rate must be finite and positive")
    return float(sampling_rate_hz)


def _validate_record_length(
    sample_count: int,
    sampling_rate: float,
    config: DetectorConfig,
) -> None:
    low_hz, high_hz = config.preprocessing_band_hz
    if high_hz >= 0.5 * sampling_rate:
        raise CardiacInputError(
            "preprocessing_band_hz must stay below the Nyquist frequency"
        )
    template_start, template_end = config.template_window_seconds
    window_samples = round((template_end - template_start) * sampling_rate)
    smoothing_samples = round(config.teager_smoothing_seconds * sampling_rate)
    filter_padding = 3 * (2 * 4 + 1)
    minimum_samples = max(window_samples, smoothing_samples, filter_padding) + 2
    if sample_count <= minimum_samples:
        raise CardiacInputError(
            "ECG is too short for the configured filter and template windows"
        )
    if low_hz <= 0.0 or low_hz >= high_hz:
        raise CardiacInputError("preprocessing_band_hz must be increasing")


def _robust_scale(signal: np.ndarray) -> float:
    mad = np.median(np.abs(signal - np.median(signal)))
    if mad > 0.0:
        return float(1.4826 * mad)
    standard_deviation = float(np.std(signal))
    if standard_deviation > 0.0:
        return standard_deviation
    raise CardiacInputError("ECG has no measurable variation")


def _bandpass(
    signal: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> np.ndarray:
    sos = butter(
        4,
        config.preprocessing_band_hz,
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    try:
        return sosfiltfilt(sos, signal)
    except ValueError as error:
        raise CardiacInputError(
            "ECG is too short for zero-phase band-pass filtering"
        ) from error


def _teager_energy(
    signal: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> np.ndarray:
    k = max(1, round(sampling_rate / (4.0 * config.teager_emphasis_hz)))
    if 2 * k >= signal.size:
        raise CardiacInputError("ECG is too short for the configured Teager lag")

    energy = np.zeros_like(signal)
    energy[k:-k] = signal[k:-k] ** 2 - signal[:-2 * k] * signal[2 * k :]
    energy = np.maximum(energy, 0.0)
    smoothing_samples = max(
        1,
        round(config.teager_smoothing_seconds * sampling_rate),
    )
    return uniform_filter1d(energy, size=smoothing_samples, mode="nearest")


def _generate_candidates(
    energy: np.ndarray,
    conditioned: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate permissive candidates with an adaptive MFR-inspired gate."""
    short_window = max(3, round(0.12 * sampling_rate))
    rhythm_window = max(short_window, round(0.8 * sampling_rate))
    moving_energy = uniform_filter1d(energy, short_window, mode="nearest")
    slope = uniform_filter1d(
        np.abs(np.gradient(conditioned)),
        short_window,
        mode="nearest",
    )
    rhythm = uniform_filter1d(energy, rhythm_window, mode="nearest")

    energy_noise = _robust_scale(energy)
    moving_noise = _robust_scale(moving_energy)
    slope_noise = _robust_scale(slope)
    rhythm_noise = _robust_scale(rhythm)
    m_threshold = np.median(moving_energy) + (
        config.candidate_prominence_mad * moving_noise
    )
    f_threshold = np.median(slope) + config.candidate_prominence_mad * slope_noise
    r_threshold = np.median(rhythm) + config.candidate_prominence_mad * rhythm_noise
    adaptive_threshold = np.maximum(
        m_threshold,
        0.5 * r_threshold + 0.5 * f_threshold,
    )

    distance = max(1, round(config.candidate_refractory_seconds * sampling_rate))
    prominence = max(
        config.candidate_prominence_mad * energy_noise,
        np.finfo(np.float64).eps,
    )
    candidate_indices, properties = find_peaks(
        energy,
        distance=distance,
        prominence=prominence,
    )
    if candidate_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    accepted = (
        moving_energy[candidate_indices] >= adaptive_threshold
    ) | (
        energy[candidate_indices]
        >= np.median(energy) + config.candidate_prominence_mad * energy_noise
    )
    candidate_indices = candidate_indices[accepted].astype(np.int64, copy=False)
    scores = properties["prominences"][accepted].astype(np.float64, copy=False)
    return candidate_indices, scores


def _estimate_period(
    candidates: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> float:
    intervals = np.diff(candidates) / sampling_rate
    if intervals.size < 2:
        raise CardiacInputError("ECG detector found too few candidate intervals")

    minimum_rr = config.minimum_rr_seconds
    maximum_rr = config.maximum_rr_seconds
    lag_two = (candidates[2:] - candidates[:-2]) / sampling_rate
    short_intervals = intervals[intervals < minimum_rr]
    if short_intervals.size >= max(2, intervals.size // 4):
        period = _interval_mode(lag_two, minimum_rr, maximum_rr)
        if period is not None:
            return period

    period = _interval_mode(intervals, minimum_rr, maximum_rr)
    if period is None:
        raise CardiacInputError(
            "ECG detector could not estimate a physiological cardiac period"
        )
    return period


def _interval_mode(
    intervals: np.ndarray,
    minimum_seconds: float,
    maximum_seconds: float,
) -> float | None:
    valid = intervals[
        (intervals >= minimum_seconds) & (intervals <= maximum_seconds)
    ]
    if valid.size == 0:
        return None
    tolerance = 0.06
    best_count = -1
    best_center = float(valid[0])
    for center in np.sort(valid):
        count = int(np.sum(np.abs(valid - center) <= tolerance))
        if count > best_count:
            best_count = count
            best_center = float(center)
    cluster = valid[np.abs(valid - best_center) <= tolerance]
    return float(np.median(cluster))


def _consolidate_candidates(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    period: float,
    sampling_rate: float,
) -> _CandidateSet:
    order = np.argsort(candidates, kind="stable")
    sorted_peaks = candidates[order]
    sorted_scores = scores[order]
    selected_peaks: list[int] = []
    selected_scores: list[float] = []
    rejected_double_mark = 0
    period_samples = period * sampling_rate
    for peak, score in zip(sorted_peaks, sorted_scores, strict=True):
        if not selected_peaks:
            selected_peaks.append(int(peak))
            selected_scores.append(float(score))
            continue
        interval = peak - selected_peaks[-1]
        if interval < 0.5 * period_samples:
            rejected_double_mark += 1
            if score > selected_scores[-1]:
                selected_peaks[-1] = int(peak)
                selected_scores[-1] = float(score)
            continue
        selected_peaks.append(int(peak))
        selected_scores.append(float(score))
    return _CandidateSet(
        peaks=np.asarray(selected_peaks, dtype=np.int64),
        scores=np.asarray(selected_scores, dtype=np.float64),
        rejected_double_mark=rejected_double_mark,
    )


def _build_template(
    morphology_signal: np.ndarray,
    peaks: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> np.ndarray:
    start, stop = _window_samples(config.template_window_seconds, sampling_rate)
    epochs = []
    for peak in peaks:
        left = int(peak) + start
        right = int(peak) + stop
        if left < 0 or right > morphology_signal.size:
            continue
        epoch = morphology_signal[left:right].copy()
        epoch -= np.median(epoch)
        norm = np.linalg.norm(epoch)
        if norm > 0.0:
            epochs.append(epoch / norm)
    if len(epochs) < 3:
        raise CardiacInputError(
            "ECG detector could not construct a complete QRS template"
        )
    return np.mean(np.stack(epochs, axis=0), axis=0)


def _select_events(
    peaks: np.ndarray,
    morphology_signal: np.ndarray,
    template: np.ndarray,
    *,
    period: float,
    sampling_rate: float,
    config: DetectorConfig,
) -> _SelectionResult:
    aligned = _align_events(peaks, morphology_signal, sampling_rate, config)
    correlations = np.asarray(
        [
            _event_correlation(
                morphology_signal,
                int(peak),
                template,
                sampling_rate,
                config,
            )
            for peak in aligned
        ],
        dtype=np.float64,
    )
    good = correlations >= config.correlation_threshold
    rejected_low_correlation = int(np.count_nonzero(~good))
    aligned = aligned[good]
    correlations = correlations[good]
    if aligned.size == 0:
        return _SelectionResult(
            peaks=np.empty(0, dtype=np.int64),
            correlations=np.empty(0, dtype=np.float64),
            rejected_low_correlation=rejected_low_correlation,
            rejected_interval=0,
            rejected_double_mark=0,
        )

    order = np.argsort(aligned, kind="stable")
    consolidated = _consolidate_candidates(
        aligned[order],
        correlations[order],
        period=period,
        sampling_rate=sampling_rate,
    )
    selected_correlations = consolidated.scores
    intervals = np.diff(consolidated.peaks) / sampling_rate
    rejected_interval = int(
        np.count_nonzero(intervals < config.minimum_rr_seconds)
    )
    return _SelectionResult(
        peaks=consolidated.peaks,
        correlations=selected_correlations,
        rejected_low_correlation=rejected_low_correlation,
        rejected_interval=rejected_interval,
        rejected_double_mark=consolidated.rejected_double_mark,
    )


def _align_events(
    peaks: np.ndarray,
    morphology_signal: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> np.ndarray:
    aligned = np.asarray(
        [
            _align_peak(
                int(peak),
                morphology_signal,
                sampling_rate,
                config,
            )
            for peak in peaks
        ],
        dtype=np.int64,
    )
    return np.unique(aligned)


def _align_peak(
    peak: int,
    morphology_signal: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> int:
    half_window = max(
        1,
        round(
            min(
                0.06,
                0.25 * config.candidate_refractory_seconds,
            )
            * sampling_rate
        ),
    )
    left = max(0, peak - half_window)
    right = min(morphology_signal.size, peak + half_window + 1)
    return int(left + np.argmax(morphology_signal[left:right]))


def _event_correlation(
    morphology_signal: np.ndarray,
    peak: int,
    template: np.ndarray,
    sampling_rate: float,
    config: DetectorConfig,
) -> float:
    start, stop = _window_samples(config.template_window_seconds, sampling_rate)
    left = peak + start
    right = peak + stop
    if left < 0 or right > morphology_signal.size:
        return 0.0
    epoch = morphology_signal[left:right].copy()
    epoch -= np.median(epoch)
    norm = np.linalg.norm(epoch)
    if norm == 0.0:
        return 0.0
    normalized_epoch = epoch / norm
    normalized_template = template - np.mean(template)
    template_norm = np.linalg.norm(normalized_template)
    if template_norm == 0.0:
        return 0.0
    return float(
        np.dot(normalized_epoch, normalized_template / template_norm)
    )


def _recover_missing_events(
    peaks: np.ndarray,
    morphology_signal: np.ndarray,
    template: np.ndarray,
    *,
    period: float,
    sampling_rate: float,
    config: DetectorConfig,
) -> np.ndarray:
    if peaks.size < 2:
        return peaks
    recovered = list(int(peak) for peak in peaks)
    for left, right in pairwise(peaks):
        interval = (right - left) / sampling_rate
        missing_count = max(0, round(interval / period) - 1)
        for index in range(1, missing_count + 1):
            expected = round(left + index * period * sampling_rate)
            candidate = _align_peak(
                expected,
                morphology_signal,
                sampling_rate,
                config,
            )
            if any(
                abs(candidate - existing)
                < round(config.candidate_refractory_seconds * sampling_rate)
                for existing in recovered
            ):
                continue
            correlation = _event_correlation(
                morphology_signal,
                candidate,
                template,
                sampling_rate,
                config,
            )
            if correlation >= config.correlation_threshold:
                recovered.append(candidate)
    return np.asarray(sorted(set(recovered)), dtype=np.int64)


def _window_samples(
    window_seconds: tuple[float, float],
    sampling_rate: float,
) -> tuple[int, int]:
    start, stop = window_seconds
    return round(start * sampling_rate), round(stop * sampling_rate)


def _quality_summary(
    *,
    candidate_count: int,
    peak_samples: np.ndarray,
    correlations: np.ndarray,
    sampling_rate: float,
    rejected_low_correlation: int,
    rejected_double_mark: int,
    rejected_interval: int,
    config: DetectorConfig,
) -> CardiacDetectionQuality:
    intervals = np.diff(peak_samples) / sampling_rate
    if intervals.size:
        median_rr = float(np.median(intervals))
        rr_iqr = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
        minimum_rr = float(np.min(intervals))
        maximum_rr = float(np.max(intervals))
        implied_rate = 60.0 / median_rr
    else:
        median_rr = float("nan")
        rr_iqr = float("nan")
        minimum_rr = float("nan")
        maximum_rr = float("nan")
        implied_rate = float("nan")
    outside_range = bool(
        intervals.size
        and np.any(
            (intervals < config.minimum_rr_seconds)
            | (intervals > config.maximum_rr_seconds)
        )
    )
    status = "degraded" if outside_range else "ok"
    rejected_count = (
        rejected_low_correlation + rejected_double_mark + rejected_interval
    )
    return CardiacDetectionQuality(
        candidate_count=candidate_count,
        accepted_count=int(peak_samples.size),
        rejected_count=rejected_count,
        median_rr_seconds=median_rr,
        rr_iqr_seconds=rr_iqr,
        minimum_rr_seconds=minimum_rr,
        maximum_rr_seconds=maximum_rr,
        implied_rate_bpm=implied_rate,
        template_correlation_median=float(np.median(correlations)),
        rejected_low_correlation=rejected_low_correlation,
        rejected_double_mark=rejected_double_mark,
        rejected_interval=rejected_interval,
        status=status,
    )
