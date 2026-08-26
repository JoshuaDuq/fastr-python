"""Descriptive acquisition diagnostics that never become production metadata."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

_EVIDENCE_DECIMALS = 6


class DiagnosticInputError(ValueError):
    """Raised when diagnostic inputs violate acquisition assumptions."""


@dataclass(frozen=True, slots=True)
class SlicePeriodCandidate:
    """One candidate repetition period and the evidence supporting it."""

    period_samples: int
    autocorrelation: float


def estimate_slice_period_candidates(
    data: npt.ArrayLike,
    *,
    minimum_period: int,
    maximum_period: int,
    candidate_count: int = 5,
) -> tuple[SlicePeriodCandidate, ...]:
    """Rank repetition periods the raw gradient waveform is consistent with.

    Scoring uses the first difference, which emphasises the sharp gradient
    switching edges, and averages the per-channel correlation at each lag. A
    period that a shorter, equally supported candidate already implies is not
    reported again, so multiples do not crowd out the fundamental.

    This is diagnostic evidence only. Acquisition group timing must come from
    BIDS metadata, so no correction function accepts a `SlicePeriodCandidate`:
    an estimate can support or contradict declared timing, never replace it.
    """
    recording = _validate_recording(data)
    _validate_period_range(
        minimum_period,
        maximum_period,
        sample_count=recording.shape[1],
    )
    if not isinstance(candidate_count, int) or candidate_count < 1:
        raise DiagnosticInputError("candidate count must be a positive integer")

    periods = np.arange(minimum_period, maximum_period + 1)
    scores = _score_periods(np.diff(recording, axis=1), periods)
    ranked = _rank_isolated_periods(periods, scores, candidate_count)
    return tuple(
        SlicePeriodCandidate(
            period_samples=int(periods[index]),
            autocorrelation=float(scores[index]),
        )
        for index in ranked
    )


def _validate_recording(data: npt.ArrayLike) -> np.ndarray:
    recording = np.asarray(data)
    if recording.ndim != 2 or recording.shape[0] == 0 or recording.shape[1] == 0:
        raise DiagnosticInputError("data must have shape (channels, samples)")
    if np.issubdtype(recording.dtype, np.bool_) or not np.issubdtype(
        recording.dtype, np.number
    ):
        raise DiagnosticInputError("data must contain only finite numeric values")
    if not np.all(np.isfinite(recording)):
        raise DiagnosticInputError("data must contain only finite numeric values")
    return recording.astype(np.float64, copy=False)


def _validate_period_range(
    minimum_period: int,
    maximum_period: int,
    *,
    sample_count: int,
) -> None:
    if not isinstance(minimum_period, int) or not isinstance(maximum_period, int):
        raise DiagnosticInputError("the period range must be given as integers")
    if minimum_period < 1 or maximum_period <= minimum_period:
        raise DiagnosticInputError("the period range must be positive and increasing")
    if maximum_period >= sample_count - 1:
        raise DiagnosticInputError(
            "the period range must stay within the searched recording"
        )


def _score_periods(differences: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Mean per-channel correlation between the waveform and its lagged self."""
    scores = np.empty(periods.size, dtype=np.float64)
    for index, period in enumerate(periods):
        scores[index] = np.mean(
            _correlate(differences[:, :-period], differences[:, period:])
        )
    return scores


def _correlate(leading: np.ndarray, lagging: np.ndarray) -> np.ndarray:
    centered_leading = leading - leading.mean(axis=1, keepdims=True)
    centered_lagging = lagging - lagging.mean(axis=1, keepdims=True)
    norms = np.sqrt(
        np.sum(centered_leading**2, axis=1) * np.sum(centered_lagging**2, axis=1)
    )
    return np.divide(
        np.sum(centered_leading * centered_lagging, axis=1),
        norms,
        out=np.zeros(leading.shape[0]),
        where=norms > 0.0,
    )


def _rank_isolated_periods(
    periods: np.ndarray,
    scores: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    """Rank local maxima, dropping periods a shorter candidate already implies."""
    peaks, _ = find_peaks(scores)
    evidence = np.round(scores, _EVIDENCE_DECIMALS)
    fundamentals = np.array(
        [
            peak
            for position, peak in enumerate(peaks)
            if not np.any(
                (periods[peak] % periods[peaks[:position]] == 0)
                & (evidence[peaks[:position]] >= evidence[peak])
            )
        ],
        dtype=np.int64,
    )
    ranked = np.lexsort((periods[fundamentals], -evidence[fundamentals]))
    return fundamentals[ranked][:candidate_count]
