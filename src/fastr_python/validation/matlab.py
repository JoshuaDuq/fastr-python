"""Compare matched MATLAB and Python FASTR outputs.

Validation instrumentation: nothing in the correction pipeline imports this
module. It exists to measure a correction after the fact, and may be changed or
removed without affecting a run.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

from ..quality.residuals import block_residual_uv


class ComparisonError(ValueError):
    """Raised when validation recordings cannot be compared."""


def compare_arrays(
    *,
    raw: npt.ArrayLike,
    matlab: npt.ArrayLike,
    python: npt.ArrayLike,
    sampling_rate: float,
    repetition_time_seconds: float,
    ecg_index: int,
) -> dict[str, object]:
    """Return matched whole-span metrics for three channel-by-sample arrays."""
    recordings = {
        "raw": _recording(raw, "raw"),
        "matlab": _recording(matlab, "MATLAB"),
        "python": _recording(python, "Python"),
    }
    shapes = {recording.shape for recording in recordings.values()}
    if len(shapes) != 1:
        raise ComparisonError("raw, MATLAB, and Python recordings need the same shape")

    rate = _positive_number(sampling_rate, "sampling rate")
    repetition_time = _positive_number(
        repetition_time_seconds,
        "repetition time",
    )
    channel_count = recordings["raw"].shape[0]
    if (
        isinstance(ecg_index, bool)
        or not isinstance(ecg_index, Integral)
        or not 0 <= int(ecg_index) < channel_count
    ):
        raise ComparisonError("ECG index is outside the channel range")
    ecg = int(ecg_index)
    eeg_indices = [index for index in range(channel_count) if index != ecg]
    if not eeg_indices:
        raise ComparisonError("comparison requires at least one EEG channel")

    rmse = np.sqrt(
        np.mean(
            (recordings["python"] - recordings["matlab"]) ** 2,
            axis=1,
        )
    )
    harmonic_rms = {
        name: _scanner_harmonic_rms(
            recording[eeg_indices],
            sampling_rate=rate,
            repetition_time_seconds=repetition_time,
        )
        for name, recording in recordings.items()
    }
    return {
        "sample_rmse": {
            "matlab_vs_python_uv_by_channel": (rmse * 1e6).tolist(),
            "median_uv": float(np.median(rmse) * 1e6),
        },
        "scanner_harmonic_rms": {
            f"{name}_uv": value for name, value in harmonic_rms.items()
        },
        "broadband_transfer": {
            name: _broadband_transfer(
                recordings["raw"][eeg_indices],
                recordings[name][eeg_indices],
                sampling_rate=rate,
                repetition_time_seconds=repetition_time,
            )
            for name in ("matlab", "python")
        },
        "ecg_correlation": {
            name: _ecg_correlation(
                recordings["raw"][ecg],
                recordings[name][ecg],
                sampling_rate=rate,
                repetition_time_seconds=repetition_time,
            )
            for name in ("matlab", "python")
        },
    }


def _recording(values: npt.ArrayLike, name: str) -> np.ndarray:
    recording = np.asarray(values)
    if (
        recording.ndim != 2
        or recording.shape[0] == 0
        or recording.shape[1] == 0
        or np.iscomplexobj(recording)
        or not np.issubdtype(recording.dtype, np.number)
        or not np.all(np.isfinite(recording))
    ):
        raise ComparisonError(
            f"{name} recording must be a nonempty finite real channel-by-sample array"
        )
    return recording.astype(np.float64, copy=False)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ComparisonError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ComparisonError(f"{name} must be a finite positive number")
    return number


def _scanner_harmonic_rms(
    data: np.ndarray,
    *,
    sampling_rate: float,
    repetition_time_seconds: float,
) -> float:
    fundamental = 1.0 / repetition_time_seconds
    maximum = min(100.0, float(np.nextafter(sampling_rate / 2.0, 0.0)))
    harmonics = tuple(
        order * fundamental
        for order in range(1, math.floor(maximum / fundamental) + 1)
        if abs(order * fundamental - 60.0) > 1.0
    )
    duration = data.shape[1] / sampling_rate
    residual = block_residual_uv(
        data * 1e6,
        sampling_rate=sampling_rate,
        harmonics=harmonics,
        block_seconds=duration,
    )
    if residual.shape[1] != 1:
        raise ComparisonError("recording is too short for scanner-harmonic measurement")
    return float(np.median(residual[:, 0]))


def _broadband_transfer(
    raw: np.ndarray,
    corrected: np.ndarray,
    *,
    sampling_rate: float,
    repetition_time_seconds: float,
) -> float:
    frequencies = np.fft.rfftfreq(raw.shape[1], 1.0 / sampling_rate)
    maximum = min(40.0, 0.45 * sampling_rate)
    keep = (frequencies >= 1.0) & (frequencies <= maximum)
    fundamental = 1.0 / repetition_time_seconds
    for order in range(1, math.floor(maximum / fundamental) + 1):
        keep &= np.abs(frequencies - order * fundamental) > 0.5
    keep &= np.abs(frequencies - 60.0) > 1.0
    if not np.any(keep):
        raise ComparisonError("no broadband frequency bins remain for comparison")

    raw_spectrum = np.fft.rfft(raw - raw.mean(axis=1, keepdims=True), axis=1)
    corrected_spectrum = np.fft.rfft(
        corrected - corrected.mean(axis=1, keepdims=True),
        axis=1,
    )
    raw_power = np.sum(np.abs(raw_spectrum[:, keep]) ** 2, axis=1)
    corrected_power = np.sum(
        np.abs(corrected_spectrum[:, keep]) ** 2,
        axis=1,
    )
    if np.any(raw_power <= 0.0):
        raise ComparisonError("raw broadband power is zero")
    return float(np.median(np.sqrt(corrected_power / raw_power)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        raise ComparisonError("ECG correlation requires nonconstant signals")
    value = float(np.corrcoef(left, right)[0, 1])
    return 1.0 if math.isclose(value, 1.0, abs_tol=1e-15) else value


def _ecg_correlation(
    raw: np.ndarray,
    corrected: np.ndarray,
    *,
    sampling_rate: float,
    repetition_time_seconds: float,
) -> float:
    """Correlate the physiological ECG band outside scanner-locked bins."""
    frequencies = np.fft.rfftfreq(raw.size, 1.0 / sampling_rate)
    keep = (frequencies >= 0.5) & (frequencies <= min(40.0, 0.45 * sampling_rate))
    fundamental = 1.0 / repetition_time_seconds
    for order in range(1, math.floor(40.0 / fundamental) + 1):
        keep &= np.abs(frequencies - order * fundamental) > 0.5
    keep &= np.abs(frequencies - 60.0) > 1.0
    if not np.any(keep):
        raise ComparisonError("no ECG frequency bins remain for comparison")

    filtered = []
    for signal in (raw, corrected):
        spectrum = np.fft.rfft(signal - signal.mean())
        spectrum[~keep] = 0.0
        filtered.append(np.fft.irfft(spectrum, n=signal.size))
    return _correlation(filtered[0], filtered[1])
