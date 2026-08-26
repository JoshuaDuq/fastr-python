"""Separate artifact-suppression and signal-transfer measurements."""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import numpy.typing as npt


class MetricInputError(ValueError):
    """Raised when a measurement is asked for something it cannot measure."""


@dataclass(frozen=True, slots=True)
class ToneTransfer:
    """How one exact frequency survived a correction."""

    amplitude_ratio: float
    phase_error_degrees: float


def tone_transfer(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    *,
    frequency: float,
    sampling_rate: float,
) -> ToneTransfer:
    """Measure amplitude and phase transfer at one exact frequency.

    Both signals are projected onto the same sine/cosine pair. Separation from
    other frequencies is exact when the window holds a whole number of cycles of
    both, which is why benchmarks measure over whole volumes; otherwise strong
    neighbouring energy leaks into the estimate.
    """
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    _validate_frequency(frequency, sampling_rate)

    basis = _tone_basis(reference.size, frequency, sampling_rate)
    reference_phasor = _project(basis, reference)
    result_phasor = _project(basis, result)
    if reference_phasor == 0.0:
        raise MetricInputError("the injected signal has no energy at this frequency")

    rotation = result_phasor / reference_phasor
    return ToneTransfer(
        amplitude_ratio=abs(rotation),
        phase_error_degrees=math.degrees(np.angle(rotation)),
    )


def band_rms_ratio(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    *,
    low: float,
    high: float,
    sampling_rate: float,
) -> float:
    """Measure the RMS a correction kept inside one frequency band."""
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    _validate_band(low, high, sampling_rate)

    frequencies = np.fft.rfftfreq(reference.size, d=1.0 / sampling_rate)
    inside = (frequencies >= low) & (frequencies <= high)
    reference_power = _band_power(reference, inside)
    if reference_power == 0.0:
        raise MetricInputError("the injected signal has no energy in this band")
    return math.sqrt(_band_power(result, inside) / reference_power)


def trigger_locked_rms(
    data: npt.ArrayLike,
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> npt.NDArray[np.float64]:
    """Measure the per-channel RMS of the artifact that repeats at the triggers.

    Anything not locked to the triggers averages towards zero, so this isolates
    residual artifact from ongoing EEG. Fractional trigger positions are sampled
    by linear interpolation rather than rounded to a different timing grid.
    """
    recording = _validate_recording(data)
    template = trigger_locked_template(
        recording,
        triggers,
        epoch_samples=epoch_samples,
    )
    return np.sqrt(np.mean(template**2, axis=1))


def trigger_locked_template(
    data: npt.ArrayLike,
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> npt.NDArray[np.float64]:
    """Return the fractional-position average of epochs at ``triggers``."""
    recording = _validate_recording(data)
    positions = _validate_triggers(
        triggers,
        epoch_samples=epoch_samples,
        sample_count=recording.shape[1],
    )
    epochs = _extract_fractional_epochs(recording, positions, epoch_samples)
    return epochs.mean(axis=1)


def event_locked_rms_ratio(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    event_starts: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> float:
    """Measure event-locked RMS transfer using fractional event positions."""
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    positions = _validate_triggers(
        event_starts,
        epoch_samples=epoch_samples,
        sample_count=reference.size,
    )
    reference_epochs = _extract_fractional_epochs(
        reference[np.newaxis, :],
        positions,
        epoch_samples,
    )[0]
    result_epochs = _extract_fractional_epochs(
        result[np.newaxis, :],
        positions,
        epoch_samples,
    )[0]
    reference_rms = np.sqrt(np.mean(reference_epochs.mean(axis=0) ** 2))
    if reference_rms == 0.0:
        raise MetricInputError("the injected event has no event-locked energy")
    return float(np.sqrt(np.mean(result_epochs.mean(axis=0) ** 2)) / reference_rms)


def _validate_signal(signal: npt.ArrayLike, *, name: str) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim != 1 or values.size < 2:
        raise MetricInputError(f"{name} must be a one-dimensional signal")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype, np.number
    ):
        raise MetricInputError(f"{name} must contain finite numbers")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise MetricInputError(f"{name} must contain finite numbers")
    return values


def _validate_recording(data: npt.ArrayLike) -> np.ndarray:
    recording = np.asarray(data)
    if recording.ndim != 2 or recording.shape[0] == 0 or recording.shape[1] == 0:
        raise MetricInputError("data must have shape (channels, samples)")
    if np.issubdtype(recording.dtype, np.bool_) or not np.issubdtype(
        recording.dtype, np.number
    ):
        raise MetricInputError("data must contain finite numbers")
    if not np.all(np.isfinite(recording)):
        raise MetricInputError("data must contain finite numbers")
    return recording.astype(np.float64, copy=False)


def _validate_frequency(frequency: float, sampling_rate: float) -> None:
    if (
        isinstance(frequency, bool)
        or not isinstance(frequency, Real)
        or not math.isfinite(float(frequency))
        or frequency <= 0.0
    ):
        raise MetricInputError("frequency must be finite and positive")
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, Real)
        or not math.isfinite(float(sampling_rate))
        or sampling_rate <= 0.0
    ):
        raise MetricInputError("sampling rate must be finite and positive")
    if frequency >= 0.5 * sampling_rate:
        raise MetricInputError("frequency must stay below the Nyquist frequency")


def _validate_band(low: float, high: float, sampling_rate: float) -> None:
    _validate_frequency(high, sampling_rate)
    if (
        isinstance(low, bool)
        or not isinstance(low, Real)
        or not math.isfinite(float(low))
        or low < 0.0
        or low >= high
    ):
        raise MetricInputError("the band must be positive and increasing")


def _validate_triggers(
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
    sample_count: int,
) -> np.ndarray:
    if not isinstance(epoch_samples, int) or epoch_samples < 1:
        raise MetricInputError("epoch samples must be a positive integer")
    positions = np.asarray(triggers)
    if positions.ndim != 1 or positions.size == 0:
        raise MetricInputError("triggers must be a nonempty one-dimensional array")
    if np.issubdtype(positions.dtype, np.bool_) or not np.issubdtype(
        positions.dtype, np.number
    ):
        raise MetricInputError("triggers must contain finite numbers")
    positions = positions.astype(np.float64, copy=False)
    if not np.all(np.isfinite(positions)) or np.any(positions < 0.0):
        raise MetricInputError("triggers must contain finite nonnegative numbers")
    if np.any(np.diff(positions) <= 0.0):
        raise MetricInputError("triggers must be strictly increasing")
    if positions[-1] + epoch_samples - 1.0 > sample_count - 1:
        raise MetricInputError("the measured epochs extend beyond the recording")
    return positions


def _extract_fractional_epochs(
    recording: np.ndarray,
    positions: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    offsets = np.arange(epoch_samples, dtype=np.float64)
    sample_positions = positions[:, np.newaxis] + offsets
    lower = np.floor(sample_positions).astype(np.int64)
    upper = np.minimum(lower + 1, recording.shape[1] - 1)
    fraction = sample_positions - lower
    return (
        recording[:, lower] * (1.0 - fraction[np.newaxis, :, :])
        + recording[:, upper] * fraction[np.newaxis, :, :]
    )


def _tone_basis(
    sample_count: int,
    frequency: float,
    sampling_rate: float,
) -> np.ndarray:
    times = np.arange(sample_count, dtype=np.float64) / sampling_rate
    return np.stack(
        [np.sin(2 * np.pi * frequency * times), np.cos(2 * np.pi * frequency * times)],
        axis=1,
    )


def _project(basis: np.ndarray, signal: np.ndarray) -> complex:
    coefficients, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return complex(coefficients[0], coefficients[1])


def _band_power(signal: np.ndarray, inside: np.ndarray) -> float:
    spectrum = np.fft.rfft(signal)
    return float(np.sum(np.abs(spectrum[inside]) ** 2))
