"""Auditable averaged-artifact subtraction methods."""

import numpy as np
import numpy.typing as npt


class CorrectionInputError(ValueError):
    """Raised when correction inputs violate acquisition assumptions."""


def analyzer_reference(
    data: npt.ArrayLike,
    volume_starts: npt.ArrayLike,
    *,
    samples_per_volume: int,
    window_size: int = 21,
) -> npt.NDArray[np.float64]:
    """Apply Analyzer-style whole-epoch, sliding-mean artifact subtraction."""
    recording = np.asarray(data)
    starts = np.asarray(volume_starts)
    _validate_reference_inputs(
        recording,
        starts,
        samples_per_volume=samples_per_volume,
        window_size=window_size,
    )

    corrected = recording.astype(np.float64, copy=True)
    rolling_sum = sum(
        (
            _baselined_epoch(recording, start, samples_per_volume)
            for start in starts[:window_size]
        ),
        start=np.zeros((recording.shape[0], samples_per_volume), dtype=np.float64),
    )
    current_window_start = 0
    half_window = window_size // 2
    last_window_start = starts.size - window_size

    for target_index, target_start in enumerate(starts):
        window_start = min(max(target_index - half_window, 0), last_window_start)
        while current_window_start < window_start:
            outgoing = starts[current_window_start]
            incoming = starts[current_window_start + window_size]
            rolling_sum -= _baselined_epoch(
                recording,
                outgoing,
                samples_per_volume,
            )
            rolling_sum += _baselined_epoch(
                recording,
                incoming,
                samples_per_volume,
            )
            current_window_start += 1

        target_slice = slice(target_start, target_start + samples_per_volume)
        corrected[:, target_slice] = recording[:, target_slice] - (
            rolling_sum / window_size
        )

    return corrected


def _baselined_epoch(
    data: np.ndarray,
    start: np.integer,
    samples_per_volume: int,
) -> npt.NDArray[np.float64]:
    epoch = data[:, start : start + samples_per_volume].astype(np.float64, copy=False)
    return epoch - epoch.mean(axis=1, keepdims=True)


def _validate_reference_inputs(
    data: np.ndarray,
    starts: np.ndarray,
    *,
    samples_per_volume: int,
    window_size: int,
) -> None:
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise CorrectionInputError("data must have shape (channels, samples)")
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        raise CorrectionInputError("data must contain only finite numeric values")
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
        message = "volume starts must be a one-dimensional integer array"
        raise CorrectionInputError(message)
    if not isinstance(samples_per_volume, int) or samples_per_volume < 1:
        raise CorrectionInputError("samples_per_volume must be a positive integer")
    if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        message = "window_size must be an odd integer of at least three"
        raise CorrectionInputError(message)
    if starts.size < window_size:
        raise CorrectionInputError("the acquisition block is shorter than window_size")
    if starts[0] < 0 or np.any(np.diff(starts) != samples_per_volume):
        raise CorrectionInputError("volume starts must form one exact contiguous block")
    if starts[-1] + samples_per_volume > data.shape[1]:
        message = "the final volume epoch extends beyond the recording"
        raise CorrectionInputError(message)
