"""Adaptive noise cancellation from the fourth FMRIB FASTR stage."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, firls

from .fastr_types import AncCorrection, FastrInputError
from .fastr_validation import (
    validate_channel_indices,
    validate_positive_finite,
    validate_recording,
    validate_sampling_rate,
)

_HIGH_PASS_HZ = 2.0
_TRANSITION_FRACTION = 0.15
_FILTER_ORDER_FACTOR = 1.2
_STEP_NUMERATOR = 0.05


def fmrib_lms(
    reference: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    filter_order: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the sample update implemented by FMRIB's ``fastranc``."""
    refs = _validate_vector(reference, "reference")
    target = _validate_vector(desired, "desired")
    if refs.shape != target.shape:
        raise FastrInputError(
            "reference and desired signals must have equal length"
        )
    order = _validate_filter_order(filter_order, refs.size)
    step = validate_positive_finite(step_size, name="ANC step size")

    weights = np.zeros(order + 1, dtype=np.float64)
    error = np.zeros(refs.size, dtype=np.float64)
    noise = np.zeros(refs.size, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(order, refs.size):
            window = refs[index - order : index + 1][::-1]
            noise[index] = weights @ window
            error[index] = target[index] - noise[index]
            weights += 2.0 * step * error[index] * window
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(noise)):
        raise FastrInputError("adaptive noise cancellation diverged")
    return error, noise


def adaptive_noise_cancel(
    corrected: npt.ArrayLike,
    artifact_estimate: npt.ArrayLike,
    *,
    sampling_rate: float,
    filter_order: int,
    excluded_channels: Sequence[int],
    sample_slice: slice | None = None,
) -> AncCorrection:
    """Remove residual artifact using the FMRIB normalized LMS setup."""
    recording = validate_recording(corrected)
    artifact = validate_recording(artifact_estimate)
    if artifact.shape != recording.shape:
        raise FastrInputError(
            "artifact estimate must match the corrected recording shape"
        )
    if np.iscomplexobj(recording) or np.iscomplexobj(artifact):
        raise FastrInputError("adaptive noise cancellation requires real data")

    rate = validate_sampling_rate(sampling_rate)
    span = _validate_sample_slice(sample_slice, recording.shape[1])
    span_length = span.stop - span.start
    order = _validate_filter_order(filter_order, span_length)
    excluded = validate_channel_indices(
        excluded_channels,
        recording.shape[0],
        name="excluded channels",
    )

    output = recording.astype(np.float64, copy=True)
    scales = np.full(recording.shape[0], np.nan, dtype=np.float64)
    steps = np.full(recording.shape[0], np.nan, dtype=np.float64)
    high_pass = _make_high_pass(rate)
    for channel_index, channel in enumerate(recording):
        if channel_index in excluded or np.std(channel) == 0.0:
            continue

        reference = artifact[channel_index, span]
        reference_energy = float(reference @ reference)
        reference_variance = float(np.var(reference, ddof=1))
        if reference_energy <= 0.0 or reference_variance <= 0.0:
            raise FastrInputError(
                f"ANC reference variance is zero on channel {channel_index}"
            )

        desired = filtfilt(high_pass, 1.0, channel)[span]
        scale = float((desired @ reference) / reference_energy)
        scaled_reference = scale * reference
        scaled_variance = float(np.var(scaled_reference, ddof=1))
        if not math.isfinite(scaled_variance) or scaled_variance <= 0.0:
            raise FastrInputError(
                f"ANC scaled reference variance is zero on channel {channel_index}"
            )
        step = _STEP_NUMERATOR / (order * scaled_variance)
        _, noise = fmrib_lms(
            scaled_reference,
            desired,
            filter_order=order,
            step_size=step,
        )
        output[channel_index, span] -= noise
        scales[channel_index] = scale
        steps[channel_index] = step

    return AncCorrection(
        data=output,
        reference_scales=scales,
        step_sizes=steps,
        filter_order=order,
    )


def _validate_vector(values: npt.ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size == 0:
        raise FastrInputError(f"{name} must be a nonempty one-dimensional array")
    if (
        np.issubdtype(vector.dtype, np.bool_)
        or not np.issubdtype(vector.dtype, np.number)
        or np.iscomplexobj(vector)
        or not np.all(np.isfinite(vector))
    ):
        raise FastrInputError(f"{name} must contain finite real numbers")
    return vector.astype(np.float64, copy=False)


def _validate_filter_order(value: object, sample_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise FastrInputError("ANC filter order must be a positive integer")
    order = int(value)
    if order >= sample_count:
        raise FastrInputError(
            "ANC filter order must be shorter than the processed signal"
        )
    return order


def _validate_sample_slice(value: object, sample_count: int) -> slice:
    if value is None:
        return slice(0, sample_count)
    if not isinstance(value, slice) or value.step is not None:
        raise FastrInputError("ANC sample slice must be a contiguous slice")
    start = 0 if value.start is None else value.start
    stop = sample_count if value.stop is None else value.stop
    if any(
        isinstance(bound, bool) or not isinstance(bound, Integral)
        for bound in (start, stop)
    ):
        raise FastrInputError("ANC sample slice bounds must be integers")
    start = int(start)
    stop = int(stop)
    if not 0 <= start < stop <= sample_count:
        raise FastrInputError("ANC sample slice is outside the recording")
    return slice(start, stop)


def _make_high_pass(sampling_rate: float) -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    stop_frequency = _HIGH_PASS_HZ * (1.0 - _TRANSITION_FRACTION)
    if _HIGH_PASS_HZ >= nyquist:
        raise FastrInputError("sampling rate is too low for the ANC high-pass")
    order = math.floor(
        _FILTER_ORDER_FACTOR * sampling_rate / stop_frequency + 0.5
    )
    order += order % 2
    return firls(
        order + 1,
        (0.0, stop_frequency / nyquist, _HIGH_PASS_HZ / nyquist, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )
