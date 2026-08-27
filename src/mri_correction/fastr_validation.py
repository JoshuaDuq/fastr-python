"""Shared validation for FASTR recordings, geometry, and parameters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

from .fastr_types import FastrInputError


def validate_recording(data: npt.ArrayLike) -> np.ndarray:
    recording = np.asarray(data)
    if recording.ndim != 2 or recording.shape[0] == 0 or recording.shape[1] == 0:
        raise FastrInputError("data must have shape (channels, samples)")
    if np.issubdtype(recording.dtype, np.bool_) or not np.issubdtype(
        recording.dtype, np.number
    ):
        raise FastrInputError("data must contain only finite numeric values")
    if not np.all(np.isfinite(recording)):
        raise FastrInputError("data must contain only finite numeric values")
    return recording


def validate_reference_channel(
    data: npt.ArrayLike,
    sample_count: int,
) -> np.ndarray:
    reference = np.asarray(data)
    if reference.ndim != 1 or reference.size != sample_count:
        raise FastrInputError(
            "reference channel must be one-dimensional with the geometry sample count"
        )
    if np.issubdtype(reference.dtype, np.bool_) or not np.issubdtype(
        reference.dtype,
        np.number,
    ):
        raise FastrInputError("reference channel must contain finite numeric values")
    if not np.all(np.isfinite(reference)):
        raise FastrInputError("reference channel must contain finite numeric values")
    return reference


def validate_group_triggers(group_triggers: npt.ArrayLike) -> np.ndarray:
    triggers = np.asarray(group_triggers)
    if triggers.ndim != 1 or triggers.size < 2:
        raise FastrInputError("group triggers must be a one-dimensional array")
    if np.issubdtype(triggers.dtype, np.bool_) or not np.issubdtype(
        triggers.dtype, np.number
    ):
        raise FastrInputError("group triggers must contain finite numbers")
    triggers = triggers.astype(np.float64, copy=False)
    if not np.all(np.isfinite(triggers)) or triggers[0] < 0.0:
        raise FastrInputError("group triggers must contain finite numbers")
    if np.any(np.diff(triggers) <= 0.0):
        raise FastrInputError("group triggers must be strictly increasing")
    return triggers


def validate_fastr_parameters(
    *,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
) -> None:
    validate_interpolation_factor(interpolation_factor)
    if not isinstance(neighbor_count, int) or neighbor_count < 2:
        raise FastrInputError("neighbor count must be an integer of at least two")
    if neighbor_count % 2:
        raise FastrInputError("neighbor count must be even")
    if not isinstance(search_radius_samples, int) or search_radius_samples < 0:
        raise FastrInputError("search radius must be a nonnegative integer")


def validate_interpolation_factor(value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise FastrInputError("interpolation factor must be a positive integer")


def validate_sampling_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("sampling rate must be a finite positive number")
    sampling_rate = float(value)
    if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise FastrInputError("sampling rate must be a finite positive number")
    return sampling_rate


def validate_positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError(f"{name} must be a finite positive number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise FastrInputError(f"{name} must be a finite positive number")
    return numeric_value


def validate_nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError(f"{name} must be a finite nonnegative number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise FastrInputError(f"{name} must be a finite nonnegative number")
    return numeric_value


def validate_unit_interval(value: object, *, name: str) -> float:
    numeric_value = validate_positive_finite(value, name=name)
    if numeric_value > 1.0:
        raise FastrInputError(f"{name} must be less than or equal to 1")
    return numeric_value


def validate_excluded_channels(
    excluded_channels: Sequence[int],
    channel_count: int,
) -> frozenset[int]:
    if isinstance(excluded_channels, str) or not isinstance(
        excluded_channels, Sequence
    ):
        raise FastrInputError("excluded channels must be a sequence of indices")
    excluded = frozenset(excluded_channels)
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, Integral)
        or not 0 <= channel < channel_count
        for channel in excluded
    ):
        raise FastrInputError("excluded channels must be valid channel indices")
    if len(excluded) == channel_count:
        raise FastrInputError(
            "excluded channels must leave at least one channel to correct"
        )
    return excluded


def validate_basis_rank(rank: int, group_count: int) -> None:
    if not isinstance(rank, int) or rank < 1:
        raise FastrInputError("basis rank must be a positive integer")
    if rank > group_count:
        raise FastrInputError(
            "basis rank cannot exceed the number of acquisition groups"
        )
