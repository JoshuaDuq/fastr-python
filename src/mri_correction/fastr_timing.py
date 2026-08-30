"""BIDS timing loading and acquisition-group trigger construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .fastr_types import FastrInputError
from .fastr_validation import validate_sampling_rate

_GRID_TOLERANCE_SAMPLES = 1e-6
_JITTER_TOLERANCE_FRACTION = 0.01
_CLOCK_TICK_SAMPLES = 1


@dataclass(frozen=True)
class FmriAcquisitionTiming:
    """Validated fMRI timing metadata used to derive acquisition groups."""

    repetition_time_seconds: float
    slice_timing_seconds: tuple[float, ...]
    multiband_acceleration_factor: int

    def __post_init__(self) -> None:
        repetition_time = _validate_repetition_time(self.repetition_time_seconds)
        slice_timing = _validate_slice_timing(
            self.slice_timing_seconds,
            repetition_time,
        )
        multiband_factor = _validate_multiband_factor(
            self.multiband_acceleration_factor
        )
        _validate_grouping(slice_timing, multiband_factor)

        object.__setattr__(self, "repetition_time_seconds", repetition_time)
        object.__setattr__(self, "slice_timing_seconds", slice_timing)
        object.__setattr__(
            self,
            "multiband_acceleration_factor",
            multiband_factor,
        )

    @property
    def group_offsets_seconds(self) -> tuple[float, ...]:
        return tuple(sorted(set(self.slice_timing_seconds)))

    @property
    def groups_per_volume(self) -> int:
        return len(self.group_offsets_seconds)


def load_bids_fmri_timing(path: str | Path) -> FmriAcquisitionTiming:
    metadata_path = _coerce_path(path)
    metadata = _read_json_object(metadata_path)
    repetition_time, slice_timing, multiband_factor = _extract_timing_fields(
        metadata
    )
    return FmriAcquisitionTiming(
        repetition_time_seconds=repetition_time,
        slice_timing_seconds=slice_timing,
        multiband_acceleration_factor=multiband_factor,
    )


def make_group_trigger_samples(
    volume_starts: object,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
) -> np.ndarray:
    starts = _validate_volume_starts(volume_starts)
    rate = validate_sampling_rate(sampling_rate)
    if not isinstance(timing, FmriAcquisitionTiming):
        raise FastrInputError("timing must be an FmriAcquisitionTiming instance")

    samples_per_volume = timing.repetition_time_seconds * rate
    rounded_samples_per_volume = round(samples_per_volume)
    if not math.isclose(
        samples_per_volume,
        rounded_samples_per_volume,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise FastrInputError(
            "repetition time times sampling rate must be an integer number "
            "of samples"
        )

    _validate_contiguous_starts(starts, rounded_samples_per_volume)
    return _make_fractional_group_starts(
        starts,
        rate,
        timing.group_offsets_seconds,
        rounded_samples_per_volume,
    )


def repair_volume_starts(
    volume_starts: object,
    *,
    samples_per_volume: int,
    expected_volume_count: int,
) -> np.ndarray:
    """Fill uniquely located interior volume markers.

    Missing boundary markers cannot be inferred from a marker series, so a
    declared count that cannot be reached through interior gaps is rejected.
    """
    starts = _validate_volume_starts(volume_starts)
    period = _validate_positive_integer(samples_per_volume, "samples per volume")
    expected = _validate_positive_integer(
        expected_volume_count,
        "expected volume count",
    )

    repaired = [int(starts[0])]
    for left, right in pairwise(starts):
        interval = int(right - left)
        multiple = round(interval / period)
        deviation = abs(interval - multiple * period)
        if multiple < 1 or deviation > _CLOCK_TICK_SAMPLES:
            raise FastrInputError(
                "volume marker interval is not an integer multiple of the "
                "repetition time"
            )
        repaired.extend(
            int(left + step * period) for step in range(1, multiple)
        )
        repaired.append(int(right))

    if len(repaired) != expected:
        raise FastrInputError(
            "repaired markers do not match the expected volume count; "
            "missing boundary markers cannot be inferred"
        )
    return np.asarray(repaired, dtype=np.int64)


def _validate_repetition_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("repetition time must be a finite positive number")
    repetition_time = float(value)
    if not math.isfinite(repetition_time) or repetition_time <= 0.0:
        raise FastrInputError("repetition time must be a finite positive number")
    return repetition_time


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise FastrInputError(f"{name} must be a positive integer")
    return int(value)


def _validate_slice_timing(
    values: object,
    repetition_time: float,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise FastrInputError("slice timing must be a nonempty tuple")

    slice_timing: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise FastrInputError("slice timing values must be finite numbers")
        offset = float(value)
        if not math.isfinite(offset) or not 0.0 <= offset < repetition_time:
            raise FastrInputError(
                "slice timing values must be finite, nonnegative, and less "
                "than the repetition time"
            )
        slice_timing.append(offset)
    return tuple(slice_timing)


def _validate_multiband_factor(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise FastrInputError(
            "multiband acceleration factor must be a positive integer"
        )
    return int(value)


def _validate_grouping(
    slice_timing: tuple[float, ...],
    multiband_factor: int,
) -> None:
    if len(slice_timing) % multiband_factor:
        raise FastrInputError(
            "slice count must be divisible by the multiband acceleration factor"
        )
    group_counts = {
        offset: slice_timing.count(offset) for offset in set(slice_timing)
    }
    if any(count != multiband_factor for count in group_counts.values()):
        raise FastrInputError(
            "each unique slice group time must occur exactly the multiband "
            "acceleration factor times"
        )


def _coerce_path(path: str | Path) -> Path:
    try:
        return Path(path)
    except TypeError as error:
        raise FastrInputError("fMRI metadata path must be a string or Path") from error


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FastrInputError(f"could not read fMRI metadata: {path}") from error

    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as error:
        raise FastrInputError("fMRI metadata must contain valid JSON") from error

    if not isinstance(metadata, dict):
        error = TypeError("top-level JSON value is not an object")
        raise FastrInputError("fMRI metadata must be a JSON object") from error
    return metadata


def _extract_timing_fields(
    metadata: dict[str, object],
) -> tuple[float, tuple[float, ...], int]:
    required_fields = (
        "RepetitionTime",
        "SliceTiming",
        "MultibandAccelerationFactor",
    )
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise FastrInputError(f"missing required field(s): {missing}")

    repetition_time = metadata["RepetitionTime"]
    slice_timing = metadata["SliceTiming"]
    multiband_factor = metadata["MultibandAccelerationFactor"]
    type_error = _find_field_type_error(
        repetition_time,
        slice_timing,
        multiband_factor,
    )
    if type_error is not None:
        raise FastrInputError(
            "fMRI timing fields have invalid JSON types"
        ) from type_error

    return (
        float(repetition_time),
        tuple(float(value) for value in slice_timing),
        int(multiband_factor),
    )


def _find_field_type_error(
    repetition_time: object,
    slice_timing: object,
    multiband_factor: object,
) -> TypeError | None:
    if isinstance(repetition_time, bool) or not isinstance(repetition_time, Real):
        return TypeError("RepetitionTime must be a JSON number")
    if not isinstance(slice_timing, list):
        return TypeError("SliceTiming must be a JSON array")
    invalid_slice_value = any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in slice_timing
    )
    if invalid_slice_value:
        return TypeError("SliceTiming must contain only JSON numbers")
    if isinstance(multiband_factor, bool) or not isinstance(multiband_factor, Integral):
        return TypeError("MultibandAccelerationFactor must be a JSON integer")
    return None


def _validate_volume_starts(volume_starts: object) -> np.ndarray:
    starts = np.asarray(volume_starts)
    if starts.ndim != 1:
        raise FastrInputError("volume starts must be one-dimensional")
    if starts.size == 0:
        raise FastrInputError("volume starts must be nonempty")
    if np.issubdtype(starts.dtype, np.bool_):
        raise FastrInputError("volume starts must not be boolean")
    if not np.issubdtype(starts.dtype, np.integer):
        raise FastrInputError("volume starts must contain integers")

    differences = np.diff(starts.astype(np.int64, copy=False))
    if np.any(differences <= 0):
        raise FastrInputError("volume starts must be strictly increasing")
    return starts.astype(np.int64, copy=False)


def _validate_contiguous_starts(
    starts: np.ndarray,
    samples_per_volume: int,
) -> None:
    """Reject spacing that is not one repetition time, aside from one clock tick.

    A single native sample (0.2 ms at 5 kHz) is within FASTR's alignment search
    and is accepted. Larger jitter and gaps still fail: a scanner break and a
    missing marker cannot be told apart from the marker series alone, so they
    must be declared rather than corrected across.
    """
    if starts.size == 1:
        return
    intervals = np.diff(starts)
    offending = np.flatnonzero(
        np.abs(intervals - samples_per_volume) > _CLOCK_TICK_SAMPLES
    )
    if not offending.size:
        return

    index = int(offending[0])
    deviation = int(intervals[index]) - samples_per_volume
    where = f"between volume markers {index + 1} and {index + 2}"
    tolerance = max(1, round(_JITTER_TOLERANCE_FRACTION * samples_per_volume))
    if abs(deviation) <= tolerance:
        raise FastrInputError(
            f"volume marker timing jitter {where}: {deviation:+d} samples off "
            f"the {samples_per_volume}-sample repetition time. Declare the "
            f"intended acquisition timing rather than correcting across it."
        )
    raise FastrInputError(
        f"acquisition gap {where}: {int(intervals[index])} samples instead of "
        f"{samples_per_volume}, {deviation:+d} off. A scanner break and missing "
        f"markers look identical here, so declare the block boundary explicitly."
    )


def _make_fractional_group_starts(
    starts: np.ndarray,
    sampling_rate: float,
    group_offsets: tuple[float, ...],
    samples_per_volume: int,
) -> np.ndarray:
    start_matrix = starts.astype(np.float64)[:, np.newaxis]
    offset_samples = np.asarray(group_offsets, dtype=np.float64) * sampling_rate
    trigger_matrix = start_matrix + offset_samples

    if not np.all(np.isfinite(trigger_matrix)):
        raise FastrInputError("group trigger samples must be finite")
    if np.any(trigger_matrix < start_matrix) or np.any(
        trigger_matrix >= start_matrix + samples_per_volume
    ):
        raise FastrInputError("group trigger samples must remain within each volume")
    return trigger_matrix.reshape(-1).astype(np.float64, copy=False)
