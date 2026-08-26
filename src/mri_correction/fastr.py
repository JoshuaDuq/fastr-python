import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np


class FastrInputError(ValueError):
    """Raised when FASTR acquisition metadata or triggers are invalid."""


@dataclass(frozen=True)
class FmriAcquisitionTiming:
    repetition_time_seconds: float
    slice_timing_seconds: tuple[float, ...]
    multiband_acceleration_factor: int

    def __post_init__(self) -> None:
        repetition_time = _validate_repetition_time(
            self.repetition_time_seconds
        )
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
    rate = _validate_sampling_rate(sampling_rate)
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


def _validate_repetition_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("repetition time must be a finite positive number")
    repetition_time = float(value)
    if not math.isfinite(repetition_time) or repetition_time <= 0.0:
        raise FastrInputError("repetition time must be a finite positive number")
    return repetition_time


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


def _validate_sampling_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("sampling rate must be a finite positive number")
    sampling_rate = float(value)
    if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise FastrInputError("sampling rate must be a finite positive number")
    return sampling_rate


def _validate_contiguous_starts(
    starts: np.ndarray,
    samples_per_volume: int,
) -> None:
    if starts.size == 1:
        return
    if not np.all(np.diff(starts) == samples_per_volume):
        raise FastrInputError(
            "volume starts must form one contiguous block with exact repetition "
            "time spacing"
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
