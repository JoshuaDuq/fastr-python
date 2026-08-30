"""Acquisition timing: declared metadata, and where every group actually fires.

A recording marks either one event per volume or one per excitation. Both
conventions resolve to the same :class:`AcquisitionGeometry`, so nothing
downstream has to know which one the recording used. Neither path infers
acquisition geometry from the EEG waveform: volume markers are expanded using
declared slice timing, and acquisition-group markers are measured where they
were recorded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .fastr_types import FastrInputError
from .fastr_validation import validate_positive_finite, validate_sampling_rate

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


@dataclass(frozen=True, slots=True)
class AcquisitionGeometry:
    """Where every acquisition group fires, and how volumes group them.

    ``group_triggers`` are zero-based input samples, one per acquisition group,
    in recording order; ``volume_starts`` are the subset that begins a volume.
    ``source`` records how the group positions were obtained, because a derived
    position and a recorded one are not the same evidence.
    """

    volume_starts: np.ndarray
    group_triggers: np.ndarray
    repetition_time_seconds: float
    groups_per_volume: int
    group_offsets_seconds: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        for field_name in ("volume_starts", "group_triggers"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)

    @property
    def volume_count(self) -> int:
        return int(self.volume_starts.size)


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


def volume_marker_geometry(
    volume_starts: object,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
) -> AcquisitionGeometry:
    """Expand one marker per volume using declared slice timing.

    A volume marker locates only the volume. Where each acquisition group fires
    inside it comes from the declared ``SliceTiming`` offsets, so the timing has
    to be supplied rather than read off the recording.
    """
    starts = _validate_volume_starts(volume_starts)
    triggers = make_group_trigger_samples(
        starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    return AcquisitionGeometry(
        volume_starts=starts,
        group_triggers=triggers,
        repetition_time_seconds=timing.repetition_time_seconds,
        groups_per_volume=timing.groups_per_volume,
        group_offsets_seconds=timing.group_offsets_seconds,
        source="declared_slice_timing",
    )


def slice_marker_geometry(
    group_markers: object,
    *,
    sampling_rate: float,
    groups_per_volume: int,
    expected_repetition_time_seconds: float | None = None,
) -> AcquisitionGeometry:
    """Measure acquisition groups from one marker per excitation.

    A slice-triggered recording marks every excitation, so the group positions
    are recorded rather than derived and only the number of groups in a volume
    has to be declared: nothing in the marker series itself says where a volume
    begins. The repetition time and the within-volume offsets are then measured
    from the markers, and checked for the periodicity that slot matching needs.

    A wrong ``groups_per_volume`` that still divides the marker count is
    self-consistent -- counting off two volumes' worth of groups measures twice
    the repetition time and offsets that repeat just as well. Only a declared
    ``expected_repetition_time_seconds`` can catch that, so it is checked here
    when the caller supplies one.
    """
    markers = _validate_volume_starts(
        group_markers,
        name="acquisition-group markers",
    )
    rate = validate_sampling_rate(sampling_rate)
    groups = _validate_positive_integer(groups_per_volume, "groups per volume")
    if markers.size % groups:
        raise FastrInputError(
            f"{markers.size} acquisition-group markers do not divide into "
            f"whole {groups}-group volumes; a partial volume cannot be matched "
            f"to an acquisition slot"
        )

    volume_starts = np.ascontiguousarray(markers[::groups])
    if volume_starts.size < 2:
        raise FastrInputError(
            "at least two volumes are required to measure the repetition time "
            "from acquisition-group markers"
        )
    samples_per_volume = round(float(np.median(np.diff(volume_starts))))
    _validate_measured_volume_period(volume_starts, samples_per_volume)
    _validate_expected_repetition_time(
        samples_per_volume,
        expected_repetition_time_seconds,
        sampling_rate=rate,
        groups_per_volume=groups,
    )
    offsets = _measure_group_offsets(
        markers,
        volume_starts,
        groups_per_volume=groups,
        samples_per_volume=samples_per_volume,
    )
    return AcquisitionGeometry(
        volume_starts=volume_starts,
        group_triggers=markers.astype(np.float64),
        repetition_time_seconds=samples_per_volume / rate,
        groups_per_volume=groups,
        group_offsets_seconds=tuple(float(value) / rate for value in offsets),
        source="measured_group_markers",
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


def _validate_measured_volume_period(
    volume_starts: np.ndarray,
    samples_per_volume: int,
) -> None:
    """Reject derived volume starts that are not one repetition time apart.

    Volume boundaries here are counted off in groups, so a single missing or
    extra acquisition-group marker moves every later boundary. Uneven volume
    spacing is that failure, and it cannot be repaired from the marker series.
    """
    intervals = np.diff(volume_starts)
    offending = np.flatnonzero(
        np.abs(intervals - samples_per_volume) > _CLOCK_TICK_SAMPLES
    )
    if not offending.size:
        return
    index = int(offending[0])
    raise FastrInputError(
        f"acquisition-group markers place volumes {index + 1} and {index + 2} "
        f"{int(intervals[index])} samples apart instead of "
        f"{samples_per_volume}. One missing or extra group marker moves every "
        f"later volume boundary, so the series cannot be split into volumes."
    )


def _validate_expected_repetition_time(
    samples_per_volume: int,
    expected_repetition_time_seconds: float | None,
    *,
    sampling_rate: float,
    groups_per_volume: int,
) -> None:
    """Check the measured volume period against the one the caller declared.

    The likely mistake this catches is counting slices where the scanner marks
    excitations, which measures a whole multiple of the repetition time and
    would otherwise look entirely consistent.
    """
    if expected_repetition_time_seconds is None:
        return
    expected = validate_positive_finite(
        expected_repetition_time_seconds,
        name="expected repetition time",
    )
    expected_samples = expected * sampling_rate
    if abs(samples_per_volume - expected_samples) <= _CLOCK_TICK_SAMPLES:
        return
    measured_seconds = samples_per_volume / sampling_rate
    raise FastrInputError(
        f"acquisition-group markers measure a {measured_seconds:.6g} s "
        f"repetition time at {groups_per_volume} groups per volume, against "
        f"the declared {expected:.6g} s. Check whether the markers count "
        f"excitations or slices."
    )


def _measure_group_offsets(
    markers: np.ndarray,
    volume_starts: np.ndarray,
    *,
    groups_per_volume: int,
    samples_per_volume: int,
) -> np.ndarray:
    """Measure each acquisition slot's offset inside its volume.

    Slot matching averages one acquisition time across volumes, so the offsets
    have to repeat. Offsets that drift mean the markers do not describe a single
    repeating acquisition, and averaging them would mix different slots.
    """
    offsets = markers.reshape(-1, groups_per_volume) - volume_starts[:, np.newaxis]
    representative = np.median(offsets, axis=0)
    deviation = np.abs(offsets - representative)
    worst = int(np.argmax(deviation))
    if deviation.flat[worst] > _CLOCK_TICK_SAMPLES:
        volume, slot = divmod(worst, groups_per_volume)
        raise FastrInputError(
            f"acquisition-group marker {slot + 1} of volume {volume + 1} sits "
            f"{int(offsets[volume, slot])} samples into its volume against "
            f"{representative[slot]:.1f} for that slot elsewhere; slot matching "
            f"averages one acquisition time across volumes, so the offsets have "
            f"to repeat"
        )
    if representative[-1] >= samples_per_volume:
        raise FastrInputError(
            "the last acquisition group of a volume starts at or after the next "
            "volume; check that groups_per_volume matches the acquisition"
        )
    return representative


def _validate_volume_starts(
    volume_starts: object,
    *,
    name: str = "volume starts",
) -> np.ndarray:
    starts = np.asarray(volume_starts)
    if starts.ndim != 1:
        raise FastrInputError(f"{name} must be one-dimensional")
    if starts.size == 0:
        raise FastrInputError(f"{name} must be nonempty")
    if np.issubdtype(starts.dtype, np.bool_):
        raise FastrInputError(f"{name} must not be boolean")
    if not np.issubdtype(starts.dtype, np.integer):
        raise FastrInputError(f"{name} must contain integers")

    differences = np.diff(starts.astype(np.int64, copy=False))
    if np.any(differences <= 0):
        raise FastrInputError(f"{name} must be strictly increasing")
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
