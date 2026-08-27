from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, firls, firwin, sosfiltfilt, upfirdn

_INTERPOLATION_HALF_SPAN = 4
_INTERPOLATION_WINDOW = ("kaiser", 5.0)
_GRID_TOLERANCE_SAMPLES = 1e-6
_ARTIFACT_SLACK_FRACTION = 0.01
_PRE_TRIGGER_FRACTION = 0.03
_JITTER_TOLERANCE_FRACTION = 0.01
_CLOCK_TICK_SAMPLES = 1
_HIGH_PASS_HZ = 70.0
_HIGH_PASS_TRANSITION_HZ = 10.0
_HIGH_PASS_ORDER_FACTOR = 1.2
_RESIDUAL_GATE_K = 8.0
_RESIDUAL_GATE_RATIO = 8.0
_RESIDUAL_GATE_MIN_NEIGHBORS = 2
_RESIDUAL_GATE_MAX_FRACTION = 0.02
_RESIDUAL_GATE_MAINS_HZ = 60.0
_ADAPTIVE_IMPROVE_RATIO = 0.85
_TEMPLATE_MEAN_CHUNK = 128


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


@dataclass(frozen=True, slots=True, eq=False)
class FastrProvenance:
    """Complete record of how each acquisition group was corrected."""

    interpolation_factor: int
    samples_before_trigger: int
    samples_after_trigger: int
    search_radius: int
    neighbor_indices: np.ndarray
    shifts: np.ndarray
    correlations: np.ndarray
    amplitudes: np.ndarray
    skipped_group_indices: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "neighbor_indices",
            "shifts",
            "correlations",
            "amplitudes",
            "skipped_group_indices",
        ):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class FastrCorrection:
    """A corrected recording on the original sample grid and its provenance."""

    data: np.ndarray
    provenance: FastrProvenance


@dataclass(frozen=True, slots=True, eq=False)
class FastrGeometry:
    """Validated acquisition geometry shared by all channel batches."""

    triggers: np.ndarray
    fine_triggers: np.ndarray
    epoch: _ArtifactEpoch
    window: _TemplateWindow
    interpolation_factor: int
    interpolation_taps: np.ndarray
    search_radius: int
    group_indices: np.ndarray
    skipped_group_indices: np.ndarray
    sample_count: int
    excluded_group_indices: np.ndarray
    adapted_group_indices: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "triggers",
            "fine_triggers",
            "interpolation_taps",
            "group_indices",
            "skipped_group_indices",
            "excluded_group_indices",
            "adapted_group_indices",
        ):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class FastrAlignment:
    """Group alignment fitted once and reusable across channel batches."""

    shifts: np.ndarray
    correlations: np.ndarray
    fitted_triggers: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("shifts", "correlations", "fitted_triggers"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


def slice_fastr(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
) -> FastrCorrection:
    """Subtract target-excluding alternating FASTR templates.

    This is the classical alternating slice-trigger variant. For multiband data,
    use :func:`acquisition_group_fastr`, which matches repeated acquisition-time
    slots instead of treating adjacent groups as interchangeable.
    """
    return _run_fastr(
        data,
        group_triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=None,
    )


def acquisition_group_fastr(
    data: npt.ArrayLike,
    volume_starts: npt.ArrayLike,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
    interpolation_factor: int = 10,
    neighbor_count: int = 20,
    search_radius_samples: int = 3,
) -> FastrCorrection:
    """Correct repeated multiband acquisition-time slots with FASTR fitting.

    ``volume_starts`` are zero-based sample positions in ``data``. The group
    triggers are derived from the validated BIDS timing, so the slot-matching
    geometry cannot be accidentally paired with a different acquisition layout.
    """
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    return _run_fastr(
        data,
        triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=timing.groups_per_volume,
    )


def slice_fastr_with_edges(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
) -> FastrCorrection:
    """Correct estimable groups and report boundary groups left untouched.

    The strict :func:`slice_fastr` core rejects incomplete epochs. This explicit
    wrapper leaves groups whose search windows exceed the recording untouched and
    records their original indices in provenance.
    """
    return _run_fastr_with_edges(
        data,
        group_triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=None,
    )


def acquisition_group_fastr_with_edges(
    data: npt.ArrayLike,
    volume_starts: npt.ArrayLike,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
    interpolation_factor: int = 10,
    neighbor_count: int = 20,
    search_radius_samples: int = 3,
) -> FastrCorrection:
    """Correct estimable complete volumes and report skipped boundary volumes."""
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    return _run_fastr_with_edges(
        data,
        triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=timing.groups_per_volume,
    )


def prepare_fastr_geometry(
    group_triggers: npt.ArrayLike,
    *,
    sample_count: int,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
    groups_per_volume: int | None = None,
    allow_edges: bool = False,
) -> FastrGeometry:
    """Validate FASTR geometry before fitting any channel data.

    When ``allow_edges`` is true, incomplete boundary epochs are excluded and
    recorded in the returned geometry. This makes the boundary policy explicit
    for streaming or batch-oriented callers.
    """
    triggers = _validate_group_triggers(group_triggers)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise FastrInputError("sample count must be a positive integer")
    if sample_count < 1:
        raise FastrInputError("sample count must be a positive integer")
    if groups_per_volume is not None and (
        isinstance(groups_per_volume, bool)
        or not isinstance(groups_per_volume, int)
        or groups_per_volume < 1
    ):
        raise FastrInputError("groups per volume must be a positive integer")
    if not isinstance(allow_edges, bool):
        raise FastrInputError("allow_edges must be a boolean")
    return _build_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=allow_edges,
    )


def gate_fastr_geometry(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    reference_channel: npt.ArrayLike,
    *,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
) -> FastrGeometry:
    """Drop first-pass residual outliers from every target's neighbour window.

    Alignment stays the original fit. Only the moving-average membership changes,
    so a motion volume cannot leak its leftover into nearby *clean* templates.
    Outlier volumes keep their original local window so a non-stationary
    gradient is still tracked. Windows that never contained an outlier are left
    untouched. If too few valid neighbours remain for a target, that target
    keeps its original window.
    """
    if not isinstance(geometry, FastrGeometry):
        raise FastrInputError("geometry must be a FastrGeometry instance")
    if not isinstance(alignment, FastrAlignment):
        raise FastrInputError("alignment must be a FastrAlignment instance")
    if alignment.fitted_triggers.shape != geometry.fine_triggers.shape:
        raise FastrInputError("alignment group count does not match the geometry")

    reference = _validate_reference_channel(reference_channel, geometry.sample_count)
    template_filter = _make_template_high_pass(
        template_high_pass_hz,
        sampling_rate=sampling_rate,
    )
    interpolated = _interpolate(
        _template_estimate_signal(reference, template_filter),
        geometry.interpolation_taps,
        geometry.interpolation_factor,
    )
    templates = _make_templates(
        interpolated,
        alignment.fitted_triggers,
        geometry.window,
        geometry.epoch,
    )
    epochs = _extract_epochs(
        interpolated,
        alignment.fitted_triggers,
        geometry.epoch.samples_before,
        geometry.epoch.samples_after,
    )
    residual = _template_residual(epochs, templates)
    scores = _residual_outlier_scores(
        residual,
        geometry=geometry,
        sampling_rate=sampling_rate,
    )
    excluded = _outlier_groups(
        scores,
        geometry.window.stride,
        protected_edge_volumes=1,
    )
    if not np.any(excluded):
        return geometry

    gated_indices = _replace_excluded_neighbors(
        geometry.window.indices,
        excluded,
        stride=geometry.window.stride,
    )
    if np.array_equal(gated_indices, geometry.window.indices):
        return geometry

    return replace(
        geometry,
        window=replace(
            geometry.window,
            indices=gated_indices,
            contains_target=False,
            summed_contiguous=False,
        ),
        excluded_group_indices=geometry.group_indices[np.flatnonzero(excluded)],
    )


def adapt_fastr_geometry(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    reference_channel: npt.ArrayLike,
    *,
    local_neighbor_count: int = 20,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
) -> FastrGeometry:
    """Shrink the neighbour window only where a wide template fits worse.

    Alignment stays the original fit. Each target is scored with the configured
    wide window and with a shorter local window of the same slot. Volumes whose
    local template cuts leftover by ``_ADAPTIVE_IMPROVE_RATIO`` keep the short
    window so a non-stationary gradient can be tracked; the rest keep N wide
    so transfer-gain inflation does not return on clean data.
    """
    if not isinstance(geometry, FastrGeometry):
        raise FastrInputError("geometry must be a FastrGeometry instance")
    if not isinstance(alignment, FastrAlignment):
        raise FastrInputError("alignment must be a FastrAlignment instance")
    if alignment.fitted_triggers.shape != geometry.fine_triggers.shape:
        raise FastrInputError("alignment group count does not match the geometry")
    if (
        isinstance(local_neighbor_count, bool)
        or not isinstance(local_neighbor_count, int)
        or local_neighbor_count < 2
        or local_neighbor_count % 2
    ):
        raise FastrInputError("local neighbour count must be an even integer of at least two")

    wide_count = geometry.window.indices.shape[1]
    if local_neighbor_count >= wide_count:
        return geometry

    reference = _validate_reference_channel(reference_channel, geometry.sample_count)
    template_filter = _make_template_high_pass(
        template_high_pass_hz,
        sampling_rate=sampling_rate,
    )
    interpolated = _interpolate(
        _template_estimate_signal(reference, template_filter),
        geometry.interpolation_taps,
        geometry.interpolation_factor,
    )
    epochs = _extract_epochs(
        interpolated,
        alignment.fitted_triggers,
        geometry.epoch.samples_before,
        geometry.epoch.samples_after,
    )
    wide_templates = _make_templates(
        interpolated,
        alignment.fitted_triggers,
        geometry.window,
        geometry.epoch,
    )
    local_indices = _nearest_slot_neighbors(
        epochs.shape[0],
        geometry.window.stride,
        local_neighbor_count,
    )
    local_templates = _mean_selected_epochs(epochs, local_indices)
    wide_scores = np.mean(_template_residual(epochs, wide_templates) ** 2, axis=1)
    local_scores = np.mean(_template_residual(epochs, local_templates) ** 2, axis=1)
    shrink = _volumes_helped_by_local_window(
        wide_scores,
        local_scores,
        geometry.window.stride,
    )
    edge_window = (geometry.window.run_starts == geometry.window.run_starts.min()) | (
        geometry.window.run_starts == geometry.window.run_starts.max()
    )
    shrink &= ~edge_window
    if not np.any(shrink):
        return geometry

    merged = np.array(geometry.window.indices, copy=True, dtype=np.int64)
    for target in np.flatnonzero(shrink):
        row = np.full(wide_count, -1, dtype=np.int64)
        row[:local_neighbor_count] = local_indices[target]
        merged[target] = row

    return replace(
        geometry,
        window=replace(
            geometry.window,
            indices=merged,
            contains_target=False,
            summed_contiguous=False,
        ),
        adapted_group_indices=geometry.group_indices[np.flatnonzero(shrink)],
    )


def fit_fastr_alignment(
    reference_channel: npt.ArrayLike,
    geometry: FastrGeometry,
    *,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
) -> FastrAlignment:
    """Fit acquisition-group alignment from one reference channel.

    When ``template_high_pass_hz`` is set, shifts are estimated on the same
    high-passed copy used to build the moving-average template (Niazy et al.
    2005, stage 2). Leaving it unset aligns the unfiltered reference.
    """
    reference = _validate_reference_channel(reference_channel, geometry.sample_count)
    template_filter = _make_template_high_pass(
        template_high_pass_hz,
        sampling_rate=sampling_rate,
    )
    alignment_signal = _interpolate(
        _template_estimate_signal(reference, template_filter),
        geometry.interpolation_taps,
        geometry.interpolation_factor,
    )
    shifts, correlations = _fit_group_shifts(
        alignment_signal,
        geometry.fine_triggers,
        geometry.window,
        geometry.epoch,
        geometry.search_radius,
    )
    fitted_triggers = geometry.fine_triggers + shifts
    _validate_fitted_triggers(fitted_triggers)
    return FastrAlignment(
        shifts=shifts,
        correlations=correlations,
        fitted_triggers=fitted_triggers,
    )


def apply_fastr_batch(
    data: npt.ArrayLike,
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    *,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
) -> FastrCorrection:
    """Apply one shared alignment to a batch of recording channels.

    When ``template_high_pass_hz`` is set, the moving-average template and the
    least-squares scalar are estimated from a high-passed copy of each channel,
    following Niazy et al. (2005) stage 2, so that segments entering the average
    share a baseline. The fitted artifact is still subtracted from the
    unfiltered channel, so slow content survives correction. Leaving it unset
    estimates from the unfiltered channel, which lets baseline drift bias both
    the template and the scalar.
    """
    if not isinstance(geometry, FastrGeometry):
        raise FastrInputError("geometry must be a FastrGeometry instance")
    if not isinstance(alignment, FastrAlignment):
        raise FastrInputError("alignment must be a FastrAlignment instance")
    recording = _validate_recording(data)
    if recording.shape[1] != geometry.sample_count:
        raise FastrInputError("batch sample count does not match the geometry")
    if alignment.shifts.shape != geometry.fine_triggers.shape:
        raise FastrInputError("alignment group count does not match the geometry")
    if alignment.correlations.shape != alignment.shifts.shape:
        raise FastrInputError("alignment correlations do not match the shifts")
    if alignment.fitted_triggers.shape != alignment.shifts.shape:
        raise FastrInputError("fitted triggers do not match the alignment")

    template_filter = _make_template_high_pass(
        template_high_pass_hz,
        sampling_rate=sampling_rate,
    )

    corrected = recording.astype(np.float64, copy=True)
    amplitudes = np.empty(
        (recording.shape[0], geometry.triggers.size),
        dtype=np.float64,
    )
    for index, channel in enumerate(recording):
        interpolated = _interpolate(
            _template_estimate_signal(channel, template_filter),
            geometry.interpolation_taps,
            geometry.interpolation_factor,
        )
        noise, amplitudes[index] = _fit_channel_noise(
            interpolated,
            alignment.fitted_triggers,
            geometry.window,
            geometry.epoch,
        )
        corrected[index] -= noise[::geometry.interpolation_factor]

    return FastrCorrection(
        data=corrected,
        provenance=FastrProvenance(
            interpolation_factor=geometry.interpolation_factor,
            samples_before_trigger=geometry.epoch.samples_before,
            samples_after_trigger=geometry.epoch.samples_after,
            search_radius=geometry.search_radius,
            neighbor_indices=_map_neighbor_indices(geometry),
            shifts=alignment.shifts,
            correlations=alignment.correlations,
            amplitudes=amplitudes,
            skipped_group_indices=geometry.skipped_group_indices,
        ),
    )


def _run_fastr(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
    groups_per_volume: int | None,
) -> FastrCorrection:
    """Run one explicit FASTR template geometry on validated trigger epochs."""
    recording = _validate_recording(data)
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=recording.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=False,
    )
    alignment = fit_fastr_alignment(recording[0], geometry)
    return apply_fastr_batch(recording, geometry, alignment)


def _run_fastr_with_edges(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
    groups_per_volume: int | None,
) -> FastrCorrection:
    recording = _validate_recording(data)
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=recording.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=True,
    )
    alignment = fit_fastr_alignment(recording[0], geometry)
    return apply_fastr_batch(recording, geometry, alignment)


def _build_fastr_geometry(
    triggers: np.ndarray,
    *,
    sample_count: int,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
    groups_per_volume: int | None,
    allow_edges: bool,
    group_indices: np.ndarray | None = None,
    skipped_group_indices: np.ndarray | None = None,
) -> FastrGeometry:
    _validate_fastr_parameters(
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
    )
    if group_indices is None:
        group_indices = np.arange(triggers.size, dtype=np.int64)
    if skipped_group_indices is None:
        skipped_group_indices = np.empty(0, dtype=np.int64)
    if group_indices.shape != (triggers.size,):
        raise FastrInputError("group indices must match the trigger count")

    fine_triggers = _to_interpolated_grid(triggers, interpolation_factor)
    epoch = _measure_artifact_epoch(
        fine_triggers,
        cover_full_gap=groups_per_volume is not None,
    )
    search_radius = search_radius_samples * interpolation_factor
    sample_count_interpolated = sample_count * interpolation_factor
    bounds_valid = (
        (fine_triggers - epoch.samples_before - search_radius >= 0)
        & (
            fine_triggers + epoch.samples_after + search_radius
            < sample_count_interpolated
        )
    )

    if allow_edges:
        if groups_per_volume is not None:
            if triggers.size % groups_per_volume:
                raise FastrInputError(
                    "the group count must be a whole number of volumes to match "
                    "acquisition slots"
                )
            volume_valid = bounds_valid.reshape(-1, groups_per_volume).all(axis=1)
            bounds_valid = np.repeat(volume_valid, groups_per_volume)
        if not np.any(bounds_valid):
            raise FastrInputError("no complete FASTR artifact epochs remain")
        skipped_group_indices = np.concatenate(
            (skipped_group_indices, group_indices[~bounds_valid])
        )
        group_indices = group_indices[bounds_valid]
        triggers = triggers[bounds_valid]
        fine_triggers = fine_triggers[bounds_valid]
    else:
        _validate_epoch_bounds(
            fine_triggers,
            samples_before=epoch.samples_before + search_radius,
            samples_after=epoch.samples_after + search_radius,
            sample_count=sample_count_interpolated,
        )

    window = _select_template_window(
        triggers.size,
        neighbor_count,
        groups_per_volume,
    )
    return FastrGeometry(
        triggers=triggers,
        fine_triggers=fine_triggers,
        epoch=epoch,
        window=window,
        interpolation_factor=interpolation_factor,
        interpolation_taps=_make_interpolation_filter(interpolation_factor),
        search_radius=search_radius,
        group_indices=group_indices,
        skipped_group_indices=skipped_group_indices,
        sample_count=sample_count,
        excluded_group_indices=np.empty(0, dtype=np.int64),
        adapted_group_indices=np.empty(0, dtype=np.int64),
    )


def residual_obs(
    residual: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    sampling_rate: float,
    excluded_channels: Sequence[int],
    rank: int = 4,
    interpolation_factor: int = 10,
) -> np.ndarray:
    """Subtract the optimal basis set of the residual gradient artifact.

    This is FASTR's third stage, never an implicit part of template subtraction.
    For each corrected channel the basis is the leading `rank` principal
    components of that channel's own high-pass residual epochs. Excluded
    channels are returned untouched, allowing callers to preserve channels that
    are not appropriate for residual artifact subtraction. Adaptive noise
    cancellation, the published fourth stage, is not implemented here.
    """
    recording = _validate_recording(residual)
    triggers = _validate_group_triggers(group_triggers)
    _validate_interpolation_factor(interpolation_factor)
    rate = _validate_sampling_rate(sampling_rate)
    excluded = _validate_excluded_channels(excluded_channels, recording.shape[0])

    fine_triggers = _to_interpolated_grid(triggers, interpolation_factor)
    epoch = _measure_artifact_epoch(fine_triggers, cover_full_gap=False)
    _validate_epoch_bounds(
        fine_triggers,
        samples_before=epoch.samples_before,
        samples_after=epoch.residual_samples_after,
        sample_count=recording.shape[1] * interpolation_factor,
    )
    _validate_basis_rank(rank, triggers.size)

    taps = _make_interpolation_filter(interpolation_factor)
    high_pass = _make_residual_high_pass(rate, interpolation_factor)
    corrected = recording.astype(np.float64, copy=True)
    for index, channel in enumerate(recording):
        if index in excluded:
            continue
        fitted = _fit_residual_basis(
            _interpolate(channel, taps, interpolation_factor),
            fine_triggers,
            epoch,
            high_pass,
            rank,
        )
        corrected[index] -= fitted[::interpolation_factor]
    return corrected



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


@dataclass(frozen=True, slots=True)
class _TemplateWindow:
    """Which epochs form each target's template, and how to sum them cheaply.

    The chosen epochs always form one run of a residue class of the group index,
    so a running total over that class gives every template in one pass.
    """

    indices: np.ndarray
    stride: int
    run_starts: np.ndarray
    run_length: int
    contains_target: bool
    summed_contiguous: bool = True

    def __post_init__(self) -> None:
        for field_name in ("indices", "run_starts"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True)
class _ArtifactEpoch:
    """Extent of one artifact epoch around its trigger, in interpolated samples."""

    samples_before: int
    samples_after: int
    slack: int

    @property
    def length(self) -> int:
        return self.samples_before + self.samples_after + 1

    @property
    def residual_samples_after(self) -> int:
        """FASTR fits residual components over the longest possible artifact."""
        return self.samples_after + self.slack


def _validate_recording(data: npt.ArrayLike) -> np.ndarray:
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


def _validate_reference_channel(
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


def _validate_group_triggers(group_triggers: npt.ArrayLike) -> np.ndarray:
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


def _validate_fastr_parameters(
    *,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
) -> None:
    _validate_interpolation_factor(interpolation_factor)
    if not isinstance(neighbor_count, int) or neighbor_count < 2:
        raise FastrInputError("neighbor count must be an integer of at least two")
    if neighbor_count % 2:
        raise FastrInputError("neighbor count must be even")
    if not isinstance(search_radius_samples, int) or search_radius_samples < 0:
        raise FastrInputError("search radius must be a nonnegative integer")


def _validate_interpolation_factor(value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise FastrInputError("interpolation factor must be a positive integer")


def _to_interpolated_grid(triggers: np.ndarray, factor: int) -> np.ndarray:
    positions = triggers * factor
    rounded = np.rint(positions)
    if not np.allclose(
        positions,
        rounded,
        rtol=0.0,
        atol=_GRID_TOLERANCE_SAMPLES,
    ):
        raise FastrInputError(
            "group triggers must fall on the interpolated sample grid; raise "
            "the interpolation factor rather than rounding acquisition timing"
        )
    return rounded.astype(np.int64)


def _measure_artifact_epoch(
    fine_triggers: np.ndarray,
    *,
    cover_full_gap: bool,
) -> _ArtifactEpoch:
    intervals = np.diff(fine_triggers)
    interval = math.ceil(np.median(intervals))
    before = _round_half_up(interval * _PRE_TRIGGER_FRACTION)
    if cover_full_gap:
        return _ArtifactEpoch(
            samples_before=before,
            samples_after=int(np.max(intervals)),
            slack=0,
        )
    slack = math.ceil((1.0 + _ARTIFACT_SLACK_FRACTION) * interval) - interval
    return _ArtifactEpoch(
        samples_before=before + slack,
        samples_after=_round_half_up((1.0 - _PRE_TRIGGER_FRACTION) * interval),
        slack=slack,
    )


def _round_half_up(value: float) -> int:
    """Round the way FASTR's epoch geometry assumes, unlike Python's round."""
    return math.floor(value + 0.5)


def _validate_epoch_bounds(
    fine_triggers: np.ndarray,
    *,
    samples_before: int,
    samples_after: int,
    sample_count: int,
) -> None:
    first = fine_triggers[0] - samples_before
    last = fine_triggers[-1] + samples_after
    if first < 0 or last >= sample_count:
        raise FastrInputError("the artifact epochs extend beyond the recording")


def _select_template_window(
    group_count: int,
    neighbor_count: int,
    groups_per_volume: int | None,
) -> _TemplateWindow:
    if groups_per_volume is None:
        return _select_alternating_neighbors(group_count, neighbor_count)
    return _select_slice_matched_neighbors(
        group_count,
        neighbor_count,
        groups_per_volume,
    )


def _select_slice_matched_neighbors(
    group_count: int,
    neighbor_count: int,
    groups_per_volume: int,
) -> _TemplateWindow:
    """Take the same acquisition-time slot in nearby volumes."""
    if not isinstance(groups_per_volume, int) or groups_per_volume < 1:
        raise FastrInputError("groups per volume must be a positive integer")
    if group_count % groups_per_volume:
        raise FastrInputError(
            "the group count must be a whole number of volumes to match "
            "acquisition slots"
        )
    volume_count = group_count // groups_per_volume
    if volume_count < neighbor_count + 1:
        raise FastrInputError(
            "not enough volumes to match acquisition slots for the requested "
            "neighbor count"
        )

    targets = np.arange(group_count, dtype=np.int64)
    run_starts = np.clip(
        targets // groups_per_volume - neighbor_count // 2,
        0,
        volume_count - neighbor_count - 1,
    )
    steps = np.arange(neighbor_count + 1, dtype=np.int64)
    members = (targets % groups_per_volume)[:, np.newaxis] + groups_per_volume * (
        run_starts[:, np.newaxis] + steps
    )
    indices = members[members != targets[:, np.newaxis]].reshape(
        group_count,
        neighbor_count,
    )
    return _TemplateWindow(
        indices=indices,
        stride=groups_per_volume,
        run_starts=run_starts,
        run_length=neighbor_count + 1,
        contains_target=True,
    )


def _select_alternating_neighbors(
    group_count: int,
    neighbor_count: int,
) -> _TemplateWindow:
    """Take the nearest groups of the opposite trigger parity to each target.

    Opposite parity is what keeps a group out of its own template, and shifting
    the window instead of shrinking it keeps every template the same width.
    """
    targets = np.arange(group_count, dtype=np.int64)
    parities = 1 - targets % 2
    available = (group_count - parities + 1) // 2
    if available.min() < neighbor_count:
        raise FastrInputError(
            "not enough acquisition groups of the opposite trigger parity for "
            "the requested neighbor count"
        )

    starts = (targets - parities - 1) // 2 - neighbor_count // 2 + 1
    starts = np.clip(starts, 0, available - neighbor_count)
    steps = np.arange(neighbor_count, dtype=np.int64)
    return _TemplateWindow(
        indices=parities[:, np.newaxis] + 2 * (starts[:, np.newaxis] + steps),
        stride=2,
        run_starts=starts,
        run_length=neighbor_count,
        contains_target=False,
    )


def _make_template_high_pass(
    cutoff_hz: float | None,
    *,
    sampling_rate: float | None,
) -> np.ndarray | None:
    """Build the stage-2 template high-pass, or None when it is disabled.

    The filter is applied at the input rate rather than on the interpolated
    grid, where a 1 Hz Butterworth is poorly conditioned.
    """
    if cutoff_hz is None:
        return None
    if isinstance(cutoff_hz, bool) or not isinstance(cutoff_hz, Real):
        raise FastrInputError("template high-pass cutoff must be a finite number")
    cutoff = float(cutoff_hz)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise FastrInputError("template high-pass cutoff must be a finite number")
    if cutoff == 0.0:
        return None
    rate = _validate_sampling_rate(sampling_rate)
    if cutoff >= 0.5 * rate:
        raise FastrInputError(
            "template high-pass cutoff must stay below the input Nyquist frequency"
        )
    return butter(2, cutoff, btype="high", fs=rate, output="sos")


def _template_estimate_signal(
    channel: np.ndarray,
    template_filter: np.ndarray | None,
) -> np.ndarray:
    if template_filter is None:
        return channel
    return sosfiltfilt(template_filter, channel)


def _make_interpolation_filter(factor: int) -> np.ndarray:
    """Build the band-limited filter FASTR's MATLAB interp() call implies."""
    if factor == 1:
        return np.ones(1)
    taps = firwin(
        2 * factor * _INTERPOLATION_HALF_SPAN + 1,
        1.0 / factor,
        window=_INTERPOLATION_WINDOW,
    )
    for phase in range(factor):
        branch = taps[phase::factor]
        branch /= branch.sum()
    return taps


def _interpolate(channel: np.ndarray, taps: np.ndarray, factor: int) -> np.ndarray:
    delay = (taps.size - 1) // 2
    upsampled = upfirdn(taps, channel.astype(np.float64, copy=False), up=factor)
    return upsampled[delay : delay + channel.size * factor]


def _extract_epochs(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    samples_before: int,
    samples_after: int,
) -> np.ndarray:
    offsets = np.arange(-samples_before, samples_after + 1)
    return signal[fine_triggers[:, np.newaxis] + offsets]


def _place_epochs(
    sample_count: int,
    starts: np.ndarray,
    epochs: np.ndarray,
) -> np.ndarray:
    """Lay fitted epochs out, each owning the grid up to the next epoch start."""
    placed = np.zeros(sample_count)
    length = epochs.shape[1]
    ends = np.append(starts[1:], starts[-1] + length)
    for index, start in enumerate(starts):
        written = min(ends[index], start + length) - start
        placed[start : start + written] = epochs[index, :written]
    return placed


def _make_templates(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
) -> np.ndarray:
    """Average each target's chosen epochs, summing each residue class once."""
    epochs = _extract_epochs(
        signal,
        fine_triggers,
        epoch.samples_before,
        epoch.samples_after,
    )
    if not window.summed_contiguous:
        return _mean_selected_epochs(epochs, window.indices)
    neighbor_count = window.indices.shape[1]
    templates = np.empty_like(epochs)
    for residue in range(window.stride):
        class_epochs = epochs[residue :: window.stride]
        totals = np.zeros((class_epochs.shape[0] + 1, epoch.length))
        np.cumsum(class_epochs, axis=0, out=totals[1:])
        targets = np.flatnonzero(window.indices[:, 0] % window.stride == residue)
        starts = window.run_starts[targets]
        summed = totals[starts + window.run_length] - totals[starts]
        if window.contains_target:
            summed = summed - epochs[targets]
        templates[targets] = summed / neighbor_count
    return templates


def _mean_selected_epochs(epochs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Average arbitrary neighbour sets without materialising the full gather."""
    templates = np.empty_like(epochs)
    for start in range(0, indices.shape[0], _TEMPLATE_MEAN_CHUNK):
        stop = min(start + _TEMPLATE_MEAN_CHUNK, indices.shape[0])
        chosen = indices[start:stop]
        selected = epochs[np.clip(chosen, 0, epochs.shape[0] - 1)]
        valid = chosen >= 0
        selected = np.where(valid[..., np.newaxis], selected, 0.0)
        counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        templates[start:stop] = selected.sum(axis=1) / counts
    return templates


def _template_residual(epochs: np.ndarray, templates: np.ndarray) -> np.ndarray:
    energies = np.sum(templates**2, axis=1)
    amplitudes = np.divide(
        np.sum(epochs * templates, axis=1),
        energies,
        out=np.ones(epochs.shape[0]),
        where=energies > 0.0,
    )
    return epochs - amplitudes[:, np.newaxis] * templates


def _residual_outlier_scores(
    residual: np.ndarray,
    *,
    geometry: FastrGeometry,
    sampling_rate: float | None,
) -> np.ndarray:
    """Score leftover gradient, not leftover EEG.

    Broadband epoch energy flags high-amplitude EEG as often as motion. Slice-
    harmonic power of the first-pass residual is the quantity the cohort QC
    actually measures. Without a sampling rate the harmonic bins are unknown,
    so the score falls back to mean-square residual.
    """
    if sampling_rate is None:
        return np.mean(residual**2, axis=1)
    fine_rate = sampling_rate * geometry.interpolation_factor
    intervals = np.diff(geometry.fine_triggers.astype(np.float64))
    if intervals.size == 0:
        return np.mean(residual**2, axis=1)
    slice_rate = fine_rate / float(np.median(intervals))
    return _slice_harmonic_energy(residual, fine_rate, slice_rate)


def _slice_harmonic_energy(
    residual: np.ndarray,
    sampling_rate: float,
    slice_rate: float,
) -> np.ndarray:
    length = residual.shape[1]
    freqs = np.fft.rfftfreq(length, d=1.0 / sampling_rate)
    window = np.hanning(length)
    power = np.abs(np.fft.rfft(residual * window, axis=1)) ** 2
    resolution = float(freqs[1]) if freqs.size > 1 else slice_rate
    half_width = max(2.0 * resolution, 0.1 * slice_rate)
    nyquist = 0.5 * sampling_rate
    mask = np.zeros(freqs.size, dtype=bool)
    order = 1
    while True:
        frequency = slice_rate * order
        if frequency >= nyquist:
            break
        if abs(frequency - _RESIDUAL_GATE_MAINS_HZ) > 1.0:
            mask |= np.abs(freqs - frequency) <= half_width
        order += 1
    if not np.any(mask):
        return np.mean(residual**2, axis=1)
    return power[:, mask].sum(axis=1)


def _outlier_groups(
    residual_energy: np.ndarray,
    stride: int,
    *,
    protected_edge_volumes: int = 1,
) -> np.ndarray:
    """Flag groups whose volume-level residual is a robust outlier.

    The first and last volumes are left eligible as neighbours. Their epochs
    overlap a missing next group, so residual energy is biased at the edges
    even on a stationary artifact.
    """
    energy = np.asarray(residual_energy, dtype=np.float64)
    if energy.ndim != 1 or energy.size == 0:
        raise FastrInputError("residual energy must be a nonempty vector")
    if not isinstance(protected_edge_volumes, int) or protected_edge_volumes < 0:
        raise FastrInputError("protected edge volumes must be a nonnegative integer")
    if stride > 1 and energy.size % stride == 0:
        volume_scores = energy.reshape(-1, stride).mean(axis=1)
        volume_flags = _robust_outliers(volume_scores)
        if protected_edge_volumes:
            volume_flags[:protected_edge_volumes] = False
            volume_flags[-protected_edge_volumes:] = False
        volume_flags = _cap_outliers(volume_flags, volume_scores)
        return np.repeat(volume_flags, stride)
    flags = _robust_outliers(energy)
    if protected_edge_volumes:
        flags[:protected_edge_volumes] = False
        flags[-protected_edge_volumes:] = False
    return _cap_outliers(flags, energy)


def _nearest_slot_neighbors(
    group_count: int,
    stride: int,
    neighbor_count: int,
) -> np.ndarray:
    """Nearest same-slot groups, excluding the target, padded with -1."""
    indices = np.full((group_count, neighbor_count), -1, dtype=np.int64)
    for residue in range(stride):
        class_indices = np.arange(residue, group_count, stride, dtype=np.int64)
        for position, target in enumerate(class_indices):
            others = np.delete(class_indices, position)
            if others.size == 0:
                continue
            order = np.argsort(np.abs(others - target), kind="stable")
            take = min(neighbor_count, others.size)
            indices[target, :take] = others[order[:take]]
    return indices


def _volumes_helped_by_local_window(
    wide_scores: np.ndarray,
    local_scores: np.ndarray,
    stride: int,
    *,
    protected_edge_volumes: int = 2,
) -> np.ndarray:
    """True for groups whose volume leftover falls enough with a short window."""
    if stride > 1 and wide_scores.size % stride == 0:
        wide_volume = wide_scores.reshape(-1, stride).mean(axis=1)
        local_volume = local_scores.reshape(-1, stride).mean(axis=1)
        helped = local_volume < _ADAPTIVE_IMPROVE_RATIO * wide_volume
        if protected_edge_volumes:
            helped[:protected_edge_volumes] = False
            helped[-protected_edge_volumes:] = False
        return np.repeat(helped, stride)
    helped = local_scores < _ADAPTIVE_IMPROVE_RATIO * wide_scores
    if protected_edge_volumes:
        helped[:protected_edge_volumes] = False
        helped[-protected_edge_volumes:] = False
    return helped


def _cap_outliers(flags: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Keep only the strongest outliers so EEG variance cannot empty the window."""
    flagged = np.flatnonzero(flags)
    maximum = max(1, int(np.floor(scores.size * _RESIDUAL_GATE_MAX_FRACTION)))
    if flagged.size <= maximum:
        return flags
    strongest = flagged[np.argsort(scores[flagged], kind="stable")[::-1][:maximum]]
    capped = np.zeros(flags.shape, dtype=bool)
    capped[strongest] = True
    return capped


def _robust_outliers(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        above = values[values > median]
        if above.size == 0:
            return np.zeros(values.shape, dtype=bool)
        scale = float(np.median(above))
        if scale <= 0.0:
            return np.zeros(values.shape, dtype=bool)
        return values > scale * _RESIDUAL_GATE_RATIO
    threshold = max(
        median * _RESIDUAL_GATE_RATIO,
        median + _RESIDUAL_GATE_K * 1.4826 * mad,
    )
    return values > threshold


def _replace_excluded_neighbors(
    indices: np.ndarray,
    excluded: np.ndarray,
    *,
    stride: int,
) -> np.ndarray:
    """Rebuild each window from the nearest non-outlier members of the same slot."""
    gated = np.array(indices, copy=True, dtype=np.int64)
    excluded = np.asarray(excluded, dtype=bool)
    n_groups, n_neighbors = gated.shape
    if excluded.shape != (n_groups,):
        raise FastrInputError("excluded groups must match the template window")
    for residue in range(stride):
        class_indices = np.arange(residue, n_groups, stride, dtype=np.int64)
        usable_all = class_indices[~excluded[class_indices]]
        for target in class_indices:
            if excluded[target]:
                continue
            members = gated[target]
            valid_members = members[members >= 0]
            if valid_members.size > 0 and not np.any(excluded[valid_members]):
                continue
            usable = usable_all[usable_all != target]
            if usable.size < _RESIDUAL_GATE_MIN_NEIGHBORS:
                continue
            kept = valid_members[~excluded[valid_members]]
            need = n_neighbors - kept.size
            if need <= 0:
                row = np.full(n_neighbors, -1, dtype=np.int64)
                row[:n_neighbors] = kept[:n_neighbors]
                gated[target] = row
                continue
            used = set(kept.tolist())
            extras = [
                int(candidate)
                for candidate in usable[np.argsort(np.abs(usable - target), kind="stable")]
                if int(candidate) not in used
            ]
            chosen = np.concatenate(
                (kept, np.asarray(extras[:need], dtype=np.int64))
            ) if extras else kept
            take = min(n_neighbors, chosen.size)
            if take < _RESIDUAL_GATE_MIN_NEIGHBORS:
                continue
            row = np.full(n_neighbors, -1, dtype=np.int64)
            row[:take] = chosen[:take]
            gated[target] = row
    return gated


def _map_neighbor_indices(geometry: FastrGeometry) -> np.ndarray:
    raw_indices = geometry.window.indices
    if raw_indices.size == 0:
        return raw_indices
    safe = np.clip(raw_indices, 0, geometry.group_indices.size - 1)
    return np.where(raw_indices >= 0, geometry.group_indices[safe], -1)


def _fit_group_shifts(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    templates = _make_templates(signal, fine_triggers, window, epoch)
    shifts = np.empty(fine_triggers.size, dtype=np.int64)
    correlations = np.empty(fine_triggers.size, dtype=np.float64)
    for index, trigger in enumerate(fine_triggers):
        searched = signal[
            trigger - epoch.samples_before - search_radius : trigger
            + epoch.samples_after
            + search_radius
            + 1
        ]
        scores = _correlate(
            sliding_window_view(searched, epoch.length),
            templates[index],
        )
        best = int(np.argmax(scores))
        shifts[index] = best - search_radius
        correlations[index] = scores[best]
    return shifts, correlations


def _correlate(candidates: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation of every candidate epoch against one template."""
    centered = candidates - candidates.mean(axis=-1, keepdims=True)
    reference = template - template.mean()
    norms = np.sqrt(np.sum(centered**2, axis=-1) * np.sum(reference**2))
    return np.divide(
        centered @ reference,
        norms,
        out=np.zeros(candidates.shape[0]),
        where=norms > 0.0,
    )


def _validate_fitted_triggers(fitted_triggers: np.ndarray) -> None:
    if np.any(np.diff(fitted_triggers) <= 0):
        raise FastrInputError(
            "the fitted artifact epochs overlap; sub-sample alignment did not "
            "preserve the acquisition group order"
        )


def _fit_channel_noise(
    signal: np.ndarray,
    fitted_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
) -> tuple[np.ndarray, np.ndarray]:
    templates = _make_templates(signal, fitted_triggers, window, epoch)
    epochs = _extract_epochs(
        signal,
        fitted_triggers,
        epoch.samples_before,
        epoch.samples_after,
    )
    energies = np.sum(templates**2, axis=1)
    amplitudes = np.divide(
        np.sum(epochs * templates, axis=1),
        energies,
        out=np.ones(fitted_triggers.size),
        where=energies > 0.0,
    )

    noise = _place_epochs(
        signal.size,
        fitted_triggers - epoch.samples_before,
        amplitudes[:, np.newaxis] * templates,
    )
    return noise, amplitudes


def _validate_excluded_channels(
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


def _validate_basis_rank(rank: int, group_count: int) -> None:
    if not isinstance(rank, int) or rank < 1:
        raise FastrInputError("basis rank must be a positive integer")
    if rank > group_count:
        raise FastrInputError(
            "basis rank cannot exceed the number of acquisition groups"
        )


def _make_residual_high_pass(sampling_rate: float, factor: int) -> np.ndarray:
    """Build the high-pass FASTR fits its residual basis set on."""
    nyquist = 0.5 * sampling_rate * factor
    stop_band = _HIGH_PASS_HZ - _HIGH_PASS_TRANSITION_HZ
    pass_band = _HIGH_PASS_HZ + _HIGH_PASS_TRANSITION_HZ
    if pass_band >= nyquist:
        raise FastrInputError(
            "the sampling rate is too low for the residual high-pass filter"
        )
    order = _round_half_up(_HIGH_PASS_ORDER_FACTOR * sampling_rate * factor / stop_band)
    return firls(
        order + order % 2 + 1,
        (0.0, stop_band / nyquist, pass_band / nyquist, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )


def _fit_residual_basis(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    epoch: _ArtifactEpoch,
    high_pass: np.ndarray,
    rank: int,
) -> np.ndarray:
    epochs = _extract_epochs(
        filtfilt(high_pass, 1.0, signal),
        fine_triggers,
        epoch.samples_before,
        epoch.residual_samples_after,
    )
    centered = epochs - epochs.mean(axis=1, keepdims=True)
    basis = np.linalg.svd(centered.T, full_matrices=False)[0][:, :rank]
    return _place_epochs(
        signal.size,
        fine_triggers - epoch.samples_before,
        epochs @ basis @ basis.T,
    )
