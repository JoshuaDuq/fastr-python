"""FASTR acquisition geometry, boundaries, and adaptive template windows."""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Real

import numpy as np
import numpy.typing as npt

from .templates import (
    _extract_epochs,
    _interpolate,
    _make_interpolation_filter,
    _make_template_high_pass,
    _make_templates,
    _mean_selected_epochs,
    _template_estimate_signal,
    _template_residual,
)
from .types import (
    FastrAlignment,
    FastrGeometry,
    FastrInputError,
    _ArtifactEpoch,
    _TemplateWindow,
)
from .validation import (
    validate_fastr_parameters,
    validate_group_triggers,
    validate_nonnegative_finite,
    validate_positive_finite,
    validate_reference_channel,
    validate_unit_interval,
)

_GRID_TOLERANCE_SAMPLES = 1e-6
_ARTIFACT_SLACK_FRACTION = 0.01
_DEFAULT_PRE_TRIGGER_FRACTION = 0.03
_RESIDUAL_GATE_MIN_NEIGHBORS = 2
_MIN_EDGE_ADAPTATION_ENERGY_RATIO = 1e-3


def prepare_fastr_geometry(
    group_triggers: npt.ArrayLike,
    *,
    sample_count: int,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
    groups_per_volume: int | None = None,
    allow_edges: bool = False,
    pre_trigger_fraction: float = _DEFAULT_PRE_TRIGGER_FRACTION,
) -> FastrGeometry:
    """Validate FASTR geometry before fitting any channel data.

    When ``allow_edges`` is true, incomplete boundary epochs are excluded and
    recorded in the returned geometry. This makes the boundary policy explicit
    for streaming or batch-oriented callers.
    """
    triggers = validate_group_triggers(group_triggers)
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
    fraction = _validate_pre_trigger_fraction(pre_trigger_fraction)
    return _build_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=allow_edges,
        pre_trigger_fraction=fraction,
    )


def gate_fastr_geometry(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    reference_channel: np.ndarray,
    *,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
    residual_gate_mad_multiplier: float = 8.0,
    residual_gate_ratio: float = 8.0,
    residual_gate_max_fraction: float = 0.02,
    mains_frequency_hz: float = 60.0,
    mains_exclusion_hz: float = 1.0,
) -> FastrGeometry:
    """Drop first-pass residual outliers from every target's neighbour window.

    Alignment stays the original fit. Only the moving-average membership changes,
    so a motion volume cannot leak its leftover into nearby *clean* templates.
    Outlier volumes keep their original local window so a non-stationary
    gradient is still tracked. Windows that never contained an outlier are left
    untouched. If too few valid neighbours remain for a target, that target
    keeps its original window.
    """
    _validate_geometry_alignment(geometry, alignment)
    mad_multiplier = validate_positive_finite(
        residual_gate_mad_multiplier,
        name="residual gate MAD multiplier",
    )
    ratio = validate_positive_finite(
        residual_gate_ratio,
        name="residual gate ratio",
    )
    max_fraction = validate_unit_interval(
        residual_gate_max_fraction,
        name="residual gate maximum fraction",
    )
    mains_frequency = validate_positive_finite(
        mains_frequency_hz,
        name="mains frequency",
    )
    mains_exclusion = validate_nonnegative_finite(
        mains_exclusion_hz,
        name="mains exclusion",
    )

    reference = validate_reference_channel(reference_channel, geometry.sample_count)
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
        mains_frequency_hz=mains_frequency,
        mains_exclusion_hz=mains_exclusion,
    )
    excluded = _outlier_groups(
        scores,
        geometry.window.stride,
        protected_edge_volumes=1,
        mad_multiplier=mad_multiplier,
        ratio=ratio,
        max_fraction=max_fraction,
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
    reference_channel: np.ndarray,
    *,
    local_neighbor_count: int = 20,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
    adaptive_improvement_ratio: float = 0.85,
) -> FastrGeometry:
    """Shrink the neighbour window only where a wide template fits worse.

    Alignment stays the original fit. Each target is scored with the configured
    wide window and with a shorter local window of the same slot. Volumes whose
    local template cuts leftover by ``adaptive_improvement_ratio`` keep the short
    window so a non-stationary gradient can be tracked; the rest keep N wide
    so transfer-gain inflation does not return on clean data.
    """
    return _adapt_fastr_geometry(
        geometry,
        alignment,
        reference_channel,
        local_neighbor_count=local_neighbor_count,
        template_high_pass_hz=template_high_pass_hz,
        sampling_rate=sampling_rate,
        adaptive_improvement_ratio=adaptive_improvement_ratio,
        protect_edge_windows=True,
    )


def adapt_channel_fastr_geometry(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    channel: np.ndarray,
    *,
    local_neighbor_count: int = 20,
    template_high_pass_hz: float | None = None,
    sampling_rate: float | None = None,
    adaptive_improvement_ratio: float = 0.85,
) -> FastrGeometry:
    """Adapt one channel, including one-sided template windows at run edges."""
    return _adapt_fastr_geometry(
        geometry,
        alignment,
        channel,
        local_neighbor_count=local_neighbor_count,
        template_high_pass_hz=template_high_pass_hz,
        sampling_rate=sampling_rate,
        adaptive_improvement_ratio=adaptive_improvement_ratio,
        protect_edge_windows=False,
    )


def _adapt_fastr_geometry(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    channel: np.ndarray,
    *,
    local_neighbor_count: int,
    template_high_pass_hz: float | None,
    sampling_rate: float | None,
    adaptive_improvement_ratio: float,
    protect_edge_windows: bool,
) -> FastrGeometry:
    _validate_geometry_alignment(geometry, alignment)
    improvement_ratio = validate_unit_interval(
        adaptive_improvement_ratio,
        name="adaptive improvement ratio",
    )
    _validate_local_neighbor_count(local_neighbor_count)

    wide_count = geometry.window.indices.shape[1]
    if local_neighbor_count >= wide_count:
        return geometry

    reference = validate_reference_channel(channel, geometry.sample_count)
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
    local_indices = _local_neighbor_indices(geometry, local_neighbor_count)
    local_templates = _mean_selected_epochs(epochs, local_indices)
    wide_scores = np.mean(_template_residual(epochs, wide_templates) ** 2, axis=1)
    local_scores = np.mean(_template_residual(epochs, local_templates) ** 2, axis=1)
    shrink = _volumes_helped_by_local_window(
        wide_scores,
        local_scores,
        geometry.window.stride,
        improvement_ratio=improvement_ratio,
    )
    excluded_targets = np.isin(
        geometry.group_indices,
        geometry.excluded_group_indices,
    )
    shrink &= ~excluded_targets
    edge_window = (geometry.window.run_starts == geometry.window.run_starts.min()) | (
        geometry.window.run_starts == geometry.window.run_starts.max()
    )
    if protect_edge_windows:
        shrink &= ~edge_window
    else:
        epoch_scores = np.mean(epochs**2, axis=1)
        edge_residual_ratio = _volume_residual_energy_ratio(
            wide_scores,
            epoch_scores,
            geometry.window.stride,
        )
        reliable_edge = edge_residual_ratio > _MIN_EDGE_ADAPTATION_ENERGY_RATIO
        shrink &= ~edge_window | reliable_edge
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


def prepare_local_fastr_geometry(
    geometry: FastrGeometry,
    *,
    local_neighbor_count: int,
) -> FastrGeometry:
    """Use the short same-slot template window for every acquisition group."""
    if not isinstance(geometry, FastrGeometry):
        raise FastrInputError("geometry must be a FastrGeometry instance")
    _validate_local_neighbor_count(local_neighbor_count)
    wide_count = geometry.window.indices.shape[1]
    if local_neighbor_count >= wide_count:
        raise FastrInputError(
            "local neighbour count must be smaller than the configured neighbour count"
        )
    local_indices = _local_neighbor_indices(geometry, local_neighbor_count)
    return replace(
        geometry,
        window=replace(
            geometry.window,
            indices=local_indices,
            contains_target=False,
            summed_contiguous=False,
        ),
        adapted_group_indices=geometry.group_indices,
    )


def _validate_local_neighbor_count(local_neighbor_count: int) -> None:
    if (
        isinstance(local_neighbor_count, bool)
        or not isinstance(local_neighbor_count, int)
        or local_neighbor_count < 2
        or local_neighbor_count % 2
    ):
        raise FastrInputError(
            "local neighbour count must be an even integer of at least two"
        )


def _validate_geometry_alignment(
    geometry: FastrGeometry,
    alignment: FastrAlignment,
) -> None:
    if not isinstance(geometry, FastrGeometry):
        raise FastrInputError("geometry must be a FastrGeometry instance")
    if not isinstance(alignment, FastrAlignment):
        raise FastrInputError("alignment must be a FastrAlignment instance")
    if alignment.fitted_triggers.shape != geometry.fine_triggers.shape:
        raise FastrInputError("alignment group count does not match the geometry")


def _build_fastr_geometry(
    triggers: np.ndarray,
    *,
    sample_count: int,
    interpolation_factor: int,
    neighbor_count: int,
    search_radius_samples: int,
    groups_per_volume: int | None,
    allow_edges: bool,
    pre_trigger_fraction: float,
    group_indices: np.ndarray | None = None,
    skipped_group_indices: np.ndarray | None = None,
) -> FastrGeometry:
    validate_fastr_parameters(
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
        pre_trigger_fraction=pre_trigger_fraction,
    )
    search_radius = search_radius_samples * interpolation_factor
    sample_count_interpolated = sample_count * interpolation_factor
    bounds_valid = (fine_triggers - epoch.samples_before - search_radius >= 0) & (
        fine_triggers + epoch.samples_after + search_radius < sample_count_interpolated
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
        pre_trigger_fraction=pre_trigger_fraction,
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
    pre_trigger_fraction: float = _DEFAULT_PRE_TRIGGER_FRACTION,
) -> _ArtifactEpoch:
    intervals = np.diff(fine_triggers)
    interval = math.ceil(np.median(intervals))
    before = _round_half_up(interval * pre_trigger_fraction)
    if cover_full_gap:
        return _ArtifactEpoch(
            samples_before=before,
            samples_after=int(np.max(intervals)),
            slack=0,
        )
    slack = math.ceil((1.0 + _ARTIFACT_SLACK_FRACTION) * interval) - interval
    return _ArtifactEpoch(
        samples_before=before + slack,
        samples_after=_round_half_up((1.0 - pre_trigger_fraction) * interval),
        slack=slack,
    )


def _validate_pre_trigger_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("pre-trigger fraction must be a finite number")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise FastrInputError("pre-trigger fraction must lie between zero and one")
    return fraction


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


def _residual_outlier_scores(
    residual: np.ndarray,
    *,
    geometry: FastrGeometry,
    sampling_rate: float | None,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
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
    return _slice_harmonic_energy(
        residual,
        fine_rate,
        slice_rate,
        mains_frequency_hz=mains_frequency_hz,
        mains_exclusion_hz=mains_exclusion_hz,
    )


def _slice_harmonic_energy(
    residual: np.ndarray,
    sampling_rate: float,
    slice_rate: float,
    *,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
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
        if abs(frequency - mains_frequency_hz) > mains_exclusion_hz:
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
    mad_multiplier: float,
    ratio: float,
    max_fraction: float,
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
        volume_flags = _robust_outliers(
            volume_scores,
            mad_multiplier=mad_multiplier,
            ratio=ratio,
        )
        if protected_edge_volumes:
            volume_flags[:protected_edge_volumes] = False
            volume_flags[-protected_edge_volumes:] = False
        volume_flags = _cap_outliers(volume_flags, volume_scores, max_fraction)
        return np.repeat(volume_flags, stride)
    flags = _robust_outliers(
        energy,
        mad_multiplier=mad_multiplier,
        ratio=ratio,
    )
    if protected_edge_volumes:
        flags[:protected_edge_volumes] = False
        flags[-protected_edge_volumes:] = False
    return _cap_outliers(flags, energy, max_fraction)


def _nearest_slot_neighbors(
    group_count: int,
    stride: int,
    neighbor_count: int,
    excluded_rows: np.ndarray,
) -> np.ndarray:
    """Nearest clean same-slot groups, excluding the target."""
    indices = np.empty((group_count, neighbor_count), dtype=np.int64)
    excluded = np.zeros(group_count, dtype=bool)
    excluded[excluded_rows] = True
    for residue in range(stride):
        class_indices = np.arange(residue, group_count, stride, dtype=np.int64)
        available = class_indices[~excluded[class_indices]]
        for target in class_indices:
            others = available[available != target]
            if others.size < neighbor_count:
                raise FastrInputError(
                    "too few non-excluded same-slot groups remain for the "
                    "requested local neighbour count"
                )
            order = np.argsort(np.abs(others - target), kind="stable")
            indices[target] = others[order[:neighbor_count]]
    return indices


def _local_neighbor_indices(
    geometry: FastrGeometry,
    neighbor_count: int,
) -> np.ndarray:
    excluded_rows = np.flatnonzero(
        np.isin(geometry.group_indices, geometry.excluded_group_indices)
    )
    return _nearest_slot_neighbors(
        geometry.group_indices.size,
        geometry.window.stride,
        neighbor_count,
        excluded_rows,
    )


def _volumes_helped_by_local_window(
    wide_scores: np.ndarray,
    local_scores: np.ndarray,
    stride: int,
    *,
    protected_edge_volumes: int = 2,
    improvement_ratio: float,
) -> np.ndarray:
    """Mark groups whose volume leftover falls enough with a short window."""
    if stride > 1 and wide_scores.size % stride == 0:
        wide_volume = wide_scores.reshape(-1, stride).mean(axis=1)
        local_volume = local_scores.reshape(-1, stride).mean(axis=1)
        helped = local_volume < improvement_ratio * wide_volume
        if protected_edge_volumes:
            helped[:protected_edge_volumes] = False
            helped[-protected_edge_volumes:] = False
        return np.repeat(helped, stride)
    helped = local_scores < improvement_ratio * wide_scores
    if protected_edge_volumes:
        helped[:protected_edge_volumes] = False
        helped[-protected_edge_volumes:] = False
    return helped


def _volume_residual_energy_ratio(
    residual_scores: np.ndarray,
    epoch_scores: np.ndarray,
    stride: int,
) -> np.ndarray:
    if stride > 1 and residual_scores.size % stride == 0:
        residual = residual_scores.reshape(-1, stride).mean(axis=1)
        epoch = epoch_scores.reshape(-1, stride).mean(axis=1)
        ratio = np.divide(
            residual,
            epoch,
            out=np.zeros(residual.shape),
            where=epoch > 0.0,
        )
        return np.repeat(ratio, stride)
    return np.divide(
        residual_scores,
        epoch_scores,
        out=np.zeros(residual_scores.shape),
        where=epoch_scores > 0.0,
    )


def _cap_outliers(
    flags: np.ndarray,
    scores: np.ndarray,
    max_fraction: float,
) -> np.ndarray:
    """Keep only the strongest outliers so EEG variance cannot empty the window."""
    flagged = np.flatnonzero(flags)
    maximum = max(1, int(np.floor(scores.size * max_fraction)))
    if flagged.size <= maximum:
        return flags
    strongest = flagged[np.argsort(scores[flagged], kind="stable")[::-1][:maximum]]
    capped = np.zeros(flags.shape, dtype=bool)
    capped[strongest] = True
    return capped


def _robust_outliers(
    values: np.ndarray,
    *,
    mad_multiplier: float,
    ratio: float,
) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        above = values[values > median]
        if above.size == 0:
            return np.zeros(values.shape, dtype=bool)
        scale = float(np.median(above))
        if scale <= 0.0:
            return np.zeros(values.shape, dtype=bool)
        return values > scale * ratio
    threshold = max(
        median * ratio,
        median + mad_multiplier * 1.4826 * mad,
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
                for candidate in usable[
                    np.argsort(np.abs(usable - target), kind="stable")
                ]
                if int(candidate) not in used
            ]
            chosen = (
                np.concatenate((kept, np.asarray(extras[:need], dtype=np.int64)))
                if extras
                else kept
            )
            take = min(n_neighbors, chosen.size)
            if take < _RESIDUAL_GATE_MIN_NEIGHBORS:
                continue
            row = np.full(n_neighbors, -1, dtype=np.int64)
            row[:take] = chosen[:take]
            gated[target] = row
    return gated
