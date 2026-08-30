"""FASTR fitting, correction, and residual-basis processing."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, firls

from .fastr_geometry import (
    _measure_artifact_epoch,
    _to_interpolated_grid,
    _validate_epoch_bounds,
    prepare_fastr_geometry,
)
from .fastr_templates import (
    _extract_epochs,
    _fit_channel_noise,
    _fit_group_shifts,
    _interpolate,
    _make_interpolation_filter,
    _make_template_high_pass,
    _place_epochs,
    _template_estimate_signal,
    _unscaled_channel_noise,
)
from .fastr_types import (
    FastrAlignment,
    FastrCorrection,
    FastrGeometry,
    FastrInputError,
    FastrProvenance,
    ResidualObsCorrection,
    _ArtifactEpoch,
)
from .fastr_validation import (
    validate_basis_rank,
    validate_channel_indices,
    validate_group_triggers,
    validate_interpolation_factor,
    validate_positive_finite,
    validate_recording,
    validate_reference_channel,
    validate_sampling_rate,
)

_HIGH_PASS_HZ = 70.0
_HIGH_PASS_TRANSITION_HZ = 10.0
_HIGH_PASS_ORDER_FACTOR = 1.2


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
    reference = validate_reference_channel(reference_channel, geometry.sample_count)
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
    unscaled_channels: Sequence[int] = (),
) -> FastrCorrection:
    """Apply one shared alignment to a batch of recording channels.

    ``unscaled_channels`` name the rows whose template is subtracted as it
    stands instead of being scaled by a least-squares fit, which is how
    `fmrib_fastr.m` treats non-EEG channels: a QRS complex has no counterpart in
    the moving average, so a scalar fitted through it is biased.

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
    recording = validate_recording(data)
    if recording.shape[1] != geometry.sample_count:
        raise FastrInputError("batch sample count does not match the geometry")
    if alignment.shifts.shape != geometry.fine_triggers.shape:
        raise FastrInputError("alignment group count does not match the geometry")
    if alignment.correlations.shape != alignment.shifts.shape:
        raise FastrInputError("alignment correlations do not match the shifts")
    if alignment.fitted_triggers.shape != alignment.shifts.shape:
        raise FastrInputError("fitted triggers do not match the alignment")

    unscaled = validate_channel_indices(
        unscaled_channels,
        recording.shape[0],
        name="unscaled channels",
    )
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
        estimate_noise = (
            _unscaled_channel_noise if index in unscaled else _fit_channel_noise
        )
        noise, amplitudes[index] = estimate_noise(
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
    pre_trigger_fraction: float,
) -> FastrCorrection:
    """Run one explicit FASTR template geometry on validated trigger epochs."""
    recording = validate_recording(data)
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=recording.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=False,
        pre_trigger_fraction=pre_trigger_fraction,
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
    pre_trigger_fraction: float,
) -> FastrCorrection:
    recording = validate_recording(data)
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=recording.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=groups_per_volume,
        allow_edges=True,
        pre_trigger_fraction=pre_trigger_fraction,
    )
    alignment = fit_fastr_alignment(recording[0], geometry)
    return apply_fastr_batch(recording, geometry, alignment)


def obs_trigger_subset(
    group_triggers: npt.ArrayLike,
    *,
    sample_count: int,
    interpolation_factor: int = 10,
) -> np.ndarray:
    """Return the triggers whose residual epochs fit inside the recording.

    `residual_obs` refuses to run when an epoch would read past either end. The
    boundary triggers are the ones at risk, so callers that want the stage
    applied to everything it can reach select the usable subset here rather
    than guessing a margin. Dropping a boundary trigger leaves that epoch
    uncorrected, which is the same treatment template subtraction gives it.
    """
    triggers = validate_group_triggers(group_triggers)
    validate_interpolation_factor(interpolation_factor)
    if isinstance(sample_count, bool) or not isinstance(
        sample_count, (int, np.integer)
    ):
        raise FastrInputError("sample_count must be an integer")
    if sample_count < 1:
        raise FastrInputError("sample_count must be positive")

    fine = _to_interpolated_grid(triggers, interpolation_factor)
    epoch = _measure_artifact_epoch(fine, cover_full_gap=False)
    limit = sample_count * interpolation_factor
    keep = (fine - epoch.samples_before >= 0) & (
        fine + epoch.residual_samples_after <= limit
    )
    kept = triggers[keep]
    if kept.size < 2:
        raise FastrInputError(
            "no residual epochs fit inside the recording"
        )
    return kept


def residual_obs(
    residual: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    sampling_rate: float,
    excluded_channels: Sequence[int],
    rank: int | Literal["auto"] = 4,
    interpolation_factor: int = 10,
    section_seconds: float | None = None,
) -> np.ndarray:
    """Subtract the optimal basis set of the residual gradient artifact.

    ``section_seconds`` re-estimates the basis over consecutive stretches of the
    recording, as `fmrib_fastr.m` does once per section, so a residual whose
    shape drifts is not forced through one basis fitted to the whole run. Left
    unset, one basis per channel covers everything. A trailing stretch too short
    to estimate a basis joins the one before it.

    This is FASTR's third stage, never an implicit part of template subtraction.
    For each corrected channel the basis is the leading `rank` principal
    components of that channel's own high-pass residual epochs. Excluded
    channels are returned untouched, allowing callers to preserve channels that
    are not appropriate for residual artifact subtraction. Adaptive noise
    cancellation is a separate fourth-stage operation in ``fastr_anc``.
    """
    result = fit_residual_obs(
        residual,
        group_triggers,
        sampling_rate=sampling_rate,
        excluded_channels=excluded_channels,
        rank=rank,
        interpolation_factor=interpolation_factor,
        section_seconds=section_seconds,
    )
    return np.array(result.data, copy=True)


def fit_residual_obs(
    residual: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    sampling_rate: float,
    excluded_channels: Sequence[int],
    rank: int | Literal["auto"] = 4,
    interpolation_factor: int = 10,
    section_seconds: float | None = None,
) -> ResidualObsCorrection:
    """Fit residual OBS and report the rank selected in every section."""
    recording = validate_recording(residual)
    triggers = validate_group_triggers(group_triggers)
    validate_interpolation_factor(interpolation_factor)
    rate = validate_sampling_rate(sampling_rate)
    excluded = validate_channel_indices(
        excluded_channels,
        recording.shape[0],
        name="excluded channels",
    )

    fine_triggers = _to_interpolated_grid(triggers, interpolation_factor)
    epoch = _measure_artifact_epoch(fine_triggers, cover_full_gap=False)
    _validate_epoch_bounds(
        fine_triggers,
        samples_before=epoch.samples_before,
        samples_after=epoch.residual_samples_after,
        sample_count=recording.shape[1] * interpolation_factor,
    )
    rank_mode = _validate_obs_rank(rank, triggers.size)

    sections = _residual_sections(
        fine_triggers,
        section_seconds=section_seconds,
        sampling_rate=rate,
        interpolation_factor=interpolation_factor,
        rank=rank_mode if isinstance(rank_mode, int) else 1,
    )

    taps = _make_interpolation_filter(interpolation_factor)
    high_pass = _make_residual_high_pass(rate, interpolation_factor)
    corrected = recording.astype(np.float64, copy=True)
    selected_ranks = np.zeros(
        (recording.shape[0], len(sections)),
        dtype=np.int64,
    )
    for index, channel in enumerate(recording):
        if index in excluded:
            continue
        fitted, channel_ranks = _fit_residual_basis(
            _interpolate(channel, taps, interpolation_factor),
            fine_triggers,
            epoch,
            high_pass,
            rank_mode,
            sections,
        )
        corrected[index] -= fitted[::interpolation_factor]
        selected_ranks[index] = channel_ranks
    return ResidualObsCorrection(
        data=corrected,
        selected_ranks=selected_ranks,
    )


def select_obs_rank(explained_variance_percent: npt.ArrayLike) -> int:
    """Select residual-basis order with the three FMRIB knee criteria."""
    values = np.asarray(explained_variance_percent, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 5
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise FastrInputError(
            "automatic OBS rank needs at least five finite nonnegative components"
        )

    stable = np.flatnonzero(np.abs(np.diff(values)) < 2.0)
    slope_rank = next(
        (
            max(int(stable[start]), 1)
            for start in range(max(stable.size - 3, 0))
            if np.all(np.diff(stable[start : start + 4]) == 1)
        ),
        None,
    )
    cumulative = np.flatnonzero(np.cumsum(values) > 80.0)
    below_five = np.flatnonzero(values < 5.0)
    if slope_rank is None or not cumulative.size or not below_five.size:
        raise FastrInputError(
            "automatic OBS rank criteria did not identify a stable rank"
        )

    cumulative_rank = int(cumulative[0]) + 1
    variance_rank = max(int(below_five[0]), 1)
    selected = int(
        np.floor(np.mean((slope_rank, cumulative_rank, variance_rank)))
    )
    return min(max(selected, 1), values.size)


def _validate_obs_rank(
    rank: object,
    group_count: int,
) -> int | Literal["auto"]:
    if rank == "auto":
        return "auto"
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise FastrInputError(
            "basis rank must be a positive integer or 'auto'"
        )
    validate_basis_rank(rank, group_count)
    return rank


def _residual_sections(
    fine_triggers: np.ndarray,
    *,
    section_seconds: float | None,
    sampling_rate: float,
    interpolation_factor: int,
    rank: int,
) -> tuple[slice, ...]:
    """Split the epochs into the consecutive runs that each get their own basis.

    The runs are balanced rather than cut at a fixed length, so no run is left
    holding a handful of epochs: a basis fitted to barely more epochs than its
    rank spans nearly everything in that stretch and takes the signal with it.
    """
    if section_seconds is None:
        return (slice(0, fine_triggers.size),)
    seconds = validate_positive_finite(
        section_seconds,
        name="residual basis section length",
    )
    epoch_seconds = float(np.median(np.diff(fine_triggers))) / (
        sampling_rate * interpolation_factor
    )
    count = max(_round_half_up(fine_triggers.size * epoch_seconds / seconds), 1)
    if fine_triggers.size // count <= rank:
        raise FastrInputError(
            "the residual basis section holds too few epochs for the requested "
            "rank; lengthen the section or lower the rank"
        )
    edges = [0, *np.cumsum([part.size for part in np.array_split(
        np.arange(fine_triggers.size), count
    )])]
    return tuple(slice(int(start), int(stop)) for start, stop in pairwise(edges))


def _map_neighbor_indices(geometry: FastrGeometry) -> np.ndarray:
    raw_indices = geometry.window.indices
    if raw_indices.size == 0:
        return raw_indices
    safe = np.clip(raw_indices, 0, geometry.group_indices.size - 1)
    return np.where(raw_indices >= 0, geometry.group_indices[safe], -1)


def _validate_fitted_triggers(fitted_triggers: np.ndarray) -> None:
    if np.any(np.diff(fitted_triggers) <= 0):
        raise FastrInputError(
            "the fitted artifact epochs overlap; sub-sample alignment did not "
            "preserve the acquisition group order"
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
    rank: int | Literal["auto"],
    sections: tuple[slice, ...],
) -> tuple[np.ndarray, np.ndarray]:
    epochs = _extract_epochs(
        filtfilt(high_pass, 1.0, signal),
        fine_triggers,
        epoch.samples_before,
        epoch.residual_samples_after,
    )
    fitted = np.empty_like(epochs)
    selected_ranks = np.empty(len(sections), dtype=np.int64)
    for section_index, section in enumerate(sections):
        block = epochs[section]
        centered = block - block.mean(axis=1, keepdims=True)
        vectors, singular_values, _ = np.linalg.svd(
            centered.T,
            full_matrices=False,
        )
        if rank == "auto":
            variance = singular_values**2
            selected_rank = select_obs_rank(100.0 * variance / variance.sum())
        else:
            selected_rank = rank
        validate_basis_rank(selected_rank, block.shape[0])
        if selected_rank > vectors.shape[1]:
            raise FastrInputError(
                "basis rank exceeds the residual epoch dimension"
            )
        basis = vectors[:, :selected_rank]
        fitted[section] = block @ basis @ basis.T
        selected_ranks[section_index] = selected_rank
    return (
        _place_epochs(
            signal.size,
            fine_triggers - epoch.samples_before,
            fitted,
        ),
        selected_ranks,
    )


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))
