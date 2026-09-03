"""Correct channel batches and evaluate local retries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mne
import numpy as np

from ..config import CorrectionConfig
from ..fastr import (
    AcquisitionGeometry,
    FastrAlignment,
    FastrGeometry,
    adaptive_noise_cancel,
    apply_channel_adaptive_fastr_batch,
    apply_fastr_batch,
    apply_selected_local_fastr_batch,
    fit_residual_obs,
)
from ..quality.residuals import (
    LocalRetryEvaluation,
    evaluate_local_retry,
    flag_spatial_channel_blocks,
    recommend_persistent_bad_channels,
)
from ..window import OutputWindow
from . import io as pipeline_io
from .models import ChannelFailurePolicyResult, PipelineInputError
from .quality import _measure_block_residuals


@dataclass(frozen=True, slots=True)
class _ChannelProcessingContext:
    """Everything a channel needs to travel the full correction path once.

    The automatic retry has to put one channel through exactly the stages the
    first pass put it through. Naming that set of stages once, and handing it
    the same context both times, is what makes the two residual measurements it
    compares measurements of the same thing.
    """

    config: CorrectionConfig
    geometry: FastrGeometry
    alignment: FastrAlignment
    input_rate: float
    output_rate: float
    window: OutputWindow
    obs_triggers: np.ndarray | None
    anc_filter_order: int | None
    anc_sample_slice: slice | None


@dataclass(frozen=True, slots=True)
class _ProcessedChannelBatch:
    """Emitted samples and per-channel diagnostics for one processed batch."""

    data: np.ndarray
    amplitude_means: np.ndarray
    amplitude_rms: np.ndarray
    adapted_group_indices: tuple[np.ndarray, ...]
    selected_obs_ranks: np.ndarray
    anc_reference_scales: np.ndarray
    anc_step_sizes: np.ndarray


def _process_configured_channel_batch(
    data: np.ndarray,
    *,
    context: _ChannelProcessingContext,
    global_channel_indices: tuple[int, ...],
    non_eeg_indices: frozenset[int],
    local_window_indices: frozenset[int],
) -> _ProcessedChannelBatch:
    """Correct one batch with whichever template window the configuration chose."""
    processing = context.config.processing
    non_eeg_rows = [
        row
        for row, channel in enumerate(global_channel_indices)
        if channel in non_eeg_indices
    ]
    local_rows = [
        row
        for row, channel in enumerate(global_channel_indices)
        if channel in local_window_indices
    ]
    # Only the channel-adaptive mode decides a window per channel, so only it
    # has group indices to report. The other modes leave the field empty, as
    # they did before this path was shared.
    adapted_group_indices = tuple(
        np.empty(0, dtype=np.int64) for _ in global_channel_indices
    )
    if processing.local_window_channels:
        selected_local_correction = apply_selected_local_fastr_batch(
            data,
            context.geometry,
            context.alignment,
            local_neighbor_count=processing.local_neighbor_count,
            local_channels=local_rows,
            template_high_pass_hz=processing.template_high_pass_hz,
            sampling_rate=context.input_rate,
            unscaled_channels=non_eeg_rows,
        )
        corrected = selected_local_correction.data
        amplitudes = selected_local_correction.amplitudes
    elif processing.channel_adaptive_window:
        channel_adaptive_correction = apply_channel_adaptive_fastr_batch(
            data,
            context.geometry,
            context.alignment,
            local_neighbor_count=processing.local_neighbor_count,
            template_high_pass_hz=processing.template_high_pass_hz,
            sampling_rate=context.input_rate,
            adaptive_improvement_ratio=processing.adaptive_improvement_ratio,
            unscaled_channels=non_eeg_rows,
        )
        corrected = channel_adaptive_correction.data
        amplitudes = channel_adaptive_correction.amplitudes
        adapted_group_indices = channel_adaptive_correction.adapted_group_indices
    else:
        correction = apply_fastr_batch(
            data,
            context.geometry,
            context.alignment,
            template_high_pass_hz=processing.template_high_pass_hz,
            sampling_rate=context.input_rate,
            unscaled_channels=non_eeg_rows,
        )
        corrected = correction.data
        amplitudes = correction.provenance.amplitudes
    return _finish_channel_batch(
        original=data,
        corrected=corrected,
        amplitudes=amplitudes,
        adapted_group_indices=adapted_group_indices,
        context=context,
        non_eeg_rows=non_eeg_rows,
    )


def _process_local_retry_channel(
    data: np.ndarray,
    *,
    context: _ChannelProcessingContext,
) -> _ProcessedChannelBatch:
    """Correct one EEG channel with the configured local window and nothing else.

    The retry is only ever offered a channel the spatial test already nominated,
    and those are EEG by construction, so the row is never unscaled.
    """
    correction = apply_selected_local_fastr_batch(
        data,
        context.geometry,
        context.alignment,
        local_neighbor_count=context.config.processing.local_neighbor_count,
        local_channels=(0,),
        template_high_pass_hz=context.config.processing.template_high_pass_hz,
        sampling_rate=context.input_rate,
    )
    return _finish_channel_batch(
        original=data,
        corrected=correction.data,
        amplitudes=correction.amplitudes,
        adapted_group_indices=(np.empty(0, dtype=np.int64),),
        context=context,
        non_eeg_rows=(),
    )


def _finish_channel_batch(
    *,
    original: np.ndarray,
    corrected: np.ndarray,
    amplitudes: np.ndarray,
    adapted_group_indices: tuple[np.ndarray, ...],
    context: _ChannelProcessingContext,
    non_eeg_rows: Sequence[int],
) -> _ProcessedChannelBatch:
    """Run every stage after template subtraction, once, for any set of channels."""
    processing = context.config.processing
    row_count = corrected.shape[0]
    selected_obs_ranks = np.empty((row_count, 0), dtype=np.int64)
    if processing.residual_obs:
        obs_result = fit_residual_obs(
            corrected,
            context.obs_triggers,
            sampling_rate=context.input_rate,
            excluded_channels=non_eeg_rows,
            rank=processing.residual_obs_rank,
            interpolation_factor=processing.interpolation_factor,
            section_seconds=processing.residual_obs_section_seconds,
        )
        corrected = obs_result.data
        selected_obs_ranks = obs_result.selected_ranks
    # `fmrib_fastr.m` low-passes both the corrected data and the artifact
    # estimate before cancelling, so the LMS reference is band-limited to the
    # configured cutoff. Cancelling first would leave the reference carrying
    # artifact out to the input Nyquist, shrinking the 0.05/(N*var(refs)) step
    # size and spending the filter's taps on content the low-pass then discards.
    filtered = pipeline_io.apply_output_low_pass(
        corrected,
        sampling_rate=context.input_rate,
        lowpass_hz=processing.lowpass_hz,
    )
    anc_reference_scales = np.full(row_count, np.nan, dtype=np.float64)
    anc_step_sizes = np.full(row_count, np.nan, dtype=np.float64)
    if processing.adaptive_noise_cancellation:
        artifact_estimate = pipeline_io.apply_output_low_pass(
            original - corrected,
            sampling_rate=context.input_rate,
            lowpass_hz=processing.lowpass_hz,
        )
        anc_result = adaptive_noise_cancel(
            filtered,
            artifact_estimate,
            sampling_rate=context.input_rate,
            filter_order=context.anc_filter_order,
            excluded_channels=non_eeg_rows,
            sample_slice=context.anc_sample_slice,
        )
        filtered = anc_result.data
        anc_reference_scales = anc_result.reference_scales
        anc_step_sizes = anc_result.step_sizes
    output = pipeline_io.emit_output_window(
        filtered,
        sampling_rate=context.input_rate,
        output_sampling_rate=context.output_rate,
        lowpass_hz=processing.lowpass_hz,
        window=context.window,
    )
    if processing.line_noise_frequencies_hz:
        unscaled = set(non_eeg_rows)
        eeg_rows = [row for row in range(row_count) if row not in unscaled]
        if eeg_rows:
            output[eeg_rows] = pipeline_io.remove_line_noise(
                output[eeg_rows],
                sampling_rate=context.output_rate,
                frequencies_hz=processing.line_noise_frequencies_hz,
            )
    return _ProcessedChannelBatch(
        data=output,
        amplitude_means=amplitudes.mean(axis=1),
        amplitude_rms=np.sqrt(np.mean(amplitudes**2, axis=1)),
        adapted_group_indices=adapted_group_indices,
        selected_obs_ranks=selected_obs_ranks,
        anc_reference_scales=anc_reference_scales,
        anc_step_sizes=anc_step_sizes,
    )


def _retry_failed_channels(
    corrected_output: np.ndarray,
    *,
    raw: mne.io.BaseRaw,
    context: _ChannelProcessingContext,
    acquisition: AcquisitionGeometry,
    non_eeg_indices: frozenset[int],
) -> tuple[ChannelFailurePolicyResult, dict[int, _ProcessedChannelBatch]]:
    """Retry isolated residual failures locally, and report what survived.

    The thresholds are frozen from the wide pass and every comparison is made
    against them, so a retry is judged against the bar its own channel failed
    rather than against one the retry itself moved. A retry is installed only
    when it is materially better; otherwise the wide row and every diagnostic
    belonging to it are kept untouched.
    """
    config = context.config
    if config.processing.channel_failure_policy != "retry_local_and_recommend_bad":
        return ChannelFailurePolicyResult.inactive(), {}
    channel_count = corrected_output.shape[0]
    eeg_channels = tuple(
        index for index in range(channel_count) if index not in non_eeg_indices
    )
    if not eeg_channels:
        return ChannelFailurePolicyResult.inactive(), {}
    wide = _measure_block_residuals(
        corrected_output,
        output_rate=context.output_rate,
        acquisition=acquisition,
        block_seconds=config.quality_control.block_seconds,
        mains_frequency_hz=config.quality_control.mains_frequency_hz,
        mains_exclusion_hz=config.quality_control.mains_exclusion_hz,
    )
    if wide.residuals_uv.shape[1] == 0:
        return ChannelFailurePolicyResult.inactive(), {}
    spatial = flag_spatial_channel_blocks(
        wide.residuals_uv,
        eeg_channels=eeg_channels,
        absolute_floor_uv=config.quality_control.bad_channel_residual_uv,
        mad_multiplier=config.quality_control.residual_mad_multiplier,
    )
    candidate_blocks = {
        index: np.flatnonzero(spatial.flags[index])
        for index in eeg_channels
        if spatial.flags[index].any()
    }

    final_residuals = wide.residuals_uv.copy()
    evaluations: dict[int, LocalRetryEvaluation] = {}
    accepted: set[int] = set()
    retried_channels: dict[int, _ProcessedChannelBatch] = {}
    for index in sorted(candidate_blocks):
        retried = _process_local_retry_channel(
            raw.get_data(picks=[index], start=0, stop=raw.n_times),
            context=context,
        )
        local = _measure_block_residuals(
            retried.data,
            output_rate=context.output_rate,
            acquisition=acquisition,
            block_seconds=config.quality_control.block_seconds,
            mains_frequency_hz=config.quality_control.mains_frequency_hz,
            mains_exclusion_hz=config.quality_control.mains_exclusion_hz,
        )
        evaluation = evaluate_local_retry(
            wide.residuals_uv[index],
            local.residuals_uv[0],
            spatial.thresholds_uv,
        )
        evaluations[index] = evaluation
        if evaluation.accepted:
            accepted.add(index)
            retried_channels[index] = retried
            final_residuals[index] = local.residuals_uv[0]

    rows = list(eeg_channels)
    final_flags = np.zeros(wide.residuals_uv.shape, dtype=bool)
    final_flags[rows] = final_residuals[rows] > spatial.thresholds_uv
    recommended = recommend_persistent_bad_channels(final_flags)
    return (
        ChannelFailurePolicyResult(
            candidate_blocks_by_channel=candidate_blocks,
            retry_evaluations=evaluations,
            accepted_channels=frozenset(accepted),
            final_failed_blocks_by_channel={
                index: np.flatnonzero(final_flags[index])
                for index in eeg_channels
                if final_flags[index].any()
            },
            recommended_bad_channels=frozenset(
                index for index in eeg_channels if recommended[index]
            ),
        ),
        retried_channels,
    )


def _record_obs_ranks(
    selected_obs_ranks: np.ndarray,
    batch_ranks: np.ndarray,
    *,
    channel_count: int,
    rows: slice,
) -> np.ndarray:
    """Record one batch's OBS ranks, sizing the run-level array on first use."""
    if selected_obs_ranks.shape[1] == 0:
        selected_obs_ranks = np.zeros(
            (channel_count, batch_ranks.shape[1]),
            dtype=np.int64,
        )
    if batch_ranks.shape[1] != selected_obs_ranks.shape[1]:
        raise PipelineInputError(
            "OBS section count changed between channel batches"
        )
    selected_obs_ranks[rows] = batch_ranks
    return selected_obs_ranks
