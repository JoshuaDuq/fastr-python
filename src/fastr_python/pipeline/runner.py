"""Configuration-driven BrainVision correction orchestration."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import mne
import numpy as np

from ..config import CorrectionConfig
from ..fastr import (
    FastrGeometry,
    FastrInputError,
    adapt_fastr_geometry,
    fit_fastr_alignment,
    gate_fastr_geometry,
    obs_trigger_subset,
    prepare_fastr_geometry,
)
from ..io.recording import (
    BrainVisionInputError,
    read_brainvision_recording,
    resample_markers,
    select_marker_sample_block,
    select_marker_samples,
    write_brainvision_recording,
)
from ..quality.psd import save_psd_plot as _save_psd_plot
from ..window import OutputWindow, resolve_output_window
from . import io as pipeline_io
from . import markers as pipeline_markers
from . import provenance as pipeline_provenance
from .acquisition import _resolve_acquisition
from .channels import (
    _ChannelProcessingContext,
    _process_configured_channel_batch,
    _record_obs_ranks,
    _retry_failed_channels,
)
from .models import CorrectionSummary, PipelineInputError
from .quality import _measure_residual_qc

__all__ = [
    "CorrectionSummary",
    "PipelineInputError",
    "run_correction",
]


def run_correction(config: CorrectionConfig) -> CorrectionSummary:
    """Run the configured FASTR correction and write a BrainVision result."""
    if not isinstance(config, CorrectionConfig):
        raise PipelineInputError("config must be a CorrectionConfig instance")
    started = time.perf_counter()
    try:
        return _run_correction(config, started=started)
    except (BrainVisionInputError, FastrInputError) as error:
        raise PipelineInputError(str(error)) from error


def _run_correction(
    config: CorrectionConfig,
    *,
    started: float,
) -> CorrectionSummary:
    pipeline_io.validate_input_files(config)
    output_paths = pipeline_io.output_paths(config.output.vhdr)
    pipeline_io.validate_output_paths(output_paths)

    recording = read_brainvision_recording(config.input.raw_vhdr)
    raw = mne.io.read_raw_brainvision(
        recording.header_path,
        preload=False,
        verbose="ERROR",
    )
    input_rate = float(raw.info["sfreq"])
    matching_marker_samples = select_marker_samples(
        recording.markers,
        marker_type=config.timing.marker_type,
        marker_description=config.timing.marker_description,
        sample_count=int(raw.n_times),
    )
    marker_samples = matching_marker_samples
    if config.timing.volume_marker_start_index is not None:
        marker_samples = select_marker_sample_block(
            matching_marker_samples,
            start_index=config.timing.volume_marker_start_index,
            count=config.timing.volume_marker_count,
        )
    resolved = _resolve_acquisition(
        config,
        marker_samples,
        sampling_rate=input_rate,
    )
    acquisition = resolved.geometry
    output_rate, decimation = pipeline_io.validate_rates(
        input_rate,
        config.processing.output_sampling_rate_hz,
        config.processing.lowpass_hz,
    )
    window = resolve_output_window(
        acquisition.volume_starts,
        mode=config.trim.mode,
        input_sample_count=int(raw.n_times),
    )
    reference_index = pipeline_io.resolve_reference_channel(
        raw.ch_names,
        config.processing.reference_channel,
    )
    geometry = prepare_fastr_geometry(
        acquisition.group_triggers,
        sample_count=int(raw.n_times),
        interpolation_factor=config.processing.interpolation_factor,
        neighbor_count=config.processing.neighbor_count,
        search_radius_samples=config.processing.search_radius_samples,
        groups_per_volume=acquisition.groups_per_volume,
        allow_edges=True,
        pre_trigger_fraction=config.processing.pre_trigger_fraction,
    )
    reference_channel = raw.get_data(
        picks=[reference_index],
        start=0,
        stop=raw.n_times,
    )[0]
    alignment = fit_fastr_alignment(
        reference_channel,
        geometry,
        template_high_pass_hz=config.processing.template_high_pass_hz,
        sampling_rate=input_rate,
    )
    if config.processing.residual_gate:
        geometry = gate_fastr_geometry(
            geometry,
            alignment,
            reference_channel,
            template_high_pass_hz=config.processing.template_high_pass_hz,
            sampling_rate=input_rate,
            residual_gate_mad_multiplier=(
                config.processing.residual_gate_mad_multiplier
            ),
            residual_gate_ratio=config.processing.residual_gate_ratio,
            residual_gate_max_fraction=config.processing.residual_gate_max_fraction,
            mains_frequency_hz=config.quality_control.mains_frequency_hz,
            mains_exclusion_hz=config.quality_control.mains_exclusion_hz,
        )
    if config.processing.adaptive_window:
        geometry = adapt_fastr_geometry(
            geometry,
            alignment,
            reference_channel,
            local_neighbor_count=config.processing.local_neighbor_count,
            template_high_pass_hz=config.processing.template_high_pass_hz,
            sampling_rate=input_rate,
            adaptive_improvement_ratio=config.processing.adaptive_improvement_ratio,
        )

    obs_triggers = (
        obs_trigger_subset(
            acquisition.volume_starts,
            sample_count=int(raw.n_times),
            interpolation_factor=config.processing.interpolation_factor,
        )
        if config.processing.residual_obs
        else None
    )

    channel_count = len(raw.ch_names)
    non_eeg_indices = frozenset(
        index
        for index, name in enumerate(raw.ch_names)
        if name in config.processing.non_eeg_channels
    )
    local_window_indices = _resolve_named_channels(
        raw.ch_names,
        config.processing.local_window_channels,
        setting="processing.local_window_channels",
    )
    input_sample_count = int(raw.n_times)
    output_sample_count = (window.length - 1) // decimation + 1
    amplitude_means = np.empty(channel_count, dtype=np.float64)
    amplitude_rms = np.empty(channel_count, dtype=np.float64)
    selected_obs_ranks = np.empty((channel_count, 0), dtype=np.int64)
    adapted_group_indices_by_channel = [
        np.empty(0, dtype=np.int64) for _ in range(channel_count)
    ]
    anc_reference_scales = np.full(channel_count, np.nan, dtype=np.float64)
    anc_step_sizes = np.full(channel_count, np.nan, dtype=np.float64)
    anc_filter_order = (
        math.ceil(geometry.epoch.length / geometry.interpolation_factor)
        if config.processing.adaptive_noise_cancellation
        else None
    )
    anc_sample_slice = (
        _corrected_input_span(geometry, input_sample_count)
        if config.processing.adaptive_noise_cancellation
        else None
    )
    with tempfile.TemporaryDirectory(
        dir=config.output.vhdr.parent,
        prefix=".fastr-python-",
    ) as temporary_directory:
        output_path = Path(temporary_directory) / "corrected-output.dat"
        corrected_output = np.memmap(
            output_path,
            mode="w+",
            dtype=np.float64,
            shape=(channel_count, output_sample_count),
        )
        context = _ChannelProcessingContext(
            config=config,
            geometry=geometry,
            alignment=alignment,
            input_rate=input_rate,
            output_rate=output_rate,
            window=window,
            obs_triggers=obs_triggers,
            anc_filter_order=anc_filter_order,
            anc_sample_slice=anc_sample_slice,
        )
        for start in range(0, channel_count, config.processing.channel_batch_size):
            stop = min(
                start + config.processing.channel_batch_size,
                channel_count,
            )
            batch = raw.get_data(
                picks=list(range(start, stop)),
                start=0,
                stop=raw.n_times,
            )
            processed = _process_configured_channel_batch(
                batch,
                context=context,
                global_channel_indices=tuple(range(start, stop)),
                non_eeg_indices=non_eeg_indices,
                local_window_indices=local_window_indices,
            )
            amplitude_means[start:stop] = processed.amplitude_means
            amplitude_rms[start:stop] = processed.amplitude_rms
            adapted_group_indices_by_channel[start:stop] = (
                processed.adapted_group_indices
            )
            if config.processing.residual_obs:
                selected_obs_ranks = _record_obs_ranks(
                    selected_obs_ranks,
                    processed.selected_obs_ranks,
                    channel_count=channel_count,
                    rows=slice(start, stop),
                )
            anc_reference_scales[start:stop] = processed.anc_reference_scales
            anc_step_sizes[start:stop] = processed.anc_step_sizes
            corrected_output[start:stop] = processed.data
        corrected_output.flush()
        policy_result, retried_channels = _retry_failed_channels(
            corrected_output,
            raw=raw,
            context=context,
            acquisition=acquisition,
            non_eeg_indices=non_eeg_indices,
        )
        for index, retried in retried_channels.items():
            corrected_output[index] = retried.data[0]
            amplitude_means[index] = retried.amplitude_means[0]
            amplitude_rms[index] = retried.amplitude_rms[0]
            adapted_group_indices_by_channel[index] = (
                retried.adapted_group_indices[0]
            )
            if config.processing.residual_obs:
                selected_obs_ranks = _record_obs_ranks(
                    selected_obs_ranks,
                    retried.selected_obs_ranks,
                    channel_count=channel_count,
                    rows=slice(index, index + 1),
                )
            anc_reference_scales[index] = retried.anc_reference_scales[0]
            anc_step_sizes[index] = retried.anc_step_sizes[0]
        if retried_channels:
            corrected_output.flush()
        # Measured after installation, so the sidecar, the PSDs and the block
        # markers all describe the samples that were actually written.
        residual_qc = _measure_residual_qc(
            corrected_output,
            channel_names=raw.ch_names,
            non_eeg_indices=non_eeg_indices,
            output_rate=output_rate,
            acquisition=acquisition,
            threshold_uv=config.processing.residual_threshold_uv,
            block_seconds=config.quality_control.block_seconds,
            mains_frequency_hz=config.quality_control.mains_frequency_hz,
            mains_exclusion_hz=config.quality_control.mains_exclusion_hz,
            volume_spectrum_max_hz=config.quality_control.volume_spectrum_max_hz,
            mad_multiplier=config.quality_control.residual_mad_multiplier,
            minimum_channels=config.quality_control.residual_minimum_channels,
            report_channel_outliers=(
                config.quality_control.report_channel_outliers
            ),
        )
        transformed_markers = resample_markers(
            recording.markers,
            factor=decimation,
            window=window,
        ) + pipeline_markers.bad_gradient_markers(
            pipeline_markers.skipped_group_spans(
                acquisition.group_triggers,
                geometry,
            ),
            window=window,
            decimation=decimation,
            output_sample_count=output_sample_count,
        ) + pipeline_markers.residual_qc_markers(
            residual_qc,
            output_rate=output_rate,
            output_sample_count=output_sample_count,
        )
        pipeline_markers.validate_marker_output_positions(
            transformed_markers,
            output_sample_count,
        )
        write_brainvision_recording(
            data=corrected_output,
            sampling_rate=output_rate,
            channel_names=raw.ch_names,
            output_vhdr=config.output.vhdr,
            markers=transformed_markers,
        )
        del corrected_output

    psd_limit = (
        output_rate / 2.0
        if config.processing.lowpass_hz == 0.0
        else config.processing.lowpass_hz
    )
    psd_max_frequency = min(
        config.diagnostics.psd_max_frequency_hz,
        psd_limit,
        output_rate / 2.0,
    )
    psd_tmin, psd_tmax = _corrected_psd_window(
        geometry,
        input_sampling_rate=input_rate,
        window=window,
    )
    window_offset_seconds = window.start / input_rate
    _save_psd_plot(
        raw,
        output_paths["psd_before"],
        title="Before scanner-gradient correction (complete epochs)",
        fmax=psd_max_frequency,
        n_fft=config.diagnostics.psd_n_fft,
        tmin=psd_tmin + window_offset_seconds,
        tmax=psd_tmax + window_offset_seconds,
    )
    corrected_raw = mne.io.read_raw_brainvision(
        output_paths["vhdr"],
        preload=False,
        verbose="ERROR",
    )
    _save_psd_plot(
        corrected_raw,
        output_paths["psd_after"],
        title="After scanner-gradient correction (complete epochs)",
        fmax=psd_max_frequency,
        n_fft=config.diagnostics.psd_n_fft,
        tmin=psd_tmin,
        tmax=psd_tmax,
    )

    provenance_path = output_paths["json"]
    provenance = pipeline_provenance.make_provenance(
        config,
        output_paths=output_paths,
        recording=recording,
        raw=raw,
        acquisition=acquisition,
        declared_timing=resolved.declared_timing,
        geometry=geometry,
        alignment=alignment,
        amplitude_means=amplitude_means,
        amplitude_rms=amplitude_rms,
        decimation=decimation,
        output_sample_count=output_sample_count,
        window=window,
        residual_qc=residual_qc,
        channel_failure_policy=policy_result,
        reference_index=reference_index,
        obs_epoch_count=(0 if obs_triggers is None else int(obs_triggers.size)),
        detected_volume_count=resolved.detected_volume_count,
        matching_marker_count=int(matching_marker_samples.size),
        selected_marker_count=int(marker_samples.size),
        adapted_group_indices_by_channel=adapted_group_indices_by_channel,
        selected_obs_ranks=selected_obs_ranks,
        anc_filter_order=anc_filter_order,
        anc_reference_scales=anc_reference_scales,
        anc_step_sizes=anc_step_sizes,
        psd_tmin=psd_tmin,
        psd_tmax=psd_tmax,
        psd_max_frequency_hz=psd_max_frequency,
        psd_n_fft=config.diagnostics.psd_n_fft,
        runtime_seconds=time.perf_counter() - started,
    )
    with provenance_path.open("x", encoding="utf-8") as output:
        json.dump(provenance, output, indent=2)
        output.write("\n")

    return CorrectionSummary(
        output_vhdr=output_paths["vhdr"],
        output_eeg=output_paths["eeg"],
        output_vmrk=output_paths["vmrk"],
        provenance_json=provenance_path,
        psd_before=output_paths["psd_before"],
        psd_after=output_paths["psd_after"],
        method=config.processing.method,
        input_sampling_rate_hz=input_rate,
        output_sampling_rate_hz=output_rate,
        channel_count=channel_count,
        input_sample_count=input_sample_count,
        output_sample_count=output_sample_count,
        marker_count=len(recording.markers),
        processed_group_count=geometry.triggers.size,
        skipped_group_count=geometry.skipped_group_indices.size,
    )


def _resolve_named_channels(
    channel_names: list[str],
    configured_names: tuple[str, ...],
    *,
    setting: str,
) -> frozenset[int]:
    missing = sorted(set(configured_names) - set(channel_names))
    if missing:
        raise PipelineInputError(
            f"{setting} contains channels absent from the recording: "
            f"{', '.join(missing)}"
        )
    selected = set(configured_names)
    return frozenset(
        index for index, name in enumerate(channel_names) if name in selected
    )


def _corrected_psd_window(
    geometry: FastrGeometry,
    *,
    input_sampling_rate: float,
    window: OutputWindow,
) -> tuple[float, float]:
    """Return a PSD interval, relative to the window, holding only corrected samples.

    Both figures are drawn over the same interval so they can be compared, so
    the bounds are expressed relative to the emitted window rather than to the
    start of the input recording. The before-figure is drawn from the input
    recording and must have ``window.start`` added back.
    """
    first_sample = float(geometry.triggers[0])
    last_sample = float(
        geometry.triggers[-1]
        + geometry.epoch.samples_after / geometry.interpolation_factor
    )
    first_sample = max(first_sample, float(window.start))
    last_sample = min(last_sample, float(window.stop - 1))
    tmin = (first_sample - window.start) / input_sampling_rate
    tmax = (last_sample - window.start) / input_sampling_rate
    if not 0.0 <= tmin < tmax:
        raise PipelineInputError("the corrected PSD interval is empty")
    return tmin, tmax


def _corrected_input_span(
    geometry: FastrGeometry,
    sample_count: int,
) -> slice:
    """Bound ANC to the span whose fitted artifact estimate is complete."""
    artifact_samples = math.ceil(
        geometry.epoch.length / geometry.interpolation_factor
    )
    start = max(
        0,
        int(geometry.triggers[0]) - math.ceil(1.25 * artifact_samples),
    )
    stop = min(
        sample_count,
        int(geometry.triggers[-1])
        + math.ceil(2.25 * artifact_samples)
        + 1,
    )
    return slice(start, stop)

