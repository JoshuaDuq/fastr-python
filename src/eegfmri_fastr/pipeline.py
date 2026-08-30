"""Public, configuration-driven BrainVision correction pipeline."""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np

from . import pipeline_io, pipeline_markers, pipeline_provenance
from .brainvision import BrainVisionMarker
from .brainvision_io import (
    BrainVisionInputError,
    BrainVisionRecording,
    read_brainvision_recording,
    resample_markers,
    select_marker_samples,
    write_brainvision_recording,
)
from .config import CorrectionConfig
from .fastr import (
    AcquisitionGeometry,
    FastrAlignment,
    FastrGeometry,
    FastrInputError,
    FmriAcquisitionTiming,
    adapt_fastr_geometry,
    adaptive_noise_cancel,
    apply_fastr_batch,
    fit_fastr_alignment,
    fit_residual_obs,
    gate_fastr_geometry,
    load_bids_fmri_timing,
    obs_trigger_subset,
    prepare_fastr_geometry,
    repair_volume_starts,
    slice_marker_geometry,
    volume_marker_geometry,
)
from .pipeline_types import PipelineInputError
from .psd import prepare_psd_raw, save_psd_plot
from .residual_qc import (
    block_residual_uv,
    flag_blocks,
    residual_qc_defaults,
    slice_harmonics,
    volume_harmonic_spectrum,
)
from .window import OutputWindow, resolve_output_window


@dataclass(frozen=True, slots=True)
class _ResolvedAcquisition:
    """Acquisition geometry, plus what the run was told and what it found.

    ``declared_timing`` is absent when acquisition-group markers supplied the
    geometry, and ``detected_volume_count`` is the count before any volume
    marker repair, so the sidecar can report both numbers.
    """

    geometry: AcquisitionGeometry
    declared_timing: FmriAcquisitionTiming | None
    detected_volume_count: int


@dataclass(frozen=True, slots=True)
class CorrectionSummary:
    """Stable summary of one completed correction run."""

    output_vhdr: Path
    output_eeg: Path
    output_vmrk: Path
    provenance_json: Path
    psd_before: Path
    psd_after: Path
    method: str
    input_sampling_rate_hz: float
    output_sampling_rate_hz: float
    channel_count: int
    input_sample_count: int
    output_sample_count: int
    marker_count: int
    processed_group_count: int
    skipped_group_count: int


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
    _validate_input_files(config)
    output_paths = _output_paths(config.output.vhdr)
    _validate_output_paths(output_paths)

    recording = read_brainvision_recording(config.input.raw_vhdr)
    raw = mne.io.read_raw_brainvision(
        recording.header_path,
        preload=False,
        verbose="ERROR",
    )
    input_rate = float(raw.info["sfreq"])
    marker_samples = select_marker_samples(
        recording.markers,
        marker_type=config.timing.marker_type,
        marker_description=config.timing.marker_description,
        sample_count=int(raw.n_times),
    )
    resolved = _resolve_acquisition(
        config,
        marker_samples,
        sampling_rate=input_rate,
    )
    acquisition = resolved.geometry
    output_rate, decimation = _validate_rates(
        input_rate,
        config.processing.output_sampling_rate_hz,
        config.processing.lowpass_hz,
    )
    window = resolve_output_window(
        acquisition.volume_starts,
        mode=config.trim.mode,
        input_sample_count=int(raw.n_times),
    )
    reference_index = _resolve_reference_channel(
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
    input_sample_count = int(raw.n_times)
    output_sample_count = (window.length - 1) // decimation + 1
    amplitude_means = np.empty(channel_count, dtype=np.float64)
    amplitude_rms = np.empty(channel_count, dtype=np.float64)
    selected_obs_ranks = np.empty((channel_count, 0), dtype=np.int64)
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
        prefix=".eegfmri-fastr-",
    ) as temporary_directory:
        output_path = Path(temporary_directory) / "corrected-output.dat"
        corrected_output = np.memmap(
            output_path,
            mode="w+",
            dtype=np.float64,
            shape=(channel_count, output_sample_count),
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
            batch_non_eeg_rows = [
                channel - start
                for channel in range(start, stop)
                if channel in non_eeg_indices
            ]
            correction = apply_fastr_batch(
                batch,
                geometry,
                alignment,
                template_high_pass_hz=config.processing.template_high_pass_hz,
                sampling_rate=input_rate,
                unscaled_channels=batch_non_eeg_rows,
            )
            amplitude_means[start:stop] = correction.provenance.amplitudes.mean(
                axis=1
            )
            amplitude_rms[start:stop] = np.sqrt(
                np.mean(correction.provenance.amplitudes**2, axis=1)
            )
            corrected_batch = correction.data
            if config.processing.residual_obs:
                obs_result = fit_residual_obs(
                    corrected_batch,
                    obs_triggers,
                    sampling_rate=input_rate,
                    excluded_channels=batch_non_eeg_rows,
                    rank=config.processing.residual_obs_rank,
                    interpolation_factor=config.processing.interpolation_factor,
                    section_seconds=(
                        config.processing.residual_obs_section_seconds
                    ),
                )
                corrected_batch = obs_result.data
                if selected_obs_ranks.shape[1] == 0:
                    selected_obs_ranks = np.zeros(
                        (channel_count, obs_result.selected_ranks.shape[1]),
                        dtype=np.int64,
                    )
                if obs_result.selected_ranks.shape[1] != selected_obs_ranks.shape[1]:
                    raise PipelineInputError(
                        "OBS section count changed between channel batches"
                    )
                selected_obs_ranks[start:stop] = obs_result.selected_ranks
            if config.processing.adaptive_noise_cancellation:
                artifact_estimate = batch - corrected_batch
                anc_result = adaptive_noise_cancel(
                    corrected_batch,
                    artifact_estimate,
                    sampling_rate=input_rate,
                    filter_order=anc_filter_order,
                    excluded_channels=batch_non_eeg_rows,
                    sample_slice=anc_sample_slice,
                )
                corrected_batch = anc_result.data
                anc_reference_scales[start:stop] = anc_result.reference_scales
                anc_step_sizes[start:stop] = anc_result.step_sizes
            output_batch = _lowpass_and_decimate(
                corrected_batch,
                sampling_rate=input_rate,
                output_sampling_rate=output_rate,
                lowpass_hz=config.processing.lowpass_hz,
                window=window,
            )
            if config.processing.line_noise_frequencies_hz:
                eeg_rows = [
                    channel - start
                    for channel in range(start, stop)
                    if channel not in non_eeg_indices
                ]
                if eeg_rows:
                    output_batch[eeg_rows] = pipeline_io.remove_line_noise(
                        output_batch[eeg_rows],
                        sampling_rate=output_rate,
                        frequencies_hz=(
                            config.processing.line_noise_frequencies_hz
                        ),
                    )
            corrected_output[start:stop] = output_batch
        corrected_output.flush()
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
        )
        transformed_markers = resample_markers(
            recording.markers,
            factor=decimation,
            window=window,
        ) + _bad_gradient_markers(
            _skipped_group_spans(acquisition.group_triggers, geometry),
            window=window,
            decimation=decimation,
            output_sample_count=output_sample_count,
        ) + _residual_qc_markers(
            residual_qc,
            output_rate=output_rate,
            output_sample_count=output_sample_count,
        )
        _validate_marker_output_positions(
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
    provenance = _make_provenance(
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
        obs_epoch_count=(0 if obs_triggers is None else int(obs_triggers.size)),
        detected_volume_count=resolved.detected_volume_count,
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


def _resolve_acquisition(
    config: CorrectionConfig,
    marker_samples: np.ndarray,
    *,
    sampling_rate: float,
) -> _ResolvedAcquisition:
    """Resolve where every acquisition group fires from the configured markers.

    Volume markers are expanded with declared slice timing, optionally after
    repairing interior gaps; acquisition-group markers are measured where they
    were recorded. Both paths end at the same geometry, so only this function
    has to know which convention the recording used.
    """
    if config.timing.marker_kind == "slice":
        geometry = slice_marker_geometry(
            marker_samples,
            sampling_rate=sampling_rate,
            groups_per_volume=config.timing.groups_per_volume,
            expected_repetition_time_seconds=(
                config.timing.expected_repetition_time_seconds
            ),
        )
        return _ResolvedAcquisition(
            geometry=geometry,
            declared_timing=None,
            detected_volume_count=geometry.volume_count,
        )

    timing = config.acquisition or load_bids_fmri_timing(config.input.fmri_metadata)
    volume_starts = marker_samples
    detected_volume_count = int(volume_starts.size)
    if config.timing.missing_volume_markers == "repair":
        volume_starts = repair_volume_starts(
            volume_starts,
            samples_per_volume=math.floor(
                timing.repetition_time_seconds * sampling_rate + 0.5
            ),
            expected_volume_count=config.timing.expected_volume_count,
        )
    return _ResolvedAcquisition(
        geometry=volume_marker_geometry(
            volume_starts,
            sampling_rate=sampling_rate,
            timing=timing,
        ),
        declared_timing=timing,
        detected_volume_count=detected_volume_count,
    )


def _validate_input_files(config: CorrectionConfig) -> None:
    pipeline_io.validate_input_files(config)


def _output_paths(output_vhdr: Path) -> dict[str, Path]:
    return pipeline_io.output_paths(output_vhdr)


def _validate_output_paths(output_paths: dict[str, Path]) -> None:
    pipeline_io.validate_output_paths(output_paths)


def _validate_rates(
    input_rate: float,
    output_rate: float,
    lowpass_hz: float,
) -> tuple[float, int]:
    return pipeline_io.validate_rates(input_rate, output_rate, lowpass_hz)


def _resolve_reference_channel(
    channel_names: list[str],
    reference: str | int,
) -> int:
    return pipeline_io.resolve_reference_channel(channel_names, reference)


def _lowpass_and_decimate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    window: OutputWindow,
) -> np.ndarray:
    return pipeline_io.lowpass_and_decimate(
        data,
        sampling_rate=sampling_rate,
        output_sampling_rate=output_sampling_rate,
        lowpass_hz=lowpass_hz,
        window=window,
    )


def _save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    *,
    fmax: float,
    title: str,
    tmin: float,
    tmax: float,
    n_fft: int | None = None,
) -> None:
    save_psd_plot(
        raw,
        output_path,
        fmax=fmax,
        n_fft=n_fft,
        title=title,
        tmin=tmin,
        tmax=tmax,
    )


def _prepare_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    return prepare_psd_raw(raw)


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


def _validate_marker_output_positions(
    markers: tuple[BrainVisionMarker, ...],
    output_sample_count: int,
) -> None:
    pipeline_markers.validate_marker_output_positions(markers, output_sample_count)


def _measure_residual_qc(
    corrected: np.ndarray,
    *,
    channel_names: list[str],
    non_eeg_indices: frozenset[int],
    output_rate: float,
    acquisition: AcquisitionGeometry,
    threshold_uv: float,
    block_seconds: float,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
    volume_spectrum_max_hz: float,
    mad_multiplier: float = residual_qc_defaults.MAD_MULTIPLIER,
    minimum_channels: int = residual_qc_defaults.MINIMUM_CHANNELS,
) -> dict[str, object]:
    """Measure residual gradient artifact across the whole corrected recording.

    The existing sidecar reports alignment correlations and amplitude fits, and
    neither moves when a correction fails: on this cohort both stayed healthy
    through blocks carrying twenty microvolts of residual artifact.
    """
    repetition_time = acquisition.repetition_time_seconds
    harmonics = slice_harmonics(
        groups_per_volume=acquisition.groups_per_volume,
        repetition_time_seconds=repetition_time,
        nyquist_hz=output_rate / 2.0,
        mains_hz=mains_frequency_hz,
        exclusion_hz=mains_exclusion_hz,
    )
    # A block boundary falling mid-volume splits one acquisition across two
    # blocks, so round the requested length to whole volumes.
    volumes_per_block = max(1, round(block_seconds / repetition_time))
    aligned_block_seconds = volumes_per_block * repetition_time
    residuals = block_residual_uv(
        np.asarray(corrected) * 1e6,
        sampling_rate=output_rate,
        harmonics=harmonics,
        block_seconds=aligned_block_seconds,
    )
    flagged = flag_blocks(
        residuals,
        mad_multiplier=mad_multiplier,
        minimum_channels=minimum_channels,
        floor_uv=threshold_uv,
    )
    maximum_spectrum_frequency = min(
        volume_spectrum_max_hz,
        float(np.nextafter(output_rate / 2.0, 0.0)),
    )
    eeg_indices = [
        index for index in range(len(channel_names)) if index not in non_eeg_indices
    ]
    volume_spectrum = volume_harmonic_spectrum(
        np.asarray(corrected)[eeg_indices] * 1e6,
        sampling_rate=output_rate,
        repetition_time_seconds=repetition_time,
        maximum_frequency_hz=maximum_spectrum_frequency,
        mains_frequency_hz=mains_frequency_hz,
        mains_exclusion_hz=mains_exclusion_hz,
    )
    if residuals.shape[1] == 0:
        worst_block = [-1] * residuals.shape[0]
        worst_uv = [0.0] * residuals.shape[0]
    else:
        worst_block = [int(index) for index in residuals.argmax(axis=1)]
        worst_uv = [float(value) for value in residuals.max(axis=1)]
    return {
        "block_seconds": float(aligned_block_seconds),
        "volumes_per_block": int(volumes_per_block),
        "harmonics_hz": [float(value) for value in harmonics],
        "mains_frequency_hz": float(mains_frequency_hz),
        "mains_exclusion_hz": float(mains_exclusion_hz),
        "floor_uv": float(threshold_uv),
        "mad_multiplier": float(mad_multiplier),
        "minimum_channels": int(minimum_channels),
        "channel_names": list(channel_names),
        "block_residual_uv": [[float(v) for v in row] for row in residuals],
        "worst_block_index": worst_block,
        "worst_block_uv": worst_uv,
        "flagged_blocks": [bool(value) for value in flagged],
        "flagged_block_count": int(flagged.sum()),
        "volume_harmonic_spectrum": [
            asdict(measurement) for measurement in volume_spectrum
        ],
    }


def _residual_qc_markers(
    residual_qc: dict[str, object],
    *,
    output_rate: float,
    output_sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    return pipeline_markers.residual_qc_markers(
        residual_qc,
        output_rate=output_rate,
        output_sample_count=output_sample_count,
    )


def _skipped_group_spans(
    group_triggers: np.ndarray,
    geometry: FastrGeometry,
) -> tuple[tuple[int, int], ...]:
    return pipeline_markers.skipped_group_spans(group_triggers, geometry)


def _bad_gradient_markers(
    spans: tuple[tuple[int, int], ...],
    *,
    window: OutputWindow,
    decimation: int,
    output_sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    return pipeline_markers.bad_gradient_markers(
        spans,
        window=window,
        decimation=decimation,
        output_sample_count=output_sample_count,
    )


def _trim_provenance(
    window: OutputWindow,
    *,
    geometry: FastrGeometry,
    input_sample_count: int,
    mode: str,
) -> dict[str, object]:
    return pipeline_provenance.trim_provenance(
        window,
        geometry=geometry,
        input_sample_count=input_sample_count,
        mode=mode,
    )


def _make_provenance(
    config: CorrectionConfig,
    *,
    output_paths: dict[str, Path],
    recording: BrainVisionRecording,
    raw: mne.io.BaseRaw,
    acquisition: AcquisitionGeometry,
    declared_timing: FmriAcquisitionTiming | None,
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    amplitude_means: np.ndarray,
    amplitude_rms: np.ndarray,
    decimation: int,
    output_sample_count: int,
    window: OutputWindow,
    residual_qc: dict[str, object],
    obs_epoch_count: int,
    detected_volume_count: int,
    selected_obs_ranks: np.ndarray,
    anc_filter_order: int | None,
    anc_reference_scales: np.ndarray,
    anc_step_sizes: np.ndarray,
    psd_tmin: float,
    psd_tmax: float,
    psd_max_frequency_hz: float,
    psd_n_fft: int | None,
    runtime_seconds: float,
) -> dict[str, object]:
    return pipeline_provenance.make_provenance(
        config,
        output_paths=output_paths,
        recording=recording,
        raw=raw,
        acquisition=acquisition,
        declared_timing=declared_timing,
        geometry=geometry,
        alignment=alignment,
        amplitude_means=amplitude_means,
        amplitude_rms=amplitude_rms,
        decimation=decimation,
        output_sample_count=output_sample_count,
        window=window,
        residual_qc=residual_qc,
        obs_epoch_count=obs_epoch_count,
        detected_volume_count=detected_volume_count,
        selected_obs_ranks=selected_obs_ranks,
        anc_filter_order=anc_filter_order,
        anc_reference_scales=anc_reference_scales,
        anc_step_sizes=anc_step_sizes,
        psd_tmin=psd_tmin,
        psd_tmax=psd_tmax,
        psd_max_frequency_hz=psd_max_frequency_hz,
        psd_n_fft=psd_n_fft,
        runtime_seconds=runtime_seconds,
    )


def _jsonable_config(config: CorrectionConfig) -> dict[str, object]:
    return pipeline_provenance.jsonable_config(config)


def _stringify_paths(value: object) -> object:
    return pipeline_provenance.stringify_paths(value)


def _sha256(path: Path) -> str:
    return pipeline_provenance.sha256(path)
