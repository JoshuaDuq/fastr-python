"""JSON-safe provenance assembly for correction runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

from . import __version__
from .brainvision_io import BrainVisionRecording
from .config import CorrectionConfig
from .fastr import FastrAlignment, FastrGeometry, FmriAcquisitionTiming
from .window import OutputWindow

FMRIB_REFERENCE_COMMIT = "2aa522bc5ec4215f42b3ba8efdb2b84d2a312935"


def trim_provenance(
    window: OutputWindow,
    *,
    geometry: FastrGeometry,
    input_sample_count: int,
    mode: str,
) -> dict[str, object]:
    """Report the emitted window against the margin the epochs actually need."""
    factor = geometry.interpolation_factor
    required_head = math.ceil(
        (geometry.epoch.samples_before + geometry.search_radius) / factor
    )
    required_tail = math.ceil(
        (geometry.epoch.samples_after + geometry.search_radius) / factor
    )
    return {
        "mode": mode,
        "window_start_sample": window.start,
        "window_stop_sample": window.stop,
        "window_length": window.length,
        "head_margin_samples": window.start,
        "tail_margin_samples": input_sample_count - window.stop,
        "required_head_margin_samples": required_head,
        "required_tail_margin_samples": required_tail,
    }


def make_provenance(
    config: CorrectionConfig,
    *,
    output_paths: dict[str, Path],
    recording: BrainVisionRecording,
    raw: mne.io.BaseRaw,
    timing: FmriAcquisitionTiming,
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
    return {
        "package_version": __version__,
        "method": config.processing.method,
        "input": {
            "raw_vhdr": str(recording.header_path),
            "raw_data": str(recording.data_path),
            "raw_vmrk": str(recording.marker_path),
            "fmri_metadata": str(config.input.fmri_metadata),
            "sha256": {
                "vhdr": sha256(recording.header_path),
                "eeg": sha256(recording.data_path),
                "vmrk": sha256(recording.marker_path),
                "fmri_metadata": sha256(config.input.fmri_metadata),
            },
        },
        "output": {
            "vhdr": str(output_paths["vhdr"]),
            "psd_before": str(output_paths["psd_before"]),
            "psd_after": str(output_paths["psd_after"]),
            "sampling_rate_hz": float(raw.info["sfreq"] / decimation),
            "sample_count": output_sample_count,
            "psd_interval_seconds": {
                "start": psd_tmin,
                "end": psd_tmax,
            },
            "psd_settings": {
                "fmax_hz": float(psd_max_frequency_hz),
                "n_fft": psd_n_fft,
            },
        },
        "trim": trim_provenance(
            window,
            geometry=geometry,
            input_sample_count=int(raw.n_times),
            mode=config.trim.mode,
        ),
        "residual_qc": residual_qc,
        "configuration": jsonable_config(config),
        "timing": {
            "repetition_time_seconds": timing.repetition_time_seconds,
            "slice_timing_seconds": list(timing.slice_timing_seconds),
            "multiband_acceleration_factor": timing.multiband_acceleration_factor,
            "groups_per_volume": timing.groups_per_volume,
        },
        "markers": {
            "count": len(recording.markers),
            "processed_group_count": int(geometry.triggers.size),
            "skipped_group_indices": geometry.skipped_group_indices.tolist(),
            "volume_marker_repair": {
                "mode": config.timing.missing_volume_markers,
                "detected_volume_count": detected_volume_count,
                "repaired_volume_count": (
                    config.timing.expected_volume_count - detected_volume_count
                    if config.timing.missing_volume_markers == "repair"
                    else 0
                ),
                "used_volume_count": (
                    config.timing.expected_volume_count
                    if config.timing.missing_volume_markers == "repair"
                    else detected_volume_count
                ),
            },
        },
        "fastr": {
            "reference": {
                "repository": "sccn/fMRIb",
                "commit": FMRIB_REFERENCE_COMMIT,
            },
            "interpolation_factor": geometry.interpolation_factor,
            "pre_trigger_fraction": geometry.pre_trigger_fraction,
            "samples_before_trigger": geometry.epoch.samples_before,
            "samples_after_trigger": geometry.epoch.samples_after,
            "search_radius_interpolated_samples": geometry.search_radius,
            "alignment": {
                "shifts": alignment.shifts.tolist(),
                "correlations": alignment.correlations.tolist(),
            },
            "amplitude_mean_by_channel": amplitude_means.tolist(),
            "amplitude_rms_by_channel": amplitude_rms.tolist(),
            "residual_gate": {
                "enabled": config.processing.residual_gate,
                "excluded_group_indices": geometry.excluded_group_indices.tolist(),
                "excluded_group_count": int(geometry.excluded_group_indices.size),
            },
            "residual_obs": {
                "enabled": config.processing.residual_obs,
                "rank_mode": config.processing.residual_obs_rank,
                "selected_ranks": selected_obs_ranks.tolist(),
                "section_seconds": (
                    config.processing.residual_obs_section_seconds
                ),
                "granularity": "volume",
                "corrected_epoch_count": obs_epoch_count,
            },
            "adaptive_noise_cancellation": {
                "enabled": config.processing.adaptive_noise_cancellation,
                "filter_order": anc_filter_order,
                "reference_scales": nullable_floats(anc_reference_scales),
                "step_sizes": nullable_floats(anc_step_sizes),
            },
            "adaptive_window": {
                "enabled": config.processing.adaptive_window,
                "local_neighbor_count": config.processing.local_neighbor_count,
                "adapted_group_indices": geometry.adapted_group_indices.tolist(),
                "adapted_group_count": int(geometry.adapted_group_indices.size),
            },
        },
        "runtime_seconds": runtime_seconds,
    }


def jsonable_config(config: CorrectionConfig) -> dict[str, object]:
    return stringify_paths(asdict(config))


def stringify_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [stringify_paths(item) for item in value]
    return value


def nullable_floats(values: np.ndarray) -> list[float | None]:
    """Represent unavailable channel diagnostics as JSON null values."""
    return [float(value) if math.isfinite(value) else None for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
