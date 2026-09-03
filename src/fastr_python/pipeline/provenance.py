"""JSON-safe provenance assembly for correction runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

from .. import __version__
from ..config import CorrectionConfig
from ..fastr import (
    AcquisitionGeometry,
    FastrAlignment,
    FastrGeometry,
    FmriAcquisitionTiming,
)
from ..io.recording import BrainVisionRecording
from ..window import OutputWindow
from .models import ChannelFailurePolicyResult

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


def timing_provenance(
    config: CorrectionConfig,
    *,
    acquisition: AcquisitionGeometry,
    declared_timing: FmriAcquisitionTiming | None,
) -> dict[str, object]:
    """Record the resolved acquisition timing and where each number came from.

    ``declared`` is the slice timing the run was given, absent when the markers
    supplied the geometry instead; ``resolved`` is what the correction actually
    used. Keeping both lets a reader tell a measured offset from a derived one
    without re-running anything.
    """
    return {
        "marker_kind": config.timing.marker_kind,
        "group_position_source": acquisition.source,
        "declared_timing_source": (
            "bids_sidecar"
            if config.input.fmri_metadata is not None
            else "configuration"
            if config.acquisition is not None
            else None
        ),
        "declared": (
            None
            if declared_timing is None
            else {
                "repetition_time_seconds": declared_timing.repetition_time_seconds,
                "slice_timing_seconds": list(declared_timing.slice_timing_seconds),
                "multiband_acceleration_factor": (
                    declared_timing.multiband_acceleration_factor
                ),
            }
        ),
        "resolved": {
            "repetition_time_seconds": acquisition.repetition_time_seconds,
            "groups_per_volume": acquisition.groups_per_volume,
            "group_offsets_seconds": list(acquisition.group_offsets_seconds),
            "volume_count": acquisition.volume_count,
        },
    }


def channel_failure_policy_provenance(
    config: CorrectionConfig,
    *,
    result: ChannelFailurePolicyResult,
    channel_names: list[str],
    reference_index: int,
) -> dict[str, object]:
    """Record every automatic channel decision the run made, and the bar it used.

    Both sides of each retry comparison are kept, because "rejected" is only
    informative next to the numbers that rejected it. The recommendation is
    advisory and lives here alone: no channel was replaced, interpolated or
    removed, so a reader who disagrees still has the data to disagree with.
    """

    def named(indices: frozenset[int] | dict[int, np.ndarray]) -> list[str]:
        return [channel_names[index] for index in sorted(indices)]

    def named_blocks(blocks: dict[int, np.ndarray]) -> dict[str, list[int]]:
        return {
            channel_names[index]: [int(block) for block in blocks[index]]
            for index in sorted(blocks)
        }

    return {
        "policy": config.processing.channel_failure_policy,
        "enabled": (
            config.processing.channel_failure_policy
            == "retry_local_and_recommend_bad"
        ),
        "absolute_floor_uv": float(
            config.quality_control.bad_channel_residual_uv
        ),
        "spatial_mad_multiplier": float(
            config.quality_control.residual_mad_multiplier
        ),
        "local_neighbor_count": config.processing.local_neighbor_count,
        "candidate_channels": named(result.candidate_blocks_by_channel),
        "candidate_blocks_by_channel": named_blocks(
            result.candidate_blocks_by_channel
        ),
        "retry_by_channel": {
            channel_names[index]: {
                "accepted": bool(evaluation.accepted),
                "reason": evaluation.reason,
                "wide_failed_blocks": [
                    int(block) for block in evaluation.wide_failed_blocks
                ],
                "local_failed_blocks": [
                    int(block) for block in evaluation.local_failed_blocks
                ],
                "wide_maximum_uv": float(evaluation.wide_maximum_uv),
                "local_maximum_uv": float(evaluation.local_maximum_uv),
            }
            for index, evaluation in sorted(result.retry_evaluations.items())
        },
        "accepted_local_window_channels": named(result.accepted_channels),
        "final_failed_blocks_by_channel": named_blocks(
            result.final_failed_blocks_by_channel
        ),
        "recommended_bad_channels": named(result.recommended_bad_channels),
        "reference_channel_recommended_bad": (
            reference_index in result.recommended_bad_channels
        ),
    }


def make_provenance(
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
    channel_failure_policy: ChannelFailurePolicyResult,
    reference_index: int,
    obs_epoch_count: int,
    detected_volume_count: int,
    matching_marker_count: int,
    selected_marker_count: int,
    adapted_group_indices_by_channel: list[np.ndarray],
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
    """Assemble the JSON-serializable provenance record for one run."""
    return {
        "package_version": __version__,
        "method": config.processing.method,
        "input": {
            "raw_vhdr": str(recording.header_path),
            "raw_data": str(recording.data_path),
            "raw_vmrk": str(recording.marker_path),
            "fmri_metadata": (
                None
                if config.input.fmri_metadata is None
                else str(config.input.fmri_metadata)
            ),
            "sha256": {
                "vhdr": sha256(recording.header_path),
                "eeg": sha256(recording.data_path),
                "vmrk": sha256(recording.marker_path),
                "fmri_metadata": (
                    None
                    if config.input.fmri_metadata is None
                    else sha256(config.input.fmri_metadata)
                ),
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
        "timing": timing_provenance(
            config,
            acquisition=acquisition,
            declared_timing=declared_timing,
        ),
        "markers": {
            "count": len(recording.markers),
            "processed_group_count": int(geometry.triggers.size),
            "skipped_group_indices": geometry.skipped_group_indices.tolist(),
            "volume_marker_selection": {
                "matching_marker_count": matching_marker_count,
                "selected_marker_count": selected_marker_count,
                "start_index": config.timing.volume_marker_start_index,
                "count": config.timing.volume_marker_count,
            },
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
            "channel_adaptive_window": {
                "enabled": config.processing.channel_adaptive_window,
                "local_neighbor_count": config.processing.local_neighbor_count,
                "adapted_group_indices_by_channel": {
                    channel_name: indices.tolist()
                    for channel_name, indices in zip(
                        raw.ch_names,
                        adapted_group_indices_by_channel,
                        strict=True,
                    )
                },
                "adapted_channel_count": sum(
                    indices.size > 0
                    for indices in adapted_group_indices_by_channel
                ),
            },
            "channel_failure_policy": channel_failure_policy_provenance(
                config,
                result=channel_failure_policy,
                channel_names=raw.ch_names,
                reference_index=reference_index,
            ),
            "local_window_channels": {
                "enabled": bool(config.processing.local_window_channels),
                "channels": list(config.processing.local_window_channels),
                "local_neighbor_count": config.processing.local_neighbor_count,
                "corrected_group_count": (
                    int(geometry.group_indices.size)
                    if config.processing.local_window_channels
                    else 0
                ),
            },
        },
        "runtime_seconds": runtime_seconds,
    }


def jsonable_config(config: CorrectionConfig) -> dict[str, object]:
    """Convert a correction configuration to JSON-compatible values."""
    return stringify_paths(asdict(config))


def stringify_paths(value: object) -> object:
    """Recursively convert ``Path`` values and tuples to JSON-safe values."""
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
    """Return the SHA-256 hexadecimal digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
