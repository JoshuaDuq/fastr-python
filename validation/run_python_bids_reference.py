"""Run the production BIDS acquisition-group geometry on a MATLAB span."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from fastr_python.fastr import (
    adaptive_noise_cancel,
    apply_fastr_batch,
    fit_fastr_alignment,
    fit_residual_obs,
    load_bids_fmri_timing,
    make_group_trigger_samples,
    obs_trigger_subset,
    prepare_fastr_geometry,
)
from fastr_python.pipeline.io import lowpass_and_decimate
from fastr_python.window import OutputWindow


def run_bids_stages(
    input_mat: Path,
    bids_metadata: Path,
    output_mat: Path,
    *,
    template_high_pass_hz: float,
    obs_rank: int | str,
    obs_section_seconds: float | None,
) -> None:
    """Apply the production template and OBS stages to one bounded span."""
    if output_mat.exists():
        raise FileExistsError(f"output MAT file already exists: {output_mat}")
    reference = loadmat(input_mat, simplify_cells=True)
    parameters = reference["parameters"]
    raw_data = np.asarray(reference["raw_data"], dtype=np.float64)
    volume_starts = np.asarray(reference["triggers"], dtype=np.int64) - 1
    sampling_rate = float(reference["sampling_rate"])
    interpolation_factor = int(parameters["interpolation_factor"])
    excluded = tuple(
        int(index - 1) for index in np.atleast_1d(parameters["excluded_channels"])
    )

    timing = load_bids_fmri_timing(bids_metadata)
    group_triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=raw_data.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=int(parameters["window"]),
        search_radius_samples=3,
        groups_per_volume=timing.groups_per_volume,
        allow_edges=True,
        pre_trigger_fraction=float(parameters["pre_trigger_fraction"]),
    )
    alignment = fit_fastr_alignment(
        raw_data[0],
        geometry,
        template_high_pass_hz=template_high_pass_hz,
        sampling_rate=sampling_rate,
    )
    template = apply_fastr_batch(
        raw_data,
        geometry,
        alignment,
        template_high_pass_hz=template_high_pass_hz,
        sampling_rate=sampling_rate,
        unscaled_channels=excluded,
    )
    obs_triggers = obs_trigger_subset(
        volume_starts,
        sample_count=raw_data.shape[1],
        interpolation_factor=interpolation_factor,
    )
    obs = fit_residual_obs(
        template.data,
        obs_triggers,
        sampling_rate=sampling_rate,
        excluded_channels=excluded,
        rank=obs_rank,
        interpolation_factor=interpolation_factor,
        section_seconds=obs_section_seconds,
    )
    corrected_data = obs.data
    anc_reference_scales = np.full(raw_data.shape[0], np.nan)
    anc_step_sizes = np.full(raw_data.shape[0], np.nan)
    anc_filter_order = 0
    if bool(parameters["anc_enabled"]):
        anc_filter_order = math.ceil(
            geometry.epoch.length / geometry.interpolation_factor
        )
        artifact_samples = anc_filter_order
        sample_slice = slice(
            max(
                0,
                int(geometry.triggers[0]) - math.ceil(1.25 * artifact_samples),
            ),
            min(
                raw_data.shape[1],
                int(geometry.triggers[-1]) + math.ceil(2.25 * artifact_samples) + 1,
            ),
        )
        anc = adaptive_noise_cancel(
            corrected_data,
            raw_data - corrected_data,
            sampling_rate=sampling_rate,
            filter_order=anc_filter_order,
            excluded_channels=excluded,
            sample_slice=sample_slice,
        )
        corrected_data = anc.data
        anc_reference_scales = anc.reference_scales
        anc_step_sizes = anc.step_sizes
    corrected_data = lowpass_and_decimate(
        corrected_data,
        sampling_rate=sampling_rate,
        output_sampling_rate=sampling_rate,
        lowpass_hz=float(parameters["lowpass_hz"]),
        window=OutputWindow(start=0, stop=raw_data.shape[1]),
    )
    savemat(
        output_mat,
        {
            "raw_data": raw_data,
            "corrected_data": corrected_data,
            "triggers": np.asarray(reference["triggers"]),
            "sampling_rate": sampling_rate,
            "channel_names": np.asarray(reference["channel_names"], dtype=object),
            "sample_start_zero_based": int(reference["sample_start_zero_based"]),
            "selected_obs_ranks": obs.selected_ranks,
            "group_count": geometry.triggers.size,
            "skipped_group_count": geometry.skipped_group_indices.size,
            "template_high_pass_hz": template_high_pass_hz,
            "anc_filter_order": anc_filter_order,
            "anc_reference_scales": anc_reference_scales,
            "anc_step_sizes": anc_step_sizes,
        },
        do_compression=True,
    )


def _rank(value: str) -> int | str:
    return "auto" if value == "auto" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path)
    parser.add_argument("bids_metadata", type=Path)
    parser.add_argument("output_mat", type=Path)
    parser.add_argument("--template-high-pass-hz", type=float, required=True)
    parser.add_argument("--obs-rank", type=_rank, required=True)
    parser.add_argument("--obs-section-seconds", type=float)
    arguments = parser.parse_args()
    run_bids_stages(
        arguments.input_mat,
        arguments.bids_metadata,
        arguments.output_mat,
        template_high_pass_hz=arguments.template_high_pass_hz,
        obs_rank=arguments.obs_rank,
        obs_section_seconds=arguments.obs_section_seconds,
    )


if __name__ == "__main__":
    main()
