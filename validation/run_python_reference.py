"""Run Python FASTR stages over the exact span exported by MATLAB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from eegfmri_fastr.fastr import (
    apply_fastr_batch,
    fit_fastr_alignment,
    fit_residual_obs,
    obs_trigger_subset,
    prepare_fastr_geometry,
)


def run_shared_stages(
    input_mat: Path,
    output_mat: Path,
    *,
    template_high_pass_hz: float,
    obs_section_seconds: float,
) -> None:
    """Apply matched template and OBS stages to a MATLAB-exported span."""
    if output_mat.exists():
        raise FileExistsError(f"output MAT file already exists: {output_mat}")
    reference = loadmat(input_mat, simplify_cells=True)
    parameters = reference["parameters"]
    if float(parameters["lowpass_hz"]) != 0.0:
        raise ValueError("shared-stage validation requires MATLAB low-pass zero")
    if bool(parameters["anc_enabled"]):
        raise ValueError("shared-stage validation requires MATLAB ANC disabled")

    raw_data = np.asarray(reference["raw_data"], dtype=np.float64)
    triggers = np.asarray(reference["triggers"], dtype=np.int64) - 1
    sampling_rate = float(reference["sampling_rate"])
    interpolation_factor = int(parameters["interpolation_factor"])
    excluded = _zero_based_indices(parameters["excluded_channels"])
    obs_rank = _obs_rank(parameters["obs_rank"])

    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=raw_data.shape[1],
        interpolation_factor=interpolation_factor,
        neighbor_count=int(parameters["window"]),
        search_radius_samples=3,
        groups_per_volume=1,
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
    if obs_rank == 0:
        corrected_data = template.data
        selected_obs_ranks = np.empty((raw_data.shape[0], 0), dtype=np.int64)
    else:
        obs_triggers = obs_trigger_subset(
            triggers,
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
        selected_obs_ranks = obs.selected_ranks
    savemat(
        output_mat,
        {
            "raw_data": raw_data,
            "corrected_data": corrected_data,
            "triggers": np.asarray(reference["triggers"]),
            "sampling_rate": sampling_rate,
            "channel_names": np.asarray(reference["channel_names"], dtype=object),
            "sample_start_zero_based": int(reference["sample_start_zero_based"]),
            "selected_obs_ranks": selected_obs_ranks,
            "template_high_pass_hz": template_high_pass_hz,
            "obs_section_seconds": obs_section_seconds,
        },
        do_compression=True,
    )


def _zero_based_indices(values: object) -> tuple[int, ...]:
    indices = np.atleast_1d(values).astype(np.int64)
    return tuple(int(index - 1) for index in indices)


def _obs_rank(value: object) -> int | str:
    if isinstance(value, str):
        return value.lower()
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path)
    parser.add_argument("output_mat", type=Path)
    parser.add_argument("--template-high-pass-hz", type=float, required=True)
    parser.add_argument("--obs-section-seconds", type=float, required=True)
    arguments = parser.parse_args()
    run_shared_stages(
        arguments.input_mat,
        arguments.output_mat,
        template_high_pass_hz=arguments.template_high_pass_hz,
        obs_section_seconds=arguments.obs_section_seconds,
    )


if __name__ == "__main__":
    main()
