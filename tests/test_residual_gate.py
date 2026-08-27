"""Residual-gated FASTR templates must drop motion volumes from the neighbour set."""

from __future__ import annotations

import numpy as np

from mri_correction.fastr import (
    FmriAcquisitionTiming,
    apply_fastr_batch,
    fit_fastr_alignment,
    gate_fastr_geometry,
    make_group_trigger_samples,
    prepare_fastr_geometry,
)


def make_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.05,
        slice_timing_seconds=(0.0, 0.025),
        multiband_acceleration_factor=1,
    )


def make_stationary_artifact(
    *,
    volume_count: int = 40,
    contamination_volume: int | None = None,
    contamination: float = 8000.0,
) -> tuple[np.ndarray, np.ndarray, FmriAcquisitionTiming, np.ndarray]:
    """Repeating two-slot artifact, optionally with an orthogonal spike in one volume."""
    timing = make_timing()
    interval = 50
    volume_starts = np.arange(volume_count, dtype=np.int64) * interval
    sample_count = interval * (volume_count + 2)
    data = np.zeros((1, sample_count), dtype=np.float64)
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )
    for trigger in triggers:
        start = int(trigger)
        data[0, start : start + 12] += 1000.0
    if contamination_volume is not None:
        spike_at = int(volume_starts[contamination_volume]) + 18
        data[0, spike_at : spike_at + 6] += contamination
    return data, volume_starts, timing, triggers


def _geometry(data: np.ndarray, triggers: np.ndarray):
    return prepare_fastr_geometry(
        triggers,
        sample_count=data.shape[1],
        interpolation_factor=1,
        neighbor_count=8,
        search_radius_samples=0,
        groups_per_volume=2,
        allow_edges=True,
    )


def test_gate_is_a_no_op_on_a_stationary_artifact() -> None:
    data, _volume_starts, _timing, triggers = make_stationary_artifact()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    gated = gate_fastr_geometry(
        geometry, alignment, data[0], sampling_rate=1_000.0
    )

    assert gated is geometry
    np.testing.assert_array_equal(
        gated.window.indices,
        geometry.window.indices,
    )


def test_gate_drops_a_contaminated_volume_from_neighbour_windows() -> None:
    contaminated = 20
    data, _volume_starts, timing, triggers = make_stationary_artifact(
        contamination_volume=contaminated,
    )
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    gated = gate_fastr_geometry(
        geometry, alignment, data[0], sampling_rate=1_000.0
    )

    original_groups = {
        contaminated * timing.groups_per_volume,
        contaminated * timing.groups_per_volume + 1,
    }
    assert original_groups <= set(gated.excluded_group_indices.tolist())
    mapped = np.where(
        gated.window.indices >= 0,
        geometry.group_indices[
            np.clip(gated.window.indices, 0, geometry.group_indices.size - 1)
        ],
        -1,
    )
    assert not original_groups & set(mapped.ravel().tolist())
    local = int(np.flatnonzero(geometry.group_indices == contaminated * 2)[0])
    neighbor_of_spike = local - 2
    assert local not in set(gated.window.indices[neighbor_of_spike].tolist())


def test_outlier_volumes_keep_their_local_neighbour_window() -> None:
    """A motion volume must still template from its local neighbours."""
    contaminated = 20
    data, _volume_starts, _timing, triggers = make_stationary_artifact(
        contamination_volume=contaminated,
    )
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    local = int(np.flatnonzero(geometry.group_indices == contaminated * 2)[0])
    original = np.array(geometry.window.indices[local], copy=True)

    gated = gate_fastr_geometry(
        geometry, alignment, data[0], sampling_rate=1_000.0
    )

    np.testing.assert_array_equal(gated.window.indices[local], original)


def test_gated_templates_stop_motion_leaking_into_neighbouring_volumes() -> None:
    contaminated = 20
    data, _volume_starts, _timing, triggers = make_stationary_artifact(
        contamination_volume=contaminated,
    )
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    ungated = apply_fastr_batch(data, geometry, alignment)
    gated = apply_fastr_batch(
        data,
        gate_fastr_geometry(
            geometry, alignment, data[0], sampling_rate=1_000.0
        ),
        alignment,
    )

    neighbor_start = int(triggers[(contaminated - 1) * 2])
    ungated_leftover = float(np.max(np.abs(ungated.data[0, neighbor_start : neighbor_start + 25])))
    gated_leftover = float(np.max(np.abs(gated.data[0, neighbor_start : neighbor_start + 25])))

    assert ungated_leftover > 50.0
    assert gated_leftover < 1.0
    assert gated_leftover < 0.1 * ungated_leftover


def test_clean_volumes_far_from_the_spike_stay_unchanged() -> None:
    data, _volume_starts, _timing, triggers = make_stationary_artifact(
        contamination_volume=20,
    )
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    ungated = apply_fastr_batch(data, geometry, alignment)
    gated = apply_fastr_batch(
        data,
        gate_fastr_geometry(
            geometry, alignment, data[0], sampling_rate=1_000.0
        ),
        alignment,
    )

    far_start = int(triggers[4])
    np.testing.assert_allclose(
        gated.data[0, far_start : far_start + 25],
        ungated.data[0, far_start : far_start + 25],
        atol=1e-9,
    )
