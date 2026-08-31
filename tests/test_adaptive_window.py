"""Adaptive FASTR windows shrink only where the artifact is non-stationary."""

from __future__ import annotations

import numpy as np
import pytest

from fastr_python.fastr import (
    FastrInputError,
    FmriAcquisitionTiming,
    adapt_fastr_geometry,
    apply_channel_adaptive_fastr_batch,
    apply_fastr_batch,
    apply_selected_local_fastr_batch,
    fit_fastr_alignment,
    make_group_trigger_samples,
    prepare_fastr_geometry,
)


def make_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.05,
        slice_timing_seconds=(0.0, 0.025),
        multiband_acceleration_factor=1,
    )


def make_nonstationary_burst(
    *,
    volume_count: int = 40,
    burst: tuple[int, int] = (18, 26),
    shift: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pulse shifts later during a contiguous burst of volumes."""
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
    burst_start, burst_stop = burst
    for volume, start in enumerate(volume_starts):
        offset = shift if burst_start <= volume < burst_stop else 0
        data[0, int(start) + offset : int(start) + offset + 12] += 1000.0
        group1 = int(start) + 25
        data[0, group1 + offset : group1 + offset + 12] += 1000.0
    return data, volume_starts, triggers


def _geometry(data: np.ndarray, triggers: np.ndarray, *, neighbor_count: int = 12):
    return prepare_fastr_geometry(
        triggers,
        sample_count=data.shape[1],
        interpolation_factor=1,
        neighbor_count=neighbor_count,
        search_radius_samples=0,
        groups_per_volume=2,
        allow_edges=True,
    )


def test_adapt_is_a_no_op_on_a_stationary_artifact() -> None:
    data, _starts, triggers = make_nonstationary_burst(burst=(0, 0))
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    adapted = adapt_fastr_geometry(
        geometry,
        alignment,
        data[0],
        local_neighbor_count=4,
        sampling_rate=1_000.0,
    )

    assert adapted is geometry


def test_adapt_shrinks_windows_inside_a_shape_change_burst() -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    adapted = adapt_fastr_geometry(
        geometry,
        alignment,
        data[0],
        local_neighbor_count=4,
        sampling_rate=1_000.0,
    )

    local = int(np.flatnonzero(geometry.group_indices == 22 * 2)[0])
    wide = set(geometry.window.indices[local].tolist())
    narrow = [index for index in adapted.window.indices[local] if index >= 0]
    assert len(narrow) == 4
    assert set(narrow) < wide
    assert max(abs(index - local) for index in narrow) < max(
        abs(index - local) for index in wide
    )


def test_adapted_windows_reduce_leftover_inside_the_burst() -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    wide = apply_fastr_batch(data, geometry, alignment)
    adapted = apply_fastr_batch(
        data,
        adapt_fastr_geometry(
            geometry,
            alignment,
            data[0],
            local_neighbor_count=4,
            sampling_rate=1_000.0,
        ),
        alignment,
    )

    burst_start = int(triggers[22 * 2])
    wide_leftover = float(np.max(np.abs(wide.data[0, burst_start : burst_start + 25])))
    adapted_leftover = float(
        np.max(np.abs(adapted.data[0, burst_start : burst_start + 25]))
    )
    assert wide_leftover > 50.0
    assert adapted_leftover < 0.5 * wide_leftover


def test_channel_adaptive_correction_selects_windows_per_eeg_channel() -> None:
    drifting, _starts, triggers = make_nonstationary_burst()
    stationary, _starts, _triggers = make_nonstationary_burst(burst=(0, 0))
    non_eeg = 2.0 * stationary
    data = np.vstack([drifting, stationary, non_eeg])
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(stationary[0], geometry)

    correction = apply_channel_adaptive_fastr_batch(
        data,
        geometry,
        alignment,
        local_neighbor_count=4,
        sampling_rate=1_000.0,
        unscaled_channels=(2,),
    )

    assert correction.data.shape == data.shape
    assert correction.amplitudes.shape == (3, geometry.triggers.size)
    assert correction.adapted_group_indices[0].size > 0
    assert correction.adapted_group_indices[1].size == 0
    assert correction.adapted_group_indices[2].size == 0

    wide = apply_fastr_batch(data, geometry, alignment, unscaled_channels=(2,))
    burst_start = int(triggers[22 * 2])
    burst_stop = burst_start + 25
    wide_leftover = np.max(np.abs(wide.data[0, burst_start:burst_stop]))
    adaptive_leftover = np.max(
        np.abs(correction.data[0, burst_start:burst_stop])
    )
    assert adaptive_leftover < 0.5 * wide_leftover
    np.testing.assert_allclose(correction.data[1:], wide.data[1:])


def test_channel_adaptive_correction_can_shrink_an_edge_window() -> None:
    drifting, _starts, triggers = make_nonstationary_burst(burst=(2, 10))
    geometry = _geometry(drifting, triggers)
    alignment = fit_fastr_alignment(drifting[0], geometry)

    correction = apply_channel_adaptive_fastr_batch(
        drifting,
        geometry,
        alignment,
        local_neighbor_count=4,
        sampling_rate=1_000.0,
    )

    target_group = 5 * 2
    assert target_group in correction.adapted_group_indices[0]


def test_selected_channels_use_the_local_window_for_every_group() -> None:
    drifting, _starts, triggers = make_nonstationary_burst()
    stationary, _starts, _triggers = make_nonstationary_burst(burst=(0, 0))
    data = np.vstack([drifting, stationary])
    wide_geometry = _geometry(data, triggers)
    local_geometry = _geometry(data, triggers, neighbor_count=4)
    alignment = fit_fastr_alignment(stationary[0], wide_geometry)

    correction = apply_selected_local_fastr_batch(
        data,
        wide_geometry,
        alignment,
        local_neighbor_count=4,
        local_channels=(0,),
        sampling_rate=1_000.0,
    )
    expected_local = apply_fastr_batch(
        data[[0]],
        local_geometry,
        alignment,
    )
    expected_wide = apply_fastr_batch(data[[1]], wide_geometry, alignment)

    np.testing.assert_allclose(correction.data[[0]], expected_local.data)
    np.testing.assert_allclose(correction.data[[1]], expected_wide.data)
    np.testing.assert_array_equal(
        correction.adapted_group_indices[0],
        wide_geometry.group_indices,
    )
    assert correction.adapted_group_indices[1].size == 0


def test_selected_local_channels_cannot_be_non_eeg() -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    with pytest.raises(FastrInputError, match="both local and unscaled"):
        apply_selected_local_fastr_batch(
            data,
            geometry,
            alignment,
            local_neighbor_count=4,
            local_channels=(0,),
            unscaled_channels=(0,),
        )


def test_volumes_far_from_the_burst_keep_the_wide_window() -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)
    adapted = adapt_fastr_geometry(
        geometry,
        alignment,
        data[0],
        local_neighbor_count=4,
        sampling_rate=1_000.0,
    )

    far = int(np.flatnonzero(geometry.group_indices == 4 * 2)[0])
    np.testing.assert_array_equal(
        adapted.window.indices[far],
        geometry.window.indices[far],
    )


def test_adapt_explicit_default_matches_implicit_default() -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    implicit = adapt_fastr_geometry(
        geometry,
        alignment,
        data[0],
        local_neighbor_count=4,
        sampling_rate=1_000.0,
    )
    explicit = adapt_fastr_geometry(
        geometry,
        alignment,
        data[0],
        local_neighbor_count=4,
        sampling_rate=1_000.0,
        adaptive_improvement_ratio=0.85,
    )

    np.testing.assert_array_equal(implicit.window.indices, explicit.window.indices)
    np.testing.assert_array_equal(
        implicit.adapted_group_indices,
        explicit.adapted_group_indices,
    )


@pytest.mark.parametrize("value", [0.0, 1.1, np.inf, True, "0.85"])
def test_adapt_rejects_invalid_improvement_ratio(value: object) -> None:
    data, _starts, triggers = make_nonstationary_burst()
    geometry = _geometry(data, triggers)
    alignment = fit_fastr_alignment(data[0], geometry)

    with pytest.raises(FastrInputError, match="adaptive improvement ratio"):
        adapt_fastr_geometry(
            geometry,
            alignment,
            data[0],
            local_neighbor_count=4,
            sampling_rate=1_000.0,
            adaptive_improvement_ratio=value,
        )
