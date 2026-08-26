import numpy as np
import pytest

from mri_correction.fastr import (
    FmriAcquisitionTiming,
    acquisition_group_fastr_with_edges,
    apply_fastr_batch,
    fit_fastr_alignment,
    make_group_trigger_samples,
    prepare_fastr_geometry,
)


def make_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.9,
        slice_timing_seconds=(0.0, 0.4),
        multiband_acceleration_factor=1,
    )


def make_data() -> tuple[np.ndarray, np.ndarray, FmriAcquisitionTiming]:
    timing = make_timing()
    volume_starts = np.arange(35, dtype=np.int64) * 900
    data = np.zeros((4, 31_500), dtype=np.float64)
    for channel, scale in enumerate((1.0, 0.7, 1.3, 0.4)):
        for trigger in make_group_trigger_samples(
            volume_starts,
            sampling_rate=1_000.0,
            timing=timing,
        ):
            start = int(trigger)
            data[channel, start : start + 20] += scale
    return data, volume_starts, timing


def test_shared_alignment_matches_the_existing_batch_result() -> None:
    data, volume_starts, timing = make_data()
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=data.shape[1],
        interpolation_factor=2,
        neighbor_count=2,
        search_radius_samples=0,
        groups_per_volume=timing.groups_per_volume,
        allow_edges=True,
    )
    alignment = fit_fastr_alignment(data[0], geometry)
    batched = apply_fastr_batch(data, geometry, alignment)
    existing = acquisition_group_fastr_with_edges(
        data,
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
        interpolation_factor=2,
        neighbor_count=2,
        search_radius_samples=0,
    )

    np.testing.assert_allclose(batched.data, existing.data)
    np.testing.assert_array_equal(
        batched.provenance.shifts,
        existing.provenance.shifts,
    )
    np.testing.assert_allclose(
        batched.provenance.amplitudes,
        existing.provenance.amplitudes,
    )


def test_acquisition_slot_default_uses_the_validated_local_window() -> None:
    data, volume_starts, timing = make_data()

    correction = acquisition_group_fastr_with_edges(
        data,
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
        interpolation_factor=2,
        search_radius_samples=0,
    )

    assert correction.provenance.neighbor_indices.shape[1] == 20


def test_multiple_channel_batches_reuse_one_alignment_fit() -> None:
    data, volume_starts, timing = make_data()
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=data.shape[1],
        interpolation_factor=2,
        neighbor_count=2,
        search_radius_samples=0,
        groups_per_volume=timing.groups_per_volume,
        allow_edges=True,
    )
    alignment = fit_fastr_alignment(data[0], geometry)

    first = apply_fastr_batch(data[:2], geometry, alignment)
    second = apply_fastr_batch(data[2:], geometry, alignment)
    combined = np.vstack((first.data, second.data))
    all_channels = apply_fastr_batch(data, geometry, alignment)

    np.testing.assert_allclose(combined, all_channels.data)
    np.testing.assert_array_equal(first.provenance.shifts, alignment.shifts)
    np.testing.assert_array_equal(second.provenance.shifts, alignment.shifts)


def test_geometry_rejects_mismatched_batch_sample_count() -> None:
    data, volume_starts, timing = make_data()
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=data.shape[1],
        interpolation_factor=2,
        neighbor_count=2,
        search_radius_samples=0,
        groups_per_volume=timing.groups_per_volume,
        allow_edges=True,
    )
    alignment = fit_fastr_alignment(data[0], geometry)

    try:
        apply_fastr_batch(data[:, :-1], geometry, alignment)
    except ValueError as error:
        assert "sample count" in str(error)
    else:
        raise AssertionError("a mismatched batch sample count must fail")


@pytest.mark.parametrize("groups_per_volume", [0, -1, 1.5, True])
def test_geometry_rejects_invalid_groups_per_volume(
    groups_per_volume: object,
) -> None:
    data, volume_starts, timing = make_data()
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )

    with pytest.raises(ValueError, match="groups per volume"):
        prepare_fastr_geometry(
            triggers,
            sample_count=data.shape[1],
            interpolation_factor=2,
            neighbor_count=2,
            search_radius_samples=0,
            groups_per_volume=groups_per_volume,
            allow_edges=True,
        )
