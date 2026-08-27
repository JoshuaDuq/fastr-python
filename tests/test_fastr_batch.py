import numpy as np
import pytest

from mri_correction.fastr import (
    FastrInputError,
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


def make_drifting_recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A periodic artifact, white EEG, and a step baseline shift partway through."""
    rng = np.random.default_rng(0)
    interval = 50
    group_count = 400
    sample_count = interval * (group_count + 2)
    phase = np.arange(interval) / interval
    shape = np.sin(2.0 * np.pi * phase) * 1000.0
    artifact = np.tile(shape, sample_count // interval + 1)[:sample_count]
    eeg = rng.standard_normal(sample_count) * 5.0
    drift = np.zeros(sample_count)
    drift[sample_count // 3 : 2 * sample_count // 3] = 400.0
    triggers = (np.arange(group_count) + 1) * interval
    return artifact, eeg + drift, triggers, sample_count


def test_baseline_drift_does_not_leak_into_the_template() -> None:
    artifact, signal, triggers, sample_count = make_drifting_recording()
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )
    alignment = fit_fastr_alignment(artifact + signal, geometry)

    leaking = apply_fastr_batch(
        np.vstack([artifact + signal]),
        geometry,
        alignment,
    )
    corrected = apply_fastr_batch(
        np.vstack([artifact + signal]),
        geometry,
        alignment,
        template_high_pass_hz=1.0,
        sampling_rate=1000.0,
    )

    span = slice(sample_count // 3 + 250, 2 * sample_count // 3 - 250)
    truth = signal[span]
    # the step is signal, not gradient artifact, so it must survive
    assert abs(corrected.data[0, span].mean() - truth.mean()) < 20.0
    assert abs(leaking.data[0, span].mean() - truth.mean()) > 50.0
    # and the scalar is no longer dragged around by the baseline
    assert corrected.provenance.amplitudes.std() < leaking.provenance.amplitudes.std()
    assert corrected.provenance.amplitudes.std() < 0.02


def test_template_high_pass_costs_little_on_a_drift_free_recording() -> None:
    """Without drift the high-pass has nothing to remove, so it should barely matter.

    It is not free: the high-passed template carries slightly different
    neighbour noise, which costs a few percent of recovery accuracy. That price
    buys a hundredfold residual reduction once the baseline does move.
    """
    rng = np.random.default_rng(1)
    interval = 50
    group_count = 300
    sample_count = interval * (group_count + 2)
    phase = np.arange(interval) / interval
    artifact = np.tile(
        np.sin(2.0 * np.pi * phase) * 1000.0,
        sample_count // interval + 1,
    )[:sample_count]
    eeg = rng.standard_normal(sample_count) * 5.0
    triggers = (np.arange(group_count) + 1) * interval
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )
    alignment = fit_fastr_alignment(artifact + eeg, geometry)

    plain = apply_fastr_batch(np.vstack([artifact + eeg]), geometry, alignment)
    filtered = apply_fastr_batch(
        np.vstack([artifact + eeg]),
        geometry,
        alignment,
        template_high_pass_hz=1.0,
        sampling_rate=1000.0,
    )

    span = slice(interval * 30, sample_count - interval * 30)
    truth = eeg[span]
    plain_error = float((plain.data[0, span] - truth).std())
    filtered_error = float((filtered.data[0, span] - truth).std())

    assert np.corrcoef(filtered.data[0, span], truth)[0, 1] > 0.95
    assert filtered_error < 1.2 * plain_error


def test_alignment_uses_the_same_high_pass_as_the_template() -> None:
    """Niazy stage 2 estimates shifts on the same high-passed copy as the template."""
    rng = np.random.default_rng(2)
    interval = 50
    group_count = 300
    sample_count = interval * (group_count + 2)
    phase = np.arange(interval) / interval
    artifact = np.tile(
        np.sin(2.0 * np.pi * phase) * 1000.0,
        sample_count // interval + 1,
    )[:sample_count]
    eeg = rng.standard_normal(sample_count) * 5.0
    triggers = (np.arange(group_count) + 1) * interval
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )

    unfiltered = fit_fastr_alignment(artifact + eeg, geometry)
    high_passed = fit_fastr_alignment(
        artifact + eeg,
        geometry,
        template_high_pass_hz=1.0,
        sampling_rate=1000.0,
    )

    # filtfilt transients can move the first and last epochs by one fine sample
    np.testing.assert_array_equal(unfiltered.shifts[1:-2], high_passed.shifts[1:-2])
    assert np.max(np.abs(high_passed.shifts)) <= 1
    assert high_passed.shifts.shape == geometry.fine_triggers.shape


def test_template_high_pass_requires_a_sampling_rate() -> None:
    artifact, signal, triggers, sample_count = make_drifting_recording()
    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=sample_count,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )
    alignment = fit_fastr_alignment(artifact + signal, geometry)

    with pytest.raises(FastrInputError, match="sampling rate"):
        apply_fastr_batch(
            np.vstack([artifact + signal]),
            geometry,
            alignment,
            template_high_pass_hz=1.0,
        )
