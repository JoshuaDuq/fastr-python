import numpy as np
import pytest

from mri_correction.aas import (
    CorrectionInputError,
    analyzer_reference,
    slice_group_aas,
)


def test_slice_group_aas_preserves_exact_tr_harmonic_better_than_volume_aas() -> None:
    sampling_rate = 1_000
    samples_per_volume = 900
    samples_per_group = 50
    volume_count = 25
    sample_count = samples_per_volume * volume_count
    times = np.arange(sample_count) / sampling_rate
    tr_signal = np.sin(2 * np.pi * times / 0.9)[None, :]
    group_starts = np.arange(sample_count // samples_per_group) * samples_per_group

    volume_corrected = analyzer_reference(
        tr_signal,
        np.arange(volume_count) * samples_per_volume,
        samples_per_volume=samples_per_volume,
        window_size=21,
    )

    corrected = slice_group_aas(
        tr_signal,
        group_starts,
        samples_per_group=samples_per_group,
        neighbor_count=20,
    )

    input_rms = np.sqrt(np.mean(tr_signal**2))
    volume_output_rms = np.sqrt(np.mean(volume_corrected**2))
    output_rms = np.sqrt(np.mean(corrected**2))
    assert volume_output_rms / input_rms < 1e-12
    assert output_rms / input_rms > 0.99


def test_slice_group_aas_suppresses_repeating_artifact_and_leaves_dead_time() -> None:
    samples_per_group = 20
    groups_per_volume = 4
    volume_length = 100
    volume_count = 8
    group_offsets = np.arange(groups_per_volume) * samples_per_group
    group_starts = np.concatenate(
        [group_offsets + volume * volume_length for volume in range(volume_count)]
    )
    data = np.zeros((1, volume_count * volume_length))
    artifact = np.sin(2 * np.pi * np.arange(samples_per_group) / samples_per_group)
    for start in group_starts:
        data[0, start : start + samples_per_group] = artifact
    dead_time = np.ones(20)
    data[0, 80:100] = dead_time

    corrected = slice_group_aas(
        data,
        group_starts,
        samples_per_group=samples_per_group,
        neighbor_count=20,
    )

    assert np.max(np.abs(corrected[0, group_starts[0] : group_starts[0] + 20])) < 1e-12
    np.testing.assert_array_equal(corrected[0, 80:100], dead_time)


def test_slice_group_aas_rejects_overlapping_groups() -> None:
    data = np.zeros((1, 200))

    with pytest.raises(CorrectionInputError, match="overlap"):
        slice_group_aas(
            data,
            np.array([0, 10, 40, 60, 80]),
            samples_per_group=20,
            neighbor_count=4,
        )
