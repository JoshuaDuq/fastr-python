import numpy as np
import pytest

from mri_correction.residual_qc import (
    ResidualQcError,
    block_residual_uv,
    slice_harmonics,
)


def test_mains_harmonic_is_excluded() -> None:
    harmonics = slice_harmonics(
        groups_per_volume=18,
        repetition_time_seconds=0.9,
        nyquist_hz=500.0,
    )

    assert 20.0 in harmonics
    assert 40.0 in harmonics
    assert 60.0 not in harmonics
    assert 80.0 in harmonics
    assert 120.0 not in harmonics


def test_a_non_colliding_slice_rate_keeps_every_harmonic() -> None:
    harmonics = slice_harmonics(
        groups_per_volume=14,
        repetition_time_seconds=0.8,
        nyquist_hz=200.0,
    )

    assert 17.5 in harmonics
    assert 52.5 in harmonics
    assert 175.0 in harmonics


def test_harmonics_stop_below_nyquist() -> None:
    harmonics = slice_harmonics(
        groups_per_volume=18,
        repetition_time_seconds=0.9,
        nyquist_hz=50.0,
    )

    assert harmonics == (20.0, 40.0)


def test_a_known_residual_amplitude_is_recovered_in_microvolts() -> None:
    sampling_rate = 1000.0
    times = np.arange(int(sampling_rate * 60.0)) / sampling_rate
    rng = np.random.default_rng(0)
    background = rng.standard_normal(times.size) * 3.0
    residual = 2.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 20.0 * times)

    measured = block_residual_uv(
        np.vstack([background + residual]),
        sampling_rate=sampling_rate,
        harmonics=(20.0,),
        block_seconds=30.0,
    )

    assert measured.shape == (1, 2)
    assert np.allclose(measured, 2.0, rtol=0.1)


def test_a_clean_recording_measures_near_zero() -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 60_000)) * 3.0

    measured = block_residual_uv(
        data,
        sampling_rate=1000.0,
        harmonics=(20.0, 40.0),
        block_seconds=30.0,
    )

    assert measured.shape == (2, 2)
    assert np.all(measured < 0.5)


def test_a_recording_shorter_than_one_block_yields_no_blocks() -> None:
    measured = block_residual_uv(
        np.zeros((2, 1000)),
        sampling_rate=1000.0,
        harmonics=(20.0,),
        block_seconds=30.0,
    )

    assert measured.shape == (2, 0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ResidualQcError, match="groups per volume"):
        slice_harmonics(
            groups_per_volume=0,
            repetition_time_seconds=0.9,
            nyquist_hz=500.0,
        )
    with pytest.raises(ResidualQcError, match="two-dimensional"):
        block_residual_uv(
            np.zeros(10),
            sampling_rate=1000.0,
            harmonics=(20.0,),
        )
