import numpy as np
import pytest

import mri_correction.pipeline as pipeline_module
from mri_correction.fastr import FmriAcquisitionTiming
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


def test_pipeline_residual_qc_forwards_configured_measurement_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timing = FmriAcquisitionTiming(
        repetition_time_seconds=0.1,
        slice_timing_seconds=(0.0, 0.05),
        multiband_acceleration_factor=1,
    )
    seen: dict[str, object] = {}

    def capture_harmonics(**kwargs: object) -> tuple[float, ...]:
        seen.update(kwargs)
        return (20.0,)

    def capture_residuals(
        data: np.ndarray,
        *,
        sampling_rate: float,
        harmonics: tuple[float, ...],
        block_seconds: float,
    ) -> np.ndarray:
        seen["block_seconds"] = block_seconds
        seen["harmonics"] = harmonics
        return np.empty((data.shape[0], 0), dtype=np.float64)

    monkeypatch.setattr(pipeline_module, "slice_harmonics", capture_harmonics)
    monkeypatch.setattr(pipeline_module, "block_residual_uv", capture_residuals)

    report = pipeline_module._measure_residual_qc(
        np.zeros((2, 100)),
        channel_names=["EEG 001", "EEG 002"],
        output_rate=500.0,
        timing=timing,
        threshold_uv=1.0,
        block_seconds=12.0,
        mains_frequency_hz=50.0,
        mains_exclusion_hz=0.5,
    )

    assert seen == {
        "groups_per_volume": 2,
        "repetition_time_seconds": 0.1,
        "nyquist_hz": 250.0,
        "mains_hz": 50.0,
        "exclusion_hz": 0.5,
        "block_seconds": 12.0,
        "harmonics": (20.0,),
    }
    assert report["block_seconds"] == 12.0
    assert report["mains_frequency_hz"] == 50.0
    assert report["mains_exclusion_hz"] == 0.5
