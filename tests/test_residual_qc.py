import numpy as np
import pytest

import eegfmri_fastr.pipeline as pipeline_module
from eegfmri_fastr.fastr import AcquisitionGeometry
from eegfmri_fastr.residual_qc import (
    ResidualQcError,
    block_residual_uv,
    slice_harmonics,
    volume_harmonic_spectrum,
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


def test_volume_harmonic_spectrum_separates_exact_bin_and_sideband() -> None:
    sampling_rate = 1_000.0
    times = np.arange(20_000) / sampling_rate
    exact = np.sin(2 * np.pi * 10.0 * times)
    sideband = 3.0 * np.sin(2 * np.pi * 10.1 * times)

    profile = volume_harmonic_spectrum(
        np.vstack([exact + sideband]),
        sampling_rate=sampling_rate,
        repetition_time_seconds=0.1,
        maximum_frequency_hz=11.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
    )

    assert len(profile) == 1
    assert profile[0].order == 1
    assert profile[0].frequency_hz == pytest.approx(10.0)
    assert profile[0].local_peak_frequency_hz == pytest.approx(10.1)
    assert profile[0].local_peak_power_db > profile[0].exact_power_db
    assert profile[0].mains_collision is False


def test_volume_harmonic_spectrum_marks_a_mains_collision() -> None:
    sampling_rate = 1_000.0
    times = np.arange(20_000) / sampling_rate
    data = np.sin(2 * np.pi * 60.0 * times)[np.newaxis, :]

    profile = volume_harmonic_spectrum(
        data,
        sampling_rate=sampling_rate,
        repetition_time_seconds=0.1,
        maximum_frequency_hz=61.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
    )

    assert profile[5].frequency_hz == pytest.approx(60.0)
    assert profile[5].mains_collision is True


def test_volume_harmonic_spectrum_is_robust_to_one_channel_outlier() -> None:
    sampling_rate = 1_000.0
    times = np.arange(20_000) / sampling_rate
    exact = np.sin(2 * np.pi * 10.0 * times)
    outlier = 100.0 * np.sin(2 * np.pi * 10.1 * times)

    profile = volume_harmonic_spectrum(
        np.vstack([exact, exact, outlier]),
        sampling_rate=sampling_rate,
        repetition_time_seconds=0.1,
        maximum_frequency_hz=11.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
    )

    assert profile[0].local_peak_frequency_hz == pytest.approx(10.0)


def _acquisition(
    repetition_time: float = 0.9,
    *,
    groups_per_volume: int = 2,
    volume_count: int = 4,
    sampling_rate: float = 1000.0,
) -> AcquisitionGeometry:
    """An evenly spaced geometry with the timing the measurement reads."""
    samples_per_volume = round(repetition_time * sampling_rate)
    offsets = np.arange(groups_per_volume) * (
        samples_per_volume // groups_per_volume
    )
    volume_starts = np.arange(volume_count, dtype=np.int64) * samples_per_volume
    triggers = (volume_starts[:, np.newaxis] + offsets).reshape(-1)
    return AcquisitionGeometry(
        volume_starts=volume_starts,
        group_triggers=triggers.astype(np.float64),
        repetition_time_seconds=repetition_time,
        groups_per_volume=groups_per_volume,
        group_offsets_seconds=tuple(offsets / sampling_rate),
        source="declared_slice_timing",
    )


def test_pipeline_volume_harmonic_spectrum_excludes_ecg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = _acquisition(0.1, sampling_rate=500.0)
    seen: dict[str, object] = {}

    def capture_volume_spectrum(data: np.ndarray, **kwargs: object) -> tuple:
        seen["shape"] = data.shape
        return ()

    monkeypatch.setattr(
        pipeline_module,
        "volume_harmonic_spectrum",
        capture_volume_spectrum,
    )

    pipeline_module._measure_residual_qc(
        np.zeros((3, 200)),
        channel_names=["EEG 001", "EEG 002", "ECG"],
        non_eeg_indices=frozenset({2}),
        output_rate=500.0,
        acquisition=acquisition,
        threshold_uv=1.0,
        block_seconds=12.0,
        mains_frequency_hz=50.0,
        mains_exclusion_hz=0.5,
        volume_spectrum_max_hz=110.0,
    )

    assert seen["shape"] == (2, 200)


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
    acquisition = _acquisition(0.1, sampling_rate=500.0)
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
        np.zeros((2, 200)),
        channel_names=["EEG 001", "EEG 002"],
        non_eeg_indices=frozenset(),
        output_rate=500.0,
        acquisition=acquisition,
        threshold_uv=1.0,
        block_seconds=12.0,
        mains_frequency_hz=50.0,
        mains_exclusion_hz=0.5,
        volume_spectrum_max_hz=110.0,
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
    assert report["volume_harmonic_spectrum"]
    assert report["volume_harmonic_spectrum"][0]["order"] == 1


# --- flag_blocks: relative, spatially consistent block flagging -------------


def _residuals(channels: int, blocks: int, baseline: float = 0.4) -> np.ndarray:
    """A calm recording: every channel sits near ``baseline`` in every block."""
    rng = np.random.default_rng(0)
    return baseline + rng.normal(0.0, 0.01, (channels, blocks))


def test_flag_blocks_flags_a_block_elevated_across_many_channels() -> None:
    from eegfmri_fastr.residual_qc import flag_blocks

    residuals = _residuals(64, 20)
    residuals[:32, 7] = 25.0

    flagged = flag_blocks(residuals)

    assert flagged.tolist() == [index == 7 for index in range(20)]


def test_flag_blocks_ignores_a_block_elevated_on_a_single_channel() -> None:
    """One noisy electrode must not condemn all 64 channels for 30 seconds."""
    from eegfmri_fastr.residual_qc import flag_blocks

    residuals = _residuals(64, 20)
    residuals[11, 7] = 25.0

    assert not flag_blocks(residuals).any()


def test_flag_blocks_ignores_a_uniformly_elevated_recording() -> None:
    """A high but flat residual is the recording's baseline, not an event."""
    from eegfmri_fastr.residual_qc import flag_blocks

    residuals = _residuals(64, 20, baseline=18.0)

    assert not flag_blocks(residuals).any()


def test_flag_blocks_respects_the_absolute_floor() -> None:
    """A statistical outlier still below the floor is not worth flagging."""
    from eegfmri_fastr.residual_qc import flag_blocks

    residuals = _residuals(64, 20, baseline=0.01)
    residuals[:32, 7] = 0.2

    assert not flag_blocks(residuals, floor_uv=1.0).any()
    assert flag_blocks(residuals, floor_uv=0.1)[7]


def test_flag_blocks_returns_nothing_when_too_few_blocks_to_calibrate() -> None:
    from eegfmri_fastr.residual_qc import flag_blocks

    residuals = np.full((64, 2), 50.0)

    assert not flag_blocks(residuals).any()


def test_flag_blocks_handles_an_empty_measurement() -> None:
    from eegfmri_fastr.residual_qc import flag_blocks

    assert flag_blocks(np.empty((64, 0))).shape == (0,)


# --- residual QC annotations must not masquerade as unusable data ----------


def _qc_report(flagged: list[bool]) -> dict[str, object]:
    return {
        "block_seconds": 30.0,
        "flagged_blocks": flagged,
        "block_residual_uv": [[0.0] * len(flagged)],
    }


def test_residual_qc_marker_is_not_rejected_by_mne_as_bad_data() -> None:
    """MNE drops any annotation whose "type/description" starts with "bad"."""
    from eegfmri_fastr.pipeline_markers import residual_qc_markers

    markers = residual_qc_markers(
        _qc_report([False, True, False]),
        output_rate=1000.0,
        output_sample_count=90_000,
    )

    assert len(markers) == 1
    label = f"{markers[0].marker_type}/{markers[0].description}"
    assert not label.lower().startswith("bad")


def test_residual_qc_marker_spans_the_flagged_block() -> None:
    from eegfmri_fastr.pipeline_markers import residual_qc_markers

    markers = residual_qc_markers(
        _qc_report([False, True, False]),
        output_rate=1000.0,
        output_sample_count=90_000,
    )

    assert markers[0].position == 30_001
    assert markers[0].size == 30_000


def test_residual_qc_markers_follow_the_precomputed_flags() -> None:
    from eegfmri_fastr.pipeline_markers import residual_qc_markers

    markers = residual_qc_markers(
        _qc_report([True, False, True]),
        output_rate=1000.0,
        output_sample_count=90_000,
    )

    assert [marker.position for marker in markers] == [1, 60_001]


# --- block boundaries and the sidecar contract -----------------------------


def test_the_volume_spectrum_limit_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported harmonics stop where the configuration says, not at 110 Hz."""
    seen: dict[str, object] = {}

    def capture_volume_spectrum(data: np.ndarray, **kwargs: object) -> tuple:
        seen.update(kwargs)
        return ()

    monkeypatch.setattr(
        pipeline_module,
        "volume_harmonic_spectrum",
        capture_volume_spectrum,
    )

    pipeline_module._measure_residual_qc(
        np.zeros((2, 3000)),
        channel_names=["EEG 001", "EEG 002"],
        non_eeg_indices=frozenset(),
        output_rate=1000.0,
        acquisition=_acquisition(0.9),
        threshold_uv=1.0,
        block_seconds=30.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
        volume_spectrum_max_hz=45.0,
    )

    assert seen["maximum_frequency_hz"] == 45.0



def test_blocks_are_rounded_to_whole_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block boundary mid-volume splits one acquisition across two blocks."""
    seen: dict[str, object] = {}

    def capture_residuals(data, *, sampling_rate, harmonics, block_seconds):
        seen["block_seconds"] = block_seconds
        return np.empty((data.shape[0], 0), dtype=np.float64)

    monkeypatch.setattr(pipeline_module, "block_residual_uv", capture_residuals)

    pipeline_module._measure_residual_qc(
        np.zeros((2, 3000)),
        channel_names=["EEG 001", "EEG 002"],
        non_eeg_indices=frozenset(),
        output_rate=1000.0,
        acquisition=_acquisition(0.9),
        threshold_uv=1.0,
        block_seconds=30.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
        volume_spectrum_max_hz=110.0,
    )

    block_seconds = float(seen["block_seconds"])
    assert block_seconds == pytest.approx(29.7)  # 33 volumes of 0.9 s
    assert (block_seconds / 0.9) == pytest.approx(round(block_seconds / 0.9))


def test_sidecar_reports_the_flag_decision_and_its_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def two_flagged_blocks(data, *, sampling_rate, harmonics, block_seconds):
        residuals = np.full((data.shape[0], 5), 0.2)
        residuals[:, 3] = 40.0
        return residuals

    monkeypatch.setattr(pipeline_module, "block_residual_uv", two_flagged_blocks)

    report = pipeline_module._measure_residual_qc(
        np.zeros((8, 3000)),
        channel_names=[f"EEG {index:03d}" for index in range(8)],
        non_eeg_indices=frozenset(),
        output_rate=1000.0,
        acquisition=_acquisition(0.9),
        threshold_uv=1.0,
        block_seconds=30.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
        volume_spectrum_max_hz=110.0,
    )

    assert report["flagged_blocks"] == [False, False, False, True, False]
    assert report["flagged_block_count"] == 1
    assert report["floor_uv"] == 1.0
    assert "mad_multiplier" in report
    assert "minimum_channels" in report
