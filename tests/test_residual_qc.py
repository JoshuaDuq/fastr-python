import numpy as np
import pytest

import fastr_python.pipeline.quality as pipeline_quality
from fastr_python.fastr import AcquisitionGeometry
from fastr_python.quality.residuals import (
    ResidualQcError,
    block_residual_uv,
    evaluate_local_retry,
    flag_spatial_channel_blocks,
    recommend_persistent_bad_channels,
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
    offsets = np.arange(groups_per_volume) * (samples_per_volume // groups_per_volume)
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
        pipeline_quality,
        "volume_harmonic_spectrum",
        capture_volume_spectrum,
    )

    pipeline_quality._measure_residual_qc(
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

    monkeypatch.setattr(pipeline_quality, "slice_harmonics", capture_harmonics)
    monkeypatch.setattr(pipeline_quality, "block_residual_uv", capture_residuals)

    report = pipeline_quality._measure_residual_qc(
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
    from fastr_python.quality.residuals import flag_blocks

    residuals = _residuals(64, 20)
    residuals[:32, 7] = 25.0

    flagged = flag_blocks(residuals)

    assert flagged.tolist() == [index == 7 for index in range(20)]


def test_flag_blocks_ignores_a_block_elevated_on_a_single_channel() -> None:
    """One noisy electrode must not condemn all 64 channels for 30 seconds."""
    from fastr_python.quality.residuals import flag_blocks

    residuals = _residuals(64, 20)
    residuals[11, 7] = 25.0

    assert not flag_blocks(residuals).any()


def test_flag_channel_blocks_reports_an_isolated_channel_failure() -> None:
    from fastr_python.quality.residuals import flag_channel_blocks

    residuals = _residuals(64, 20)
    residuals[11, 7] = 25.0

    flagged = flag_channel_blocks(residuals)

    assert flagged.shape == residuals.shape
    assert flagged[11, 7]
    assert flagged.sum() == 1


def test_flag_channel_blocks_returns_empty_width_without_calibration() -> None:
    from fastr_python.quality.residuals import flag_channel_blocks

    flagged = flag_channel_blocks(np.full((64, 2), 50.0))

    assert flagged.shape == (64, 2)
    assert not flagged.any()


def test_flag_blocks_ignores_a_uniformly_elevated_recording() -> None:
    """A high but flat residual is the recording's baseline, not an event."""
    from fastr_python.quality.residuals import flag_blocks

    residuals = _residuals(64, 20, baseline=18.0)

    assert not flag_blocks(residuals).any()


def test_flag_blocks_respects_the_absolute_floor() -> None:
    """A statistical outlier still below the floor is not worth flagging."""
    from fastr_python.quality.residuals import flag_blocks

    residuals = _residuals(64, 20, baseline=0.01)
    residuals[:32, 7] = 0.2

    assert not flag_blocks(residuals, floor_uv=1.0).any()
    assert flag_blocks(residuals, floor_uv=0.1)[7]


def test_flag_blocks_returns_nothing_when_too_few_blocks_to_calibrate() -> None:
    from fastr_python.quality.residuals import flag_blocks

    residuals = np.full((64, 2), 50.0)

    assert not flag_blocks(residuals).any()


def test_flag_blocks_handles_an_empty_measurement() -> None:
    from fastr_python.quality.residuals import flag_blocks

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
    from fastr_python.pipeline.markers import residual_qc_markers

    markers = residual_qc_markers(
        _qc_report([False, True, False]),
        output_rate=1000.0,
        output_sample_count=90_000,
    )

    assert len(markers) == 1
    label = f"{markers[0].marker_type}/{markers[0].description}"
    assert not label.lower().startswith("bad")


def test_residual_qc_marker_spans_the_flagged_block() -> None:
    from fastr_python.pipeline.markers import residual_qc_markers

    markers = residual_qc_markers(
        _qc_report([False, True, False]),
        output_rate=1000.0,
        output_sample_count=90_000,
    )

    assert markers[0].position == 30_001
    assert markers[0].size == 30_000


def test_residual_qc_markers_follow_the_precomputed_flags() -> None:
    from fastr_python.pipeline.markers import residual_qc_markers

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
        pipeline_quality,
        "volume_harmonic_spectrum",
        capture_volume_spectrum,
    )

    pipeline_quality._measure_residual_qc(
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

    monkeypatch.setattr(pipeline_quality, "block_residual_uv", capture_residuals)

    pipeline_quality._measure_residual_qc(
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

    monkeypatch.setattr(pipeline_quality, "block_residual_uv", two_flagged_blocks)

    report = pipeline_quality._measure_residual_qc(
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


def test_sidecar_reports_isolated_channel_blocks_without_spatial_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def isolated_failure(data, *, sampling_rate, harmonics, block_seconds):
        residuals = np.full((data.shape[0], 5), 0.2)
        residuals[1, 3] = 40.0
        return residuals

    monkeypatch.setattr(pipeline_quality, "block_residual_uv", isolated_failure)

    report = pipeline_quality._measure_residual_qc(
        np.zeros((3, 3000)),
        channel_names=["EEG 001", "EEG 002", "ECG"],
        non_eeg_indices=frozenset({2}),
        output_rate=1000.0,
        acquisition=_acquisition(0.9),
        threshold_uv=1.0,
        block_seconds=30.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
        volume_spectrum_max_hz=110.0,
        report_channel_outliers=True,
    )

    assert report["flagged_blocks"] == [False] * 5
    assert report["flagged_channel_blocks_by_channel"] == {"EEG 002": [3]}
    assert report["flagged_channel_block_count"] == 1


def test_sidecar_omits_channel_flags_when_reporting_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def isolated_failure(data, *, sampling_rate, harmonics, block_seconds):
        residuals = np.full((data.shape[0], 5), 0.2)
        residuals[1, 3] = 40.0
        return residuals

    monkeypatch.setattr(pipeline_quality, "block_residual_uv", isolated_failure)

    report = pipeline_quality._measure_residual_qc(
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
        report_channel_outliers=False,
    )

    assert report["report_channel_outliers"] is False
    assert report["flagged_channel_blocks_by_channel"] == {}
    assert report["flagged_channel_block_count"] == 0


def test_spatial_channel_flags_require_absolute_and_robust_excess() -> None:
    residuals = np.full((5, 4), 0.5)
    residuals[1, 2] = 20.0

    result = flag_spatial_channel_blocks(
        residuals,
        eeg_channels=(0, 1, 2, 3),
        absolute_floor_uv=5.0,
        mad_multiplier=6.0,
    )

    np.testing.assert_array_equal(result.thresholds_uv, np.full(4, 5.0))
    assert np.argwhere(result.flags).tolist() == [[1, 2]]
    assert not result.flags[4].any()


def test_uniformly_high_channels_are_not_isolated_spatial_failures() -> None:
    residuals = np.full((4, 3), 20.0)

    result = flag_spatial_channel_blocks(
        residuals,
        eeg_channels=(0, 1, 2, 3),
        absolute_floor_uv=5.0,
        mad_multiplier=6.0,
    )

    assert not result.flags.any()


def test_spatial_flags_need_three_calibration_blocks() -> None:
    residuals = np.full((4, 2), 20.0)

    result = flag_spatial_channel_blocks(
        residuals,
        eeg_channels=(0, 1, 2, 3),
        absolute_floor_uv=5.0,
        mad_multiplier=6.0,
    )

    np.testing.assert_array_equal(result.thresholds_uv, np.full(2, 5.0))
    assert not result.flags.any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eeg_channels": ()}, "at least one EEG channel"),
        ({"eeg_channels": (0, 0, 1)}, "unique"),
        ({"eeg_channels": (0, 9)}, "within the recording"),
        ({"eeg_channels": (0, -1)}, "within the recording"),
        ({"absolute_floor_uv": 0.0}, "absolute floor"),
        ({"absolute_floor_uv": float("nan")}, "absolute floor"),
        ({"mad_multiplier": -1.0}, "MAD multiplier"),
    ],
)
def test_spatial_flags_reject_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "eeg_channels": (0, 1, 2),
        "absolute_floor_uv": 5.0,
        "mad_multiplier": 6.0,
    }
    arguments.update(kwargs)

    with pytest.raises(ResidualQcError, match=message):
        flag_spatial_channel_blocks(np.zeros((4, 5)), **arguments)


def test_spatial_flags_reject_a_non_finite_matrix() -> None:
    residuals = np.full((4, 5), 1.0)
    residuals[0, 0] = np.inf

    with pytest.raises(ResidualQcError, match="finite"):
        flag_spatial_channel_blocks(
            residuals,
            eeg_channels=(0, 1, 2, 3),
            absolute_floor_uv=5.0,
            mad_multiplier=6.0,
        )


def test_local_retry_requires_fewer_failures_and_fifteen_percent_improvement() -> None:
    thresholds = np.full(4, 5.0)
    wide = np.array([1.0, 20.0, 15.0, 1.0])
    local = np.array([1.0, 3.0, 4.0, 1.0])

    result = evaluate_local_retry(wide, local, thresholds)

    assert result.accepted is True
    assert result.reason == "fewer_failed_blocks_and_lower_maximum"
    assert result.wide_failed_blocks.tolist() == [1, 2]
    assert result.local_failed_blocks.tolist() == []
    assert result.wide_maximum_uv == 20.0
    assert result.local_maximum_uv == 4.0


def test_local_retry_is_rejected_when_the_failed_block_count_is_unchanged() -> None:
    thresholds = np.full(4, 5.0)
    wide = np.array([1.0, 20.0, 15.0, 1.0])
    local = np.array([1.0, 6.0, 5.5, 1.0])

    result = evaluate_local_retry(wide, local, thresholds)

    assert result.accepted is False
    assert result.reason == "failed_block_count_not_reduced"
    assert result.local_failed_blocks.tolist() == [1, 2]


def test_local_retry_is_rejected_without_a_lower_maximum() -> None:
    """One block fixed while the worst one barely moves is not an improvement."""
    thresholds = np.full(4, 5.0)
    wide = np.array([1.0, 20.0, 6.0, 1.0])
    local = np.array([1.0, 18.0, 4.0, 1.0])

    result = evaluate_local_retry(wide, local, thresholds)

    assert result.accepted is False
    assert result.reason == "maximum_residual_not_reduced_by_fifteen_percent"
    assert result.wide_failed_blocks.tolist() == [1, 2]
    assert result.local_failed_blocks.tolist() == [1]


def test_local_retry_rejects_mismatched_or_non_finite_vectors() -> None:
    with pytest.raises(ResidualQcError, match="same length"):
        evaluate_local_retry(
            np.zeros(4),
            np.zeros(4),
            np.zeros(3),
        )
    with pytest.raises(ResidualQcError, match="one-dimensional"):
        evaluate_local_retry(
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            np.zeros(4),
        )
    with pytest.raises(ResidualQcError, match="finite"):
        evaluate_local_retry(
            np.full(4, np.nan),
            np.zeros(4),
            np.zeros(4),
        )


def test_persistent_failures_require_two_blocks_and_ten_percent() -> None:
    flags = np.zeros((3, 20), dtype=bool)
    flags[0, :2] = True
    flags[1, 0] = True

    recommended = recommend_persistent_bad_channels(flags)

    assert recommended.tolist() == [True, False, False]


def test_persistent_failures_scale_with_the_recording_length() -> None:
    """Two blocks out of a hundred is a moment, not a broken electrode."""
    flags = np.zeros((2, 100), dtype=bool)
    flags[0, :10] = True
    flags[1, :9] = True

    recommended = recommend_persistent_bad_channels(flags)

    assert recommended.tolist() == [True, False]


def test_persistent_failures_need_three_calibration_blocks() -> None:
    flags = np.ones((2, 2), dtype=bool)

    recommended = recommend_persistent_bad_channels(flags)

    assert recommended.tolist() == [False, False]


def test_block_residual_measurement_matches_the_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.arange(15, dtype=float).reshape(3, 5)
    monkeypatch.setattr(
        pipeline_quality,
        "block_residual_uv",
        lambda *args, **kwargs: expected,
    )

    measurement = pipeline_quality._measure_block_residuals(
        np.zeros((3, 3000)),
        output_rate=1000.0,
        acquisition=_acquisition(0.9),
        block_seconds=30.0,
        mains_frequency_hz=60.0,
        mains_exclusion_hz=1.0,
    )

    np.testing.assert_array_equal(measurement.residuals_uv, expected)
    assert measurement.volumes_per_block == 33
    assert measurement.block_seconds == pytest.approx(29.7)
    assert measurement.harmonics_hz == slice_harmonics(
        groups_per_volume=2,
        repetition_time_seconds=0.9,
        nyquist_hz=500.0,
        mains_hz=60.0,
        exclusion_hz=1.0,
    )
