import json
from pathlib import Path

import numpy as np
import pytest

from mri_correction.fastr import (
    FastrInputError,
    FmriAcquisitionTiming,
    acquisition_group_fastr,
    acquisition_group_fastr_with_edges,
    make_group_trigger_samples,
    slice_fastr_with_edges,
)
from mri_correction.metrics import MetricInputError, trigger_locked_rms


def make_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.9,
        slice_timing_seconds=(0.0, 0.4),
        multiband_acceleration_factor=1,
    )


def make_slot_input() -> tuple[np.ndarray, np.ndarray, FmriAcquisitionTiming]:
    timing = make_timing()
    volume_starts = np.array([100, 1000, 1900, 2800], dtype=np.int64)
    data = np.zeros((2, 4_000), dtype=np.float64)
    return data, volume_starts, timing


def test_acquisition_group_fastr_derives_slot_matching_from_bids_timing() -> None:
    data, volume_starts, timing = make_slot_input()

    correction = acquisition_group_fastr(
        data,
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
        neighbor_count=2,
        search_radius_samples=0,
    )

    neighbors = correction.provenance.neighbor_indices
    targets = np.arange(neighbors.shape[0])[:, np.newaxis]
    assert np.all(neighbors % timing.groups_per_volume == targets % 2)
    assert np.all(neighbors != targets)


def test_acquisition_group_fastr_rejects_nonmatching_volume_starts() -> None:
    data, volume_starts, timing = make_slot_input()
    volume_starts[2] += 1

    with pytest.raises(FastrInputError, match="jitter"):
        acquisition_group_fastr(
            data,
            volume_starts,
            sampling_rate=1_000.0,
            timing=timing,
            neighbor_count=2,
            search_radius_samples=0,
        )


def test_edge_wrapper_reports_and_preserves_unestimable_groups() -> None:
    triggers = np.arange(40, dtype=np.float64) * 50.0
    data = np.zeros((1, 1_990), dtype=np.float64)

    correction = slice_fastr_with_edges(
        data,
        triggers,
        interpolation_factor=1,
        neighbor_count=2,
        search_radius_samples=0,
    )

    skipped = correction.provenance.skipped_group_indices
    assert skipped.size >= 2
    assert skipped[0] == 0
    assert skipped[-1] == triggers.size - 1
    np.testing.assert_array_equal(correction.data[:, :50], data[:, :50])


def test_acquisition_group_edge_wrapper_reports_skipped_volume_edges() -> None:
    timing = make_timing()
    volume_starts = np.arange(35, dtype=np.int64) * 900
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
    )
    data = np.zeros((1, 31_500), dtype=np.float64)

    correction = acquisition_group_fastr_with_edges(
        data,
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
        neighbor_count=2,
        search_radius_samples=0,
    )

    skipped = correction.provenance.skipped_group_indices
    assert skipped[0] == 0
    assert skipped[-1] == triggers.size - 1


def test_provenance_arrays_are_read_only() -> None:
    data, volume_starts, timing = make_slot_input()
    provenance = acquisition_group_fastr(
        data,
        volume_starts,
        sampling_rate=1_000.0,
        timing=timing,
        neighbor_count=2,
        search_radius_samples=0,
    ).provenance

    with pytest.raises(ValueError):
        provenance.shifts[0] = 1


def test_trigger_locked_rms_uses_fractional_trigger_positions() -> None:
    samples = np.arange(200, dtype=np.float64)
    data = np.sin(2.0 * np.pi * samples / 20.0)[np.newaxis, :]
    triggers = np.array([20.5, 60.5, 100.5], dtype=np.float64)

    measured = trigger_locked_rms(data, triggers, epoch_samples=20)
    expected_epochs = np.stack(
        [np.interp(np.arange(20) + trigger, samples, data[0]) for trigger in triggers]
    )
    expected = np.sqrt(np.mean(expected_epochs.mean(axis=0) ** 2))

    assert measured[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "triggers",
    [np.array([1.0, np.nan]), np.array([True, False]), np.array([1.0, 1.0])],
)
def test_trigger_locked_rms_rejects_invalid_triggers(triggers: np.ndarray) -> None:
    with pytest.raises(MetricInputError):
        trigger_locked_rms(
            np.zeros((1, 100)),
            triggers,
            epoch_samples=10,
        )


def test_cli_module_can_be_imported() -> None:
    from mri_correction.cli import main

    assert callable(main)


def test_cli_validate_timing_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "timing.json"
    metadata.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.9,
                "SliceTiming": [0.0],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")

    from mri_correction.cli import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "mri-correct",
            "validate-timing",
            "--metadata",
            str(metadata),
            "--volume-starts",
            "0",
            "900",
            "--sampling-rate",
            "1000",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError):
        main()


def test_cli_reports_timing_failure_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = tmp_path / "timing.json"
    metadata.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.9,
                "SliceTiming": [0.0],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )

    from mri_correction.cli import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "mri-correct",
            "validate-timing",
            "--metadata",
            str(metadata),
            "--volume-starts",
            "0",
            "901",
            "--sampling-rate",
            "1000",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert main() == 1
    assert "volume marker timing jitter" in capsys.readouterr().err


def _write_timing_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.9,
                "SliceTiming": [0.0],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("volume_starts", "expected"),
    [
        (["0", "900", "1801"], "jitter"),
        (["0", "900", "4500"], "acquisition gap"),
    ],
)
def test_cli_reports_invalid_timing_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    volume_starts: list[str],
    expected: str,
) -> None:
    """A validation command must say what is wrong, not raise at the operator."""
    metadata = tmp_path / "timing.json"
    _write_timing_metadata(metadata)

    from mri_correction.cli import main

    status = main(
        [
            "validate-timing",
            "--metadata",
            str(metadata),
            "--volume-starts",
            *volume_starts,
            "--sampling-rate",
            "1000",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert expected in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "result.json").exists()
