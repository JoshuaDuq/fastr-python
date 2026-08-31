import json
from pathlib import Path

import pytest
from test_pipeline import make_fixture

from fastr_python.cli import main


def test_run_command_executes_yaml_pipeline(tmp_path: Path, capsys) -> None:
    config_path = make_fixture(tmp_path)

    assert main(["run", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["method"] == "acquisition_group_fastr"
    assert Path(summary["output_vhdr"]).is_file()


def test_validate_timing_reads_configured_brainvision_markers(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path)

    assert main(
        [
            "validate-timing",
            "--metadata",
            str(tmp_path / "bold.json"),
            "--sampling-rate",
            "1000",
            "--vhdr",
            str(tmp_path / "source.vhdr"),
            "--marker-type",
            "Volume",
            "--marker-description",
            "volume-start",
            "--output",
            str(tmp_path / "timing.json"),
        ]
    ) == 0


def test_validate_timing_reports_the_resolved_geometry(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    output = tmp_path / "timing.json"

    assert main(
        [
            "validate-timing",
            "--metadata",
            str(tmp_path / "bold.json"),
            "--sampling-rate",
            "1000",
            "--vhdr",
            str(tmp_path / "source.vhdr"),
            "--marker-type",
            "Volume",
            "--marker-description",
            "volume-start",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["marker_kind"] == "volume"
    assert report["group_position_source"] == "declared_slice_timing"
    assert report["groups_per_volume"] == 2
    assert report["repetition_time_seconds"] == pytest.approx(0.1)


def test_validate_timing_accepts_acquisition_group_markers(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path)
    output = tmp_path / "timing.json"

    assert main(
        [
            "validate-timing",
            "--marker-kind",
            "slice",
            "--groups-per-volume",
            "2",
            "--expected-repetition-time-seconds",
            "0.2",
            "--sampling-rate",
            "1000",
            "--volume-starts",
            "0",
            "100",
            "200",
            "300",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["marker_kind"] == "slice"
    assert report["group_position_source"] == "measured_group_markers"
    assert report["metadata"] is None
    assert report["repetition_time_seconds"] == pytest.approx(0.2)
    assert report["group_offsets_seconds"] == pytest.approx([0.0, 0.1])


def test_validate_timing_rejects_two_timing_sources(
    tmp_path: Path,
    capsys,
) -> None:
    make_fixture(tmp_path)

    assert main(
        [
            "validate-timing",
            "--marker-kind",
            "slice",
            "--groups-per-volume",
            "2",
            "--metadata",
            str(tmp_path / "bold.json"),
            "--sampling-rate",
            "1000",
            "--volume-starts",
            "0",
            "100",
            "--output",
            str(tmp_path / "timing.json"),
        ]
    ) == 1
    assert "drop --metadata" in capsys.readouterr().err


def test_validate_timing_requires_metadata_for_volume_markers(
    tmp_path: Path,
    capsys,
) -> None:
    assert main(
        [
            "validate-timing",
            "--sampling-rate",
            "1000",
            "--volume-starts",
            "0",
            "100",
            "--output",
            str(tmp_path / "timing.json"),
        ]
    ) == 1
    assert "--metadata is required" in capsys.readouterr().err


def test_run_command_reports_invalid_config_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text("unexpected: true\n", encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 1
    assert "unknown field" in capsys.readouterr().err


def test_help_text_is_study_independent(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    help_text = capsys.readouterr().out.lower()
    assert "participant" not in help_text
    assert "thermal" not in help_text
    assert "analyzer" not in help_text


def test_run_command_reports_output_collision_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = make_fixture(tmp_path)

    assert main(["run", "--config", str(config_path)]) == 0
    assert main(["run", "--config", str(config_path)]) == 1
    assert "already exists" in capsys.readouterr().err
