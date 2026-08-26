import json
from pathlib import Path

import pytest
from test_pipeline import make_fixture

from mri_correction.cli import main


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
