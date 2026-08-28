from pathlib import Path

from mri_correction.cli import main
from mri_correction.compare.config import load_compare_config
from mri_correction.compare.pairs import pair_recordings, recording_key


def test_recording_key_strips_fastr_and_step1_suffixes() -> None:
    assert (
        recording_key(
            "BaselineEEG_sub0000_2026-02-09_10h56.55.966_fastr"
        )
        == "BaselineEEG_sub0000_2026-02-09_10h56.55.966"
    )
    assert (
        recording_key(
            "BaselineEEG_sub0000_2026-02-09_10h56.55.966"
            "_first_to_last_volume_scanner_artifact_with_pulse_markers"
        )
        == "BaselineEEG_sub0000_2026-02-09_10h56.55.966"
    )


def _write_compare_yaml(tmp_path: Path) -> Path:
    fastr = tmp_path / "fastr" / "sub-0000"
    uncorrected = tmp_path / "step1"
    fastr.mkdir(parents=True)
    uncorrected.mkdir()
    (fastr / "BaselineEEG_sub0000_stamp_fastr.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    (
        uncorrected
        / (
            "BaselineEEG_sub0000_stamp_first_to_last_volume"
            "_scanner_artifact_with_pulse_markers.vhdr"
        )
    ).write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    path = tmp_path / "compare.yaml"
    path.write_text(
        f"""
paths:
  uncorrected_root: {uncorrected}
  fastr_root: {tmp_path / "fastr"}
  output_root: {tmp_path / "out"}
plot:
  channel: Cz
  epoch_start_seconds: 10
  epoch_seconds: 3
  psd_max_hz: 100
subjects:
  include: []
  exclude: []
""",
        encoding="utf-8",
    )
    return path


def test_pair_recordings_matches_flat_step1_to_nested_fastr(
    tmp_path: Path,
) -> None:
    config = load_compare_config(_write_compare_yaml(tmp_path))
    pairs = pair_recordings(config)
    assert len(pairs) == 1
    assert pairs[0].bids_id == "sub-0000"
    assert pairs[0].key == "BaselineEEG_sub0000_stamp"


def test_compare_help_is_registered() -> None:
    try:
        main(["compare", "--help"])
    except SystemExit as error:
        assert error.code == 0
