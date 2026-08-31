from pathlib import Path

import pytest

from eegfmri_fastr.cli import main
from eegfmri_fastr.compare.config import (
    ConfigurationError,
    NamingConfig,
    load_compare_config,
)
from eegfmri_fastr.compare.pairs import RecordingPair, pair_recordings, recording_key

ANALYZER_NAMING = NamingConfig(
    corrected_suffixes=("_fastr",),
    uncorrected_suffixes=(
        "_first_to_last_volume_scanner_artifact_with_pulse_markers",
        "_scanner_artifact_with_pulse_markers",
    ),
    first_run_prefixes=("BaselineEEG",),
)

HEADER = "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120)


def write_header(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.vhdr").write_text(HEADER, encoding="utf-8")


def write_compare_yaml(tmp_path: Path, naming: str = "") -> Path:
    path = tmp_path / "compare.yaml"
    path.write_text(
        f"""
paths:
  uncorrected_root: {tmp_path / "uncorrected"}
  fastr_root: {tmp_path / "fastr"}
  output_root: {tmp_path / "out"}
plot:
  psd_max_hz: 100
subjects:
  include: []
  exclude: []
{naming}""",
        encoding="utf-8",
    )
    return path


ANALYZER_NAMING_YAML = """naming:
  corrected_suffixes: ["_fastr"]
  uncorrected_suffixes:
    - "_first_to_last_volume_scanner_artifact_with_pulse_markers"
    - "_scanner_artifact_with_pulse_markers"
  first_run_prefixes: ["BaselineEEG"]
"""


# --- deriving the shared recording key --------------------------------------


def test_declared_suffixes_are_stripped_from_both_sides() -> None:
    stem = "BaselineEEG_sub0000_2026-02-09_10h56.55.966"

    assert recording_key(f"{stem}_fastr", ANALYZER_NAMING) == stem
    assert (
        recording_key(
            f"{stem}_first_to_last_volume_scanner_artifact_with_pulse_markers",
            ANALYZER_NAMING,
        )
        == stem
    )


def test_the_longest_matching_suffix_wins() -> None:
    """A specific suffix ending in a general one must not be half-stripped."""
    stem = "run01"
    key = recording_key(
        f"{stem}_first_to_last_volume_scanner_artifact_with_pulse_markers",
        ANALYZER_NAMING,
    )

    assert key == stem


def test_an_undeclared_suffix_is_left_alone() -> None:
    assert recording_key("run01_analyzer", NamingConfig()) == "run01_analyzer"


# --- pairing across two roots ------------------------------------------------


def test_declared_naming_pairs_a_flat_root_to_subject_folders(
    tmp_path: Path,
) -> None:
    write_header(tmp_path / "fastr" / "sub-0000", "BaselineEEG_sub0000_stamp_fastr")
    write_header(
        tmp_path / "uncorrected",
        "BaselineEEG_sub0000_stamp_first_to_last_volume"
        "_scanner_artifact_with_pulse_markers",
    )

    config = load_compare_config(write_compare_yaml(tmp_path, ANALYZER_NAMING_YAML))
    pairs = pair_recordings(config)

    assert len(pairs) == 1
    assert pairs[0].bids_id == "sub-0000"
    assert pairs[0].key == "BaselineEEG_sub0000_stamp"


def test_an_undeclared_uncorrected_suffix_pairs_nothing(tmp_path: Path) -> None:
    """Nothing about one lab's export naming is assumed by default."""
    write_header(tmp_path / "fastr" / "sub-0000", "BaselineEEG_sub0000_stamp_fastr")
    write_header(
        tmp_path / "uncorrected",
        "BaselineEEG_sub0000_stamp_first_to_last_volume"
        "_scanner_artifact_with_pulse_markers",
    )

    config = load_compare_config(write_compare_yaml(tmp_path))

    assert pair_recordings(config) == []


def test_an_unrelated_naming_convention_pairs_too(tmp_path: Path) -> None:
    """A different site's layout works with nothing but its own naming."""
    write_header(tmp_path / "fastr" / "P01", "task-rest_acq-01-corrected")
    write_header(tmp_path / "uncorrected" / "P01", "task-rest_acq-01-raw")
    naming = """naming:
  corrected_suffixes: ["-corrected"]
  uncorrected_suffixes: ["-raw"]
  subject_directory_prefix: "P"
"""

    config = load_compare_config(write_compare_yaml(tmp_path, naming))
    pairs = pair_recordings(config)

    assert len(pairs) == 1
    assert pairs[0].bids_id == "P01"
    assert pairs[0].key == "task-rest_acq-01"


def test_runs_are_ordered_by_the_declared_token(tmp_path: Path) -> None:
    subject = tmp_path / "fastr" / "sub-0000"
    for stem in ("task2_fastr", "task10_fastr", "task1_fastr", "Rest_fastr"):
        write_header(subject, stem)
    for stem in ("task2", "task10", "task1", "Rest"):
        write_header(tmp_path / "uncorrected", stem)
    naming = """naming:
  corrected_suffixes: ["_fastr"]
  uncorrected_suffixes: []
  first_run_prefixes: ["Rest"]
  run_index_token: "task"
"""

    config = load_compare_config(write_compare_yaml(tmp_path, naming))
    keys = [pair.key for pair in pair_recordings(config)]

    assert keys == ["Rest", "task1", "task2", "task10"]


def test_an_unknown_naming_field_is_rejected(tmp_path: Path) -> None:
    config_path = write_compare_yaml(tmp_path, "naming:\n  suffix: \"_fastr\"\n")

    with pytest.raises(ConfigurationError, match="unknown field"):
        load_compare_config(config_path)


def test_compare_help_is_registered() -> None:
    try:
        main(["compare", "--help"])
    except SystemExit as error:
        assert error.code == 0


def _raw_with_volumes(onsets: list[float], *, n_seconds: float = 5.0):
    import mne
    import numpy as np

    sfreq = 1000.0
    n = int(n_seconds * sfreq)
    rng = np.random.default_rng(0)
    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(rng.normal(0, 1e-6, (3, n)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onsets, [0.0] * len(onsets), ["Volume/V  1"] * len(onsets))
    )
    return raw


def test_align_to_fastr_accepts_shared_origin() -> None:
    from eegfmri_fastr.compare.plots import align_to_fastr

    left = _raw_with_volumes([0.0, 0.9, 1.8], n_seconds=5.0)
    right = _raw_with_volumes([0.0, 0.9, 1.8, 2.7], n_seconds=6.0)
    aligned, cropped = align_to_fastr(left, right)
    assert aligned.n_times == cropped.n_times
    assert right.n_times == 6000, "input must not be cropped in place"


def test_align_to_fastr_rejects_offset_recordings() -> None:
    import pytest

    from eegfmri_fastr.compare.plots import AlignmentError, align_to_fastr

    left = _raw_with_volumes([0.0, 0.9, 1.8])
    right = _raw_with_volumes([0.5, 1.4, 2.3])
    with pytest.raises(AlignmentError, match=r"500\.0 ms"):
        align_to_fastr(left, right)


def test_align_to_fastr_rejects_missing_volume_markers() -> None:
    import pytest

    from eegfmri_fastr.compare.plots import AlignmentError, align_to_fastr

    with pytest.raises(AlignmentError, match="volume markers are missing"):
        align_to_fastr(_raw_with_volumes([]), _raw_with_volumes([0.0]))


def test_eeg_rms_excludes_ecg() -> None:
    import numpy as np

    from eegfmri_fastr.compare.plots import eeg_rms

    raw = _raw_with_volumes([0.0])
    data = raw.get_data()
    data[raw.ch_names.index("ECG")] = 1.0  # 1 V, dwarfing the EEG channels
    raw._data = data
    assert eeg_rms(raw) < 10.0, "ECG must not leak into the EEG RMS"
    with_ecg = float(np.sqrt(np.mean(np.square(data * 1e6))))
    assert with_ecg > 1e5


def _recording_pair(tmp_path: Path) -> RecordingPair:
    return RecordingPair(
        bids_id="sub-0000",
        idx_run=1,
        key="run01",
        uncorrected_vhdr=tmp_path / "uncorrected.vhdr",
        fastr_vhdr=tmp_path / "fastr.vhdr",
    )


@pytest.mark.parametrize(
    "error",
    [OSError("missing header"), RuntimeError("invalid header"), ValueError("bad data")],
)
def test_compare_reports_expected_loader_failures(
    monkeypatch,
    tmp_path: Path,
    error: Exception,
) -> None:
    from eegfmri_fastr.compare import pipeline as compare_pipeline

    def fail(_path: Path) -> object:
        raise error

    monkeypatch.setattr(compare_pipeline, "load_vhdr", fail)

    assert compare_pipeline._load_traces(_recording_pair(tmp_path)) == {}


def test_compare_surfaces_unexpected_loader_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from eegfmri_fastr.compare import pipeline as compare_pipeline

    def fail(_path: Path) -> object:
        raise TypeError("programming error")

    monkeypatch.setattr(compare_pipeline, "load_vhdr", fail)

    with pytest.raises(TypeError, match="programming error"):
        compare_pipeline._load_traces(_recording_pair(tmp_path))
