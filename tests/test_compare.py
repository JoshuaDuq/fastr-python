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
    from mri_correction.compare.plots import align_to_fastr

    left = _raw_with_volumes([0.0, 0.9, 1.8], n_seconds=5.0)
    right = _raw_with_volumes([0.0, 0.9, 1.8, 2.7], n_seconds=6.0)
    aligned, cropped = align_to_fastr(left, right)
    assert aligned.n_times == cropped.n_times
    assert right.n_times == 6000, "input must not be cropped in place"


def test_align_to_fastr_rejects_offset_recordings() -> None:
    import pytest

    from mri_correction.compare.plots import AlignmentError, align_to_fastr

    left = _raw_with_volumes([0.0, 0.9, 1.8])
    right = _raw_with_volumes([0.5, 1.4, 2.3])
    with pytest.raises(AlignmentError, match=r"500\.0 ms"):
        align_to_fastr(left, right)


def test_align_to_fastr_rejects_missing_volume_markers() -> None:
    import pytest

    from mri_correction.compare.plots import AlignmentError, align_to_fastr

    with pytest.raises(AlignmentError, match="volume markers are missing"):
        align_to_fastr(_raw_with_volumes([]), _raw_with_volumes([0.0]))


def test_eeg_rms_excludes_ecg() -> None:
    import numpy as np

    from mri_correction.compare.plots import eeg_rms

    raw = _raw_with_volumes([0.0])
    data = raw.get_data()
    data[raw.ch_names.index("ECG")] = 1.0  # 1 V, dwarfing the EEG channels
    raw._data = data
    assert eeg_rms(raw) < 10.0, "ECG must not leak into the EEG RMS"
    with_ecg = float(np.sqrt(np.mean(np.square(data * 1e6))))
    assert with_ecg > 1e5
