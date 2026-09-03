"""The generated demo has to be runnable, or it is not a demo."""

import json
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml

from fastr_python.cli import main
from fastr_python.config import load_config
from fastr_python.demo import write_demo_dataset
from fastr_python.pipeline import run_correction

PROBE_HZ = 10.5
PROBE_MICROVOLTS = 12.0
GROUPS_PER_VOLUME = 18


def tone_amplitude(raw: mne.io.BaseRaw, hz: float, channel: str) -> float:
    """Amplitude of one exact frequency, in microvolts."""
    data = raw.get_data(picks=[channel])[0] * 1e6
    times = np.arange(data.size) / float(raw.info["sfreq"])
    return 2.0 * abs(np.dot(data, np.exp(-2j * np.pi * hz * times))) / data.size


def test_the_demo_writes_every_file_it_names(tmp_path: Path) -> None:
    paths = write_demo_dataset(tmp_path)

    assert paths.raw_vhdr.is_file()
    assert paths.raw_vhdr.with_suffix(".eeg").is_file()
    assert paths.raw_vhdr.with_suffix(".vmrk").is_file()
    assert paths.fmri_metadata.is_file()
    assert paths.config.is_file()


def test_the_demo_refuses_to_overwrite(tmp_path: Path) -> None:
    write_demo_dataset(tmp_path)

    with pytest.raises(FileExistsError, match="demo files already exist"):
        write_demo_dataset(tmp_path)


def test_the_demo_metadata_describes_the_simulated_acquisition(
    tmp_path: Path,
) -> None:
    paths = write_demo_dataset(tmp_path)

    metadata = json.loads(paths.fmri_metadata.read_text(encoding="utf-8"))
    slice_timing = metadata["SliceTiming"]
    multiband = metadata["MultibandAccelerationFactor"]

    assert len(set(slice_timing)) == GROUPS_PER_VOLUME
    assert len(slice_timing) == GROUPS_PER_VOLUME * multiband
    assert max(slice_timing) < metadata["RepetitionTime"]


def test_the_demo_corrects_the_artifact_and_keeps_the_probe(
    tmp_path: Path,
) -> None:
    """The claim the demo makes about itself has to hold."""
    paths = write_demo_dataset(tmp_path)

    summary = run_correction(load_config(paths.config))

    before = mne.io.read_raw_brainvision(paths.raw_vhdr, preload=True, verbose="ERROR")
    after = mne.io.read_raw_brainvision(
        summary.output_vhdr, preload=True, verbose="ERROR"
    )
    artifact_before = float(before.get_data(picks=["Cz"])[0].std()) * 1e6
    artifact_after = float(after.get_data(picks=["Cz"])[0].std()) * 1e6

    assert artifact_after < 0.2 * artifact_before
    assert tone_amplitude(before, PROBE_HZ, "Cz") == pytest.approx(
        PROBE_MICROVOLTS, rel=0.05
    )
    assert tone_amplitude(after, PROBE_HZ, "Cz") == pytest.approx(
        PROBE_MICROVOLTS, rel=0.15
    )

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert provenance["residual_qc"]["flagged_block_count"] == 0


def test_the_demo_recording_also_corrects_from_group_markers(
    tmp_path: Path,
) -> None:
    """The commented slice-marker alternative in the demo config has to work."""
    paths = write_demo_dataset(tmp_path)
    document = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    del document["input"]["fmri_metadata"]
    document["timing"] = {
        "marker_type": "Slice",
        "marker_description": "slice-start",
        "marker_kind": "slice",
        "groups_per_volume": GROUPS_PER_VOLUME,
        "expected_repetition_time_seconds": 0.9,
    }
    document["output"] = {"vhdr": "from_markers.vhdr"}
    variant = tmp_path / "from_markers.yml"
    variant.write_text(yaml.safe_dump(document), encoding="utf-8")

    from_markers = run_correction(load_config(variant))
    from_sidecar = run_correction(load_config(paths.config))

    assert from_markers.output_eeg.read_bytes() == (
        from_sidecar.output_eeg.read_bytes()
    )


def test_the_demo_command_reports_where_it_wrote(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo", "--output-dir", str(tmp_path)]) == 0

    reported = json.loads(capsys.readouterr().out)
    assert Path(reported["config"]) == (tmp_path / "demo.yml").resolve()


def test_the_demo_command_reports_a_refusal_as_an_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_demo_dataset(tmp_path)

    assert main(["demo", "--output-dir", str(tmp_path)]) == 1
    assert "demo files already exist" in capsys.readouterr().err
