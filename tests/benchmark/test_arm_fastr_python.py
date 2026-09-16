import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.arms import ArmError
from benchmark.arms.fastr_python import FastrPythonArm
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

DEMO_SETTINGS = MatchedSettings(
    marker_description="volume-start",
    reference_channel="Cz",
    neighbor_count=20,
    non_eeg_channels=("ECG",),
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("demo")
    subprocess.run(
        [
            str(Path(sys.executable).parent / "fastr-python"),
            "demo",
            "--output-dir",
            str(directory),
        ],
        check=True,
        capture_output=True,
    )
    return directory


def _run_spec(demo: Path) -> RunSpec:
    return RunSpec(
        participant="sub-demo",
        run_index=1,
        raw_vhdr=demo / "demo.vhdr",
        protocol_json=demo / "demo_bold.json",
        mean_framewise_displacement=0.1,
        motion_stratum="low",
        marker_block=None,
    )


def test_arm_corrects_the_demo_recording_and_reports_its_cost(demo, tmp_path):
    result = FastrPythonArm(DEMO_SETTINGS).correct(
        _run_spec(demo), output_directory=tmp_path / "out"
    )
    assert result.corrected_vhdr.is_file()
    assert result.sampling_rate == 1000.0
    assert result.first_volume_sample == 0
    assert result.volume_count > 0
    assert result.cost.wall_clock_seconds > 0
    assert result.cost.peak_memory_bytes > 0
    assert result.cost.exit_code == 0


def test_arm_writes_the_matched_settings_into_its_configuration(demo, tmp_path):
    FastrPythonArm(DEMO_SETTINGS).correct(
        _run_spec(demo), output_directory=tmp_path / "out"
    )
    sidecar = json.loads(
        (tmp_path / "out" / "demo_fastr.json").read_text(encoding="utf-8")
    )
    processing = sidecar["configuration"]["processing"]
    assert processing["interpolation_factor"] == 10
    assert processing["neighbor_count"] == 20
    assert processing["lowpass_hz"] == 100.0
    assert processing["output_sampling_rate_hz"] == 1000.0
    assert processing["residual_obs"] is False
    assert processing["adaptive_noise_cancellation"] is False


def test_arm_reports_whole_volumes_only(demo, tmp_path):
    result = FastrPythonArm(DEMO_SETTINGS).correct(
        _run_spec(demo), output_directory=tmp_path / "out"
    )
    sidecar = json.loads(result.corrected_vhdr.with_suffix(".json").read_text())
    samples_per_volume = round(
        sidecar["timing"]["resolved"]["repetition_time_seconds"] * result.sampling_rate
    )
    assert result.volume_count * samples_per_volume <= sidecar["output"]["sample_count"]


def test_arm_raises_when_the_recording_cannot_be_corrected(demo, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    for suffix in (".vhdr", ".vmrk", ".eeg", "_bold.json"):
        shutil.copy(demo / f"demo{suffix}", broken / f"demo{suffix}")
    (broken / "demo.vmrk").write_text("nonsense", encoding="utf-8")
    spec = RunSpec(
        participant="sub-demo",
        run_index=1,
        raw_vhdr=broken / "demo.vhdr",
        protocol_json=broken / "demo_bold.json",
        mean_framewise_displacement=0.1,
        motion_stratum="low",
        marker_block=None,
    )
    with pytest.raises(ArmError, match="failed on demo.vhdr"):
        FastrPythonArm(DEMO_SETTINGS).correct(spec, output_directory=tmp_path / "out2")
