import subprocess
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

from benchmark.arms.matlab_fmrib import MatlabFmribArm
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

MATLAB = Path("/Applications/MATLAB_R2026a.app/bin/matlab")
EEGLAB = Path(
    "/Users/joduq24/Library/Application Support/MathWorks/MATLAB Add-Ons"
    "/Collections/EEGLAB"
)
TR = 0.9

pytestmark = pytest.mark.skipif(
    not MATLAB.is_file() or not (EEGLAB / "eeglab.m").is_file(),
    reason="MATLAB with the EEGLAB FMRIB plug-in is not installed",
)

DEMO_SETTINGS = MatchedSettings(
    marker_description="volume-start", reference_channel="Cz", neighbor_count=20
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


def _arm() -> MatlabFmribArm:
    return MatlabFmribArm(DEMO_SETTINGS, matlab=MATLAB, eeglab_root=EEGLAB)


def _spec(demo: Path) -> RunSpec:
    return RunSpec(
        participant="sub-demo",
        run_index=1,
        raw_vhdr=demo / "demo.vhdr",
        protocol_json=demo / "demo_bold.json",
        mean_framewise_displacement=0.1,
        motion_stratum="low",
        marker_block=None,
    )


def _harmonic_and_tone(
    data: np.ndarray, rate: float, volumes: int
) -> tuple[float, float]:
    start = round(2 * TR * rate)
    count = round(volumes * TR * rate)
    window = data[:, start : start + count] * 1e6
    spectrum = np.fft.rfft(window, axis=1)
    frequencies = np.fft.rfftfreq(count, 1 / rate)
    harmonics = [np.argmin(abs(frequencies - k / TR)) for k in range(1, 91)]
    harmonic = float(
        np.sqrt((np.abs(spectrum[:, harmonics]) ** 2).sum(axis=1)).mean() / count
    )
    tone = float(np.abs(spectrum[:, np.argmin(abs(frequencies - 10.5))]).mean() / count)
    return harmonic, tone


@pytest.fixture(scope="module")
def corrected(demo, tmp_path_factory):
    directory = tmp_path_factory.mktemp("fmrib")
    return _arm().correct(_spec(demo), output_directory=directory)


def test_arm_writes_a_readable_recording_and_reports_its_cost(corrected):
    assert corrected.corrected_vhdr.is_file()
    assert corrected.sampling_rate == 5000.0
    assert corrected.volume_count > 0
    assert corrected.cost.exit_code == 0
    assert corrected.cost.wall_clock_seconds > 0
    assert corrected.cost.peak_memory_bytes > 0


def test_arm_leaves_no_intermediate_binary_behind(corrected):
    assert not corrected.corrected_vhdr.with_suffix(".f32").exists()


def test_arm_suppresses_the_harmonic_comb_it_was_given_volume_triggers_for(
    corrected, demo
):
    raw = mne.io.read_raw_brainvision(demo / "demo.vhdr", preload=True, verbose="error")
    picks = [index for index, name in enumerate(raw.ch_names) if name != "ECG"]
    volumes = 36
    before, tone_before = _harmonic_and_tone(
        raw.get_data()[picks], corrected.sampling_rate, volumes
    )
    output = mne.io.read_raw_brainvision(
        corrected.corrected_vhdr, preload=True, verbose="error"
    ).get_data()[picks]
    after, tone_after = _harmonic_and_tone(output, corrected.sampling_rate, volumes)
    assert before / after > 100
    assert tone_after / tone_before > 0.85


def test_arm_reports_a_missing_matlab_rather_than_a_silent_skip(demo, tmp_path):
    arm = MatlabFmribArm(
        DEMO_SETTINGS, matlab=tmp_path / "no-matlab", eeglab_root=EEGLAB
    )
    with pytest.raises((FileNotFoundError, PermissionError, OSError)):
        arm.correct(_spec(demo), output_directory=tmp_path / "out")
