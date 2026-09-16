import json
import subprocess
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

from benchmark.arms import ArmError
from benchmark.arms.facetpy import SLOT_MATCHED, VOLUME_AVERAGED, FacetpyArm
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

INTERPRETER = Path(__file__).parents[2] / ".local" / "facetpy-env" / "bin" / "python"
TR = 0.9

pytestmark = pytest.mark.skipif(
    not INTERPRETER.is_file(), reason="the FACETpy environment is not installed"
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


def _suppression(
    raw_vhdr: Path, corrected_vhdr: Path, rate: float
) -> tuple[float, float]:
    raw = mne.io.read_raw_brainvision(raw_vhdr, preload=True, verbose="error")
    picks = [index for index, name in enumerate(raw.ch_names) if name != "ECG"]
    start, count = round(2 * TR * rate), round(36 * TR * rate)

    def score(data: np.ndarray) -> tuple[float, float]:
        window = data[picks][:, start : start + count] * 1e6
        spectrum = np.fft.rfft(window, axis=1)
        frequencies = np.fft.rfftfreq(count, 1 / rate)
        lines = [np.argmin(abs(frequencies - k / TR)) for k in range(1, 91)]
        harmonic = np.sqrt((np.abs(spectrum[:, lines]) ** 2).sum(axis=1)).mean()
        tone = np.abs(spectrum[:, np.argmin(abs(frequencies - 10.5))]).mean()
        return float(harmonic), float(tone)

    before, tone_before = score(raw.get_data())
    after, tone_after = score(
        mne.io.read_raw_brainvision(
            corrected_vhdr, preload=True, verbose="error"
        ).get_data()
    )
    return before / after, tone_after / tone_before


@pytest.fixture(scope="module", params=[SLOT_MATCHED, VOLUME_AVERAGED])
def corrected(request, demo, tmp_path_factory):
    arm = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode=request.param)
    directory = tmp_path_factory.mktemp(request.param)
    return arm, arm.correct(_spec(demo), output_directory=directory)


def test_both_configurations_correct_and_report_their_cost(corrected, demo):
    _, result = corrected
    assert result.corrected_vhdr.is_file()
    assert result.sampling_rate == 5000.0
    assert result.volume_count > 0
    assert result.cost.exit_code == 0
    assert result.cost.peak_memory_bytes > 0


def test_both_configurations_suppress_the_comb_and_keep_the_tone(corrected, demo):
    _, result = corrected
    suppression, tone = _suppression(
        demo / "demo.vhdr", result.corrected_vhdr, result.sampling_rate
    )
    assert suppression > 100
    assert tone > 0.9


def test_the_arm_names_say_which_configuration_produced_a_result():
    matched = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode=SLOT_MATCHED)
    averaged = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode=VOLUME_AVERAGED)
    assert matched.name == "facetpy_slot_matched"
    assert averaged.name == "facetpy_volume_averaged"


def test_an_unknown_configuration_is_refused_before_anything_runs(demo, tmp_path):
    arm = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode="whatever")
    with pytest.raises(ArmError, match="unknown FACETpy mode"):
        arm.correct(_spec(demo), output_directory=tmp_path)


def test_the_slot_matched_arm_is_given_one_trigger_per_acquisition_group(
    demo, tmp_path
):
    arm = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode=SLOT_MATCHED)
    arm.correct(_spec(demo), output_directory=tmp_path)
    request = json.loads(
        (tmp_path / "demo_facetpy_slot_matched_request.json").read_text()
    )
    assert len(request["triggers"]) == 18 * len(request["volume_starts"])


def test_a_failing_pipeline_is_reported_rather_than_passed_off_as_success(
    demo, tmp_path
):
    """FACETpy returns failures in its result object instead of raising."""
    arm = FacetpyArm(DEMO_SETTINGS, interpreter=INTERPRETER, mode=SLOT_MATCHED)
    arm.correct(_spec(demo), output_directory=tmp_path)
    with pytest.raises(ArmError, match="failed on demo"):
        arm.correct(_spec(demo), output_directory=tmp_path)
