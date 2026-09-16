"""Every arm's decimated grid must start on the same acquisition."""

import numpy as np
import pytest
from pybv import write_brainvision

from benchmark.arms import ArmOutput, MeasuredRun
from benchmark.config import BenchmarkConfig, BenchmarkPaths
from benchmark.harmonise import CommonWindow
from benchmark.orchestrator import _on_grid

RATE, OUT, TR = 5000.0, 1000.0, 0.9
TONE_HZ = 23.0


def _config(tmp_path) -> BenchmarkConfig:
    return BenchmarkConfig(paths=BenchmarkPaths(tmp_path, tmp_path, tmp_path, tmp_path))


def _recording(tmp_path, name: str, first_volume: int, samples: int) -> ArmOutput:
    """Write a recording whose tone has zero phase at its first volume."""
    times = (np.arange(samples) - first_volume) / RATE
    data = np.stack([np.sin(2 * np.pi * TONE_HZ * times)] * 2)
    write_brainvision(
        data=data,
        sfreq=RATE,
        ch_names=["Cz", "Pz"],
        fname_base=name,
        folder_out=tmp_path,
        events=[],
        unit="µV",
        overwrite=True,
    )
    return ArmOutput(
        corrected_vhdr=tmp_path / f"{name}.vhdr",
        sampling_rate=RATE,
        first_volume_sample=first_volume,
        volume_count=(samples - first_volume) // round(TR * RATE),
        cost=MeasuredRun(1.0, 1, 0, ""),
    )


@pytest.mark.parametrize("first_volume", [77030, 59438, 59439])
def test_a_first_volume_off_the_decimated_grid_is_still_measurable(
    tmp_path, first_volume
):
    """Markers land where the scanner fired, not on multiples of the decimation."""
    window = CommonWindow(
        volume_count=4, guard_volumes=1, samples_per_volume=900, sampling_rate=OUT
    )
    output = _recording(tmp_path, "arm", first_volume, first_volume + 40 * 4500)
    data, names = _on_grid(output, window, _config(tmp_path))
    assert data.shape == (2, 3600)
    assert names == ["Cz", "Pz"]


def test_two_arms_starting_at_different_samples_land_in_phase(tmp_path):
    """One arm trims to its first volume and another does not; both must align."""
    window = CommonWindow(
        volume_count=4, guard_volumes=1, samples_per_volume=900, sampling_rate=OUT
    )
    config = _config(tmp_path)
    trimmed = _recording(tmp_path, "trimmed", 0, 40 * 4500)
    untrimmed = _recording(tmp_path, "untrimmed", 59438, 59438 + 40 * 4500)
    a, _ = _on_grid(trimmed, window, config)
    b, _ = _on_grid(untrimmed, window, config)
    # Same tone, same phase reference, so the two must agree sample for sample.
    assert np.abs(a - b).max() < 1e-3 * np.abs(a).max()


def test_the_grid_starts_one_guard_volume_after_the_first_volume(tmp_path):
    window = CommonWindow(
        volume_count=2, guard_volumes=1, samples_per_volume=900, sampling_rate=OUT
    )
    output = _recording(tmp_path, "phase", 59438, 59438 + 20 * 4500)
    data, _ = _on_grid(output, window, _config(tmp_path))
    # The guard volume is 0.9 s, so the tone restarts its cycle at sample 0.
    expected = np.sin(2 * np.pi * TONE_HZ * (np.arange(data.shape[1]) / OUT + TR))
    assert np.abs(data[0] / np.abs(data[0]).max() - expected).max() < 0.02
