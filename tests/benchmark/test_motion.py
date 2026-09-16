from pathlib import Path

import numpy as np
import pytest

from benchmark.motion import (
    MotionError,
    block_peak_displacement,
    framewise_displacement,
    mean_framewise_displacement,
)


def _write(path: Path, values: list[str]) -> Path:
    rows = "".join(f"0.1\t{value}\n" for value in values)
    path.write_text("csf\tframewise_displacement\n" + rows, encoding="utf-8")
    return path


def test_framewise_displacement_keeps_volume_indexing(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "0.2", "0.4"])
    values = framewise_displacement(path)
    assert values.size == 3
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(0.2)


def test_mean_ignores_the_unset_first_volume(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "0.2", "0.4"])
    assert mean_framewise_displacement(path) == pytest.approx(0.3)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "c.tsv"
    path.write_text("csf\ttrans_x\n0.1\t0.2\n", encoding="utf-8")
    with pytest.raises(MotionError, match="framewise_displacement"):
        framewise_displacement(path)


def test_all_unset_raises(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "n/a"])
    with pytest.raises(MotionError, match="usable"):
        framewise_displacement(path)


def test_block_peak_takes_the_largest_volume_in_each_block(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "0.1", "9.0", "0.2", "0.3", "0.4"])
    peaks = block_peak_displacement(
        path, repetition_time_seconds=1.0, block_seconds=3.0, block_count=2
    )
    assert peaks == pytest.approx([9.0, 0.4])


def test_block_peak_reports_blocks_past_the_estimates_as_unmeasured(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "0.1", "0.2"])
    peaks = block_peak_displacement(
        path, repetition_time_seconds=1.0, block_seconds=3.0, block_count=3
    )
    assert peaks[0] == pytest.approx(0.2)
    assert np.isnan(peaks[1:]).all()


def test_block_peak_rejects_a_block_shorter_than_one_volume(tmp_path):
    path = _write(tmp_path / "c.tsv", ["n/a", "0.1"])
    with pytest.raises(MotionError, match="at least one volume"):
        block_peak_displacement(
            path, repetition_time_seconds=1.0, block_seconds=0.2, block_count=1
        )
