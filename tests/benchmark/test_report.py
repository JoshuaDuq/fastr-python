import csv
import io
from pathlib import Path

import numpy as np

from benchmark.cohort import RunSpec
from benchmark.metrics import ToneResult
from benchmark.orchestrator import MeasuredArm
from benchmark.report import PEAK_MEMORY, RESIDUAL, TRANSFER, WALL_CLOCK, write_rows

RUN = RunSpec(
    participant="sub-0001",
    run_index=3,
    raw_vhdr=Path("raw.vhdr"),
    protocol_json=Path("bold.json"),
    mean_framewise_displacement=0.07,
    motion_stratum="middle",
    marker_block=None,
)


def _measured(**overrides) -> MeasuredArm:
    defaults = dict(
        arm="fastr_python",
        channel_names=("Fp1", "Cz"),
        block_residual_microvolts=np.array([[0.5, 0.6], [0.7, 0.8]]),
        block_peak_displacement=np.array([0.2, np.nan]),
        tone_transfer=(
            ToneResult(23.3, "on_volume_harmonic", "Fp1", 0.02, 3.0),
            ToneResult(23.9, "between_volume_harmonics", "Fp1", 1.01, 0.1),
        ),
        wall_clock_seconds=61.0,
        peak_memory_bytes=400_000_000,
    )
    return MeasuredArm(**(defaults | overrides))


def _read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _rows(tmp_path, *measured):
    path = tmp_path / "measurements.csv"
    write_rows(measured, RUN, path)
    return _read(path)


def test_every_row_names_its_recording_and_arm(tmp_path):
    rows = _rows(tmp_path, _measured())
    assert all(row["participant"] == "sub-0001" for row in rows)
    assert all(row["run_index"] == "3" for row in rows)
    assert all(row["motion_stratum"] == "middle" for row in rows)
    assert {row["arm"] for row in rows} == {"fastr_python"}


def test_residual_is_recorded_per_channel_per_block(tmp_path):
    rows = [r for r in _rows(tmp_path, _measured()) if r["measure"] == RESIDUAL]
    assert len(rows) == 4
    assert {(r["channel"], r["block"]) for r in rows} == {
        ("Fp1", "0"),
        ("Fp1", "1"),
        ("Cz", "0"),
        ("Cz", "1"),
    }


def test_residual_carries_the_motion_in_its_own_block(tmp_path):
    rows = [r for r in _rows(tmp_path, _measured()) if r["measure"] == RESIDUAL]
    first = [r for r in rows if r["block"] == "0"]
    second = [r for r in rows if r["block"] == "1"]
    assert all(r["framewise_displacement_mm"] == "0.2" for r in first)
    # An unestimated block is left empty rather than reported as motionless.
    assert all(r["framewise_displacement_mm"] == "" for r in second)


def test_transfer_keeps_the_placement_that_makes_it_interpretable(tmp_path):
    rows = [r for r in _rows(tmp_path, _measured()) if r["measure"] == TRANSFER]
    placements = {r["placement"]: float(r["value"]) for r in rows}
    assert placements["on_volume_harmonic"] == 0.02
    assert placements["between_volume_harmonics"] == 1.01


def test_cost_is_recorded_once_per_arm(tmp_path):
    rows = _rows(tmp_path, _measured())
    assert len([r for r in rows if r["measure"] == WALL_CLOCK]) == 1
    assert len([r for r in rows if r["measure"] == PEAK_MEMORY]) == 1


def test_a_second_arm_appends_rather_than_replacing(tmp_path):
    path = tmp_path / "measurements.csv"
    write_rows([_measured()], RUN, path)
    write_rows([_measured(arm="matlab_fmrib")], RUN, path)
    rows = _read(path)
    assert {row["arm"] for row in rows} == {"fastr_python", "matlab_fmrib"}
    assert sum(1 for row in rows if row["measure"] == WALL_CLOCK) == 2


def test_an_arm_that_failed_simply_has_no_rows(tmp_path):
    rows = _rows(tmp_path, _measured())
    assert "facetpy_slot_matched" not in {row["arm"] for row in rows}


def test_a_long_run_resumes_from_the_recordings_already_measured(tmp_path):
    """A cohort run does not fit in one sitting, and outputs cannot be rewritten."""
    import json

    from benchmark.orchestrator import completed_runs

    path = tmp_path / "outcomes.jsonl"
    assert completed_runs(path) == frozenset()
    path.write_text(
        "\n".join(
            json.dumps({"participant": p, "run_index": r, "arm": "a", "status": "ok"})
            for p, r in (("sub-0001", 4), ("sub-0001", 4), ("sub-0002", 1))
        ),
        encoding="utf-8",
    )
    assert completed_runs(path) == frozenset({("sub-0001", 4), ("sub-0002", 1)})


def test_a_failed_recording_counts_as_measured_and_is_not_retried(tmp_path):
    """Nothing is retried: a failure is a result, not an incomplete attempt."""
    import json

    from benchmark.orchestrator import completed_runs

    path = tmp_path / "outcomes.jsonl"
    path.write_text(
        json.dumps({"participant": "sub-0003", "run_index": 2, "status": "failed"}),
        encoding="utf-8",
    )
    assert ("sub-0003", 2) in completed_runs(path)


def test_naming_arms_selects_only_those(monkeypatch, tmp_path):
    """Naming arms is how a run reuses corrections it already has."""
    import argparse

    from benchmark.cli import _arms
    from benchmark.config import BenchmarkConfig, BenchmarkPaths

    config = BenchmarkConfig(
        paths=BenchmarkPaths(tmp_path, tmp_path, tmp_path, tmp_path)
    )
    arguments = argparse.Namespace(
        matlab=None, eeglab=None, facetpy=None, facetpy_source=None, arm=None
    )
    assert [arm.name for arm in _arms(config, arguments)] == ["fastr_python"]

    arguments.arm = ["fastr_python"]
    assert [arm.name for arm in _arms(config, arguments)] == ["fastr_python"]


def test_an_unknown_arm_name_is_refused_rather_than_ignored(tmp_path):
    import argparse

    import pytest

    from benchmark.cli import _arms
    from benchmark.config import BenchmarkConfig, BenchmarkPaths

    config = BenchmarkConfig(
        paths=BenchmarkPaths(tmp_path, tmp_path, tmp_path, tmp_path)
    )
    arguments = argparse.Namespace(
        matlab=None,
        eeglab=None,
        facetpy=None,
        facetpy_source=None,
        arm=["facetpy_volume_averaged"],
    )
    with pytest.raises(SystemExit, match="no such arm"):
        _arms(config, arguments)
