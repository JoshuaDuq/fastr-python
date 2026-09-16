import json

import numpy as np
import pandas as pd
import pytest

from benchmark.analysis import (
    cost,
    load_measurements,
    load_outcomes,
    motion_sensitivity,
    robustness,
    suppression_and_transfer,
)
from benchmark.report import PEAK_MEMORY, RESIDUAL, TRANSFER, WALL_CLOCK


def _rows(arm: str, *, residual: float, on: float, between: float, blocks: int = 40):
    rng = np.random.default_rng(abs(hash(arm)) % 2**32)
    rows = []
    for offset, participant in ((0.0, "sub-0001"), (0.35, "sub-0002")):
        for block in range(blocks):
            motion = 0.05 + 0.02 * block
            rows.append(
                {
                    "participant": participant,
                    "run_index": 1,
                    "motion_stratum": "low",
                    "arm": arm,
                    "measure": RESIDUAL,
                    "channel": "Cz",
                    "block": block,
                    "value": residual + offset + 2.0 * motion + rng.normal(0, 0.01),
                    "framewise_displacement_mm": motion,
                }
            )
    for placement, value in (
        ("on_volume_harmonic", on),
        ("between_volume_harmonics", between),
    ):
        rows.append(
            {
                "participant": "sub-0001",
                "run_index": 1,
                "arm": arm,
                "measure": TRANSFER,
                "channel": "Cz",
                "frequency_hz": 23.3,
                "placement": placement,
                "value": value,
            }
        )
    rows.append(
        {
            "participant": "sub-0001",
            "run_index": 1,
            "arm": arm,
            "measure": WALL_CLOCK,
            "value": 60.0,
        }
    )
    rows.append(
        {
            "participant": "sub-0001",
            "run_index": 1,
            "arm": arm,
            "measure": PEAK_MEMORY,
            "value": 4e9,
        }
    )
    return rows


@pytest.fixture
def measurements() -> pd.DataFrame:
    return pd.DataFrame(
        _rows("fastr_python", residual=0.5, on=0.02, between=1.02)
        + _rows("facetpy_slot_matched", residual=0.2, on=0.00, between=1.00)
    )


def test_summary_keeps_the_two_tone_placements_apart(measurements):
    summary = suppression_and_transfer(measurements)
    assert summary.loc["fastr_python", "transfer_on_comb"] == pytest.approx(0.02)
    assert summary.loc["fastr_python", "transfer_between_comb"] == pytest.approx(1.02)


def test_summary_reports_suppression_beside_transfer_not_instead_of_it(measurements):
    summary = suppression_and_transfer(measurements)
    assert {"residual_uv_median", "transfer_on_comb", "transfer_between_comb"} <= set(
        summary.columns
    )
    # The arm with the lowest residual keeps the least on-comb signal here, so a
    # ranking on residual alone would invert the transfer ranking.
    assert summary.index[0] == "facetpy_slot_matched"
    assert (
        summary.loc["facetpy_slot_matched", "transfer_on_comb"]
        < summary.loc["fastr_python", "transfer_on_comb"]
    )


def test_summary_reports_the_worst_block_not_only_the_median(measurements):
    summary = suppression_and_transfer(measurements)
    assert (
        summary.loc["fastr_python", "residual_uv_worst"]
        > summary.loc["fastr_python", "residual_uv_median"]
    )


def test_motion_slope_recovers_the_relationship_that_was_put_in(measurements):
    fitted = {arm.arm: arm for arm in motion_sensitivity(measurements)}
    assert set(fitted) == {"fastr_python", "facetpy_slot_matched"}
    for arm in fitted.values():
        assert arm.slope_uv_per_mm == pytest.approx(2.0, rel=0.05)
        assert arm.participants == 2
        assert arm.converged
        assert arm.participant_variance > 0


def test_an_arm_without_enough_motion_range_is_not_fitted():
    flat = pd.DataFrame(_rows("flat", residual=0.5, on=0.0, between=1.0, blocks=3))
    assert motion_sensitivity(flat) == ()


def test_cost_reports_clock_and_memory_per_arm(measurements):
    table = cost(measurements)
    assert table.loc["fastr_python", "wall_clock_seconds_median"] == 60.0
    assert table.loc["fastr_python", "peak_memory_gb"] == pytest.approx(4.0)


def test_robustness_counts_failures_as_results(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    lines = [
        {"participant": "sub-0001", "arm": "ml_demucs", "status": "failed"},
        {"participant": "sub-0002", "arm": "ml_demucs", "status": "failed"},
        {"participant": "sub-0001", "arm": "fastr_python", "status": "ok"},
        {"participant": "sub-0002", "arm": "fastr_python", "status": "ok"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    table = robustness(load_outcomes(path))
    assert table.loc["fastr_python", "failure_rate"] == 0.0
    assert table.loc["ml_demucs", "failure_rate"] == 1.0
    assert table.loc["ml_demucs", "participants_with_a_failure"] == 2
    assert table.index[0] == "fastr_python"


def test_measurements_round_trip_from_disk(tmp_path, measurements):
    path = tmp_path / "m.csv"
    measurements.to_csv(path, index=False)
    assert len(load_measurements(path)) == len(measurements)
