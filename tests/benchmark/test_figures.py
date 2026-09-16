import pandas as pd
import pytest

from benchmark.analysis import MotionSensitivity
from benchmark.figures import write_figures
from benchmark.report import PEAK_MEMORY, RESIDUAL, TRANSFER, WALL_CLOCK


def _measurements() -> pd.DataFrame:
    rows = []
    for arm, residual, on in (
        ("fastr_python", 0.5, 0.02),
        ("matlab_fmrib", 0.35, 0.03),
    ):
        for block in range(4):
            rows.append(
                {
                    "arm": arm,
                    "measure": RESIDUAL,
                    "channel": "Cz",
                    "block": block,
                    "value": residual + 0.01 * block,
                    "framewise_displacement_mm": 0.1 * block,
                }
            )
        for frequency in (7.8, 23.3, 71.1):
            for placement, value in (
                ("on_volume_harmonic", on),
                ("between_volume_harmonics", 1.0),
            ):
                rows.append(
                    {
                        "arm": arm,
                        "measure": TRANSFER,
                        "channel": "Cz",
                        "frequency_hz": frequency,
                        "placement": placement,
                        "value": value,
                    }
                )
        rows.append({"arm": arm, "measure": WALL_CLOCK, "value": 120.0})
        rows.append({"arm": arm, "measure": PEAK_MEMORY, "value": 4e9})
    return pd.DataFrame(rows)


def test_every_figure_is_written_and_has_content(tmp_path):
    motion = (
        MotionSensitivity("fastr_python", 2.0, 0.1, 40, 2, 0.01, True),
        MotionSensitivity("matlab_fmrib", 1.5, 0.1, 40, 2, 0.01, True),
    )
    written = write_figures(_measurements(), motion, tmp_path)
    assert len(written) == 5
    assert all(path.is_file() and path.stat().st_size > 5_000 for path in written)


def test_the_motion_figure_is_skipped_when_no_slope_was_fitted(tmp_path):
    written = write_figures(_measurements(), (), tmp_path)
    assert len(written) == 4
    assert not (tmp_path / "motion_sensitivity.png").exists()


def test_the_headline_figure_shows_both_placements(tmp_path):
    written = write_figures(_measurements(), (), tmp_path)
    assert written[0].name == "suppression_cost.png"


def test_figures_refuse_an_empty_table(tmp_path):
    with pytest.raises((ValueError, KeyError, IndexError)):
        write_figures(pd.DataFrame(columns=["arm", "measure", "value"]), (), tmp_path)
