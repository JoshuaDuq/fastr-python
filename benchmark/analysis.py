"""Turn the tidy table into the five answers the benchmark was built to give.

Every summary here reports suppression and transfer together, and reports the
two tone placements apart. A single averaged transfer figure would sit halfway
between a frequency the correction is blind to and one it leaves alone, and
would describe neither.

Motion is fitted at block scale with a random intercept per participant. Across
this cohort run means span 0.05 to 0.33 mm where block peaks reach 8 mm, so the
run scale has no range to fit a slope against, and blocks from one participant
are not independent of each other.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    SingularMatrixWarning,
)

from benchmark.probes import BETWEEN_HARMONICS, ON_HARMONIC
from benchmark.report import (
    PEAK_MEMORY,
    RESIDUAL,
    TRANSFER,
    WALL_CLOCK,
)

# A slope needs blocks that actually differ in how much the head moved.
MINIMUM_MOTION_RANGE_MM = 0.1
MINIMUM_BLOCKS = 30


@dataclass(frozen=True, slots=True)
class MotionSensitivity:
    """How much an arm's residual grows with head motion."""

    arm: str
    slope_uv_per_mm: float
    standard_error: float
    blocks: int
    participants: int
    participant_variance: float
    converged: bool


def load_measurements(path: Path) -> pd.DataFrame:
    """Read the tidy table one benchmark run wrote."""
    return pd.read_csv(path)


def load_outcomes(path: Path) -> pd.DataFrame:
    """Read every attempt, including the ones that failed."""
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame([json.loads(line) for line in handle if line.strip()])


def suppression_and_transfer(measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarise each arm by what it removed and what it kept.

    The two placements stay in separate columns because they answer different
    questions: what happens to signal the correction cannot distinguish from
    artifact, and what happens to everything else.
    """
    residual = _measure(measurements, RESIDUAL)
    transfer = _measure(measurements, TRANSFER)
    summary = pd.DataFrame(
        {
            "residual_uv_median": residual.groupby("arm")["value"].median(),
            "residual_uv_iqr": residual.groupby("arm")["value"].apply(_iqr),
            "residual_uv_worst": residual.groupby("arm")["value"].max(),
            "transfer_on_comb": _placement(transfer, ON_HARMONIC),
            "transfer_between_comb": _placement(transfer, BETWEEN_HARMONICS),
        }
    )
    return summary.sort_values("residual_uv_median")


def cost(measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarise what each arm cost to run, in clock time and memory."""
    wall = _measure(measurements, WALL_CLOCK).groupby("arm")["value"]
    memory = _measure(measurements, PEAK_MEMORY).groupby("arm")["value"]
    return pd.DataFrame(
        {
            "wall_clock_seconds_median": wall.median(),
            "wall_clock_seconds_worst": wall.max(),
            "peak_memory_gb": memory.max() / 1e9,
            "recordings": wall.count(),
        }
    ).sort_values("wall_clock_seconds_median")


def motion_sensitivity(measurements: pd.DataFrame) -> tuple[MotionSensitivity, ...]:
    """Fit each arm's residual against the motion in the same block.

    A random intercept per participant, because blocks from one head are not
    independent observations of how a correction handles movement.
    """
    residual = _measure(measurements, RESIDUAL).dropna(
        subset=["framewise_displacement_mm"]
    )
    fitted: list[MotionSensitivity] = []
    for arm, rows in residual.groupby("arm"):
        blocks = rows.groupby(
            ["participant", "run_index", "block"], as_index=False
        ).agg(
            value=("value", "median"),
            framewise_displacement_mm=("framewise_displacement_mm", "first"),
        )
        if not _fittable(blocks):
            continue
        fitted.append(_fit_one(str(arm), blocks))
    return tuple(fitted)


def robustness(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Count how often each arm ran, and how often it did not.

    A failure is a measurement, not a gap. An arm that corrects beautifully on
    the recordings it survives is not a usable arm if it refuses a third of the
    cohort, and this is the table that says so.
    """
    attempts = outcomes.groupby("arm")["status"]
    summary = pd.DataFrame(
        {
            "attempts": attempts.count(),
            "failed": attempts.apply(lambda status: (status == "failed").sum()),
        }
    )
    summary["failure_rate"] = summary["failed"] / summary["attempts"]
    summary["participants_with_a_failure"] = (
        outcomes[outcomes["status"] == "failed"].groupby("arm")["participant"].nunique()
    )
    return summary.fillna({"participants_with_a_failure": 0}).sort_values(
        "failure_rate"
    )


def _fit_one(arm: str, blocks: pd.DataFrame) -> MotionSensitivity:
    """Fit one arm's residual against block motion, with participant intercepts.

    A singular random-effects covariance is reported rather than raised. It
    means the participants did not differ in their own right, which is a result
    about the cohort and leaves the slope itself estimable; the variance is
    returned so a reader can see the intercept did nothing.
    """
    model = smf.mixedlm(
        "value ~ framewise_displacement_mm", blocks, groups=blocks["participant"]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SingularMatrixWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = model.fit(reml=True, method="lbfgs")
    except (ValueError, np.linalg.LinAlgError):
        return MotionSensitivity(
            arm, float("nan"), float("nan"), len(blocks), 0, float("nan"), False
        )
    return MotionSensitivity(
        arm=arm,
        slope_uv_per_mm=float(result.params["framewise_displacement_mm"]),
        standard_error=float(result.bse["framewise_displacement_mm"]),
        blocks=len(blocks),
        participants=int(blocks["participant"].nunique()),
        participant_variance=float(result.cov_re.iloc[0, 0]),
        converged=bool(result.converged),
    )


def _fittable(blocks: pd.DataFrame) -> bool:
    """Report whether these blocks span enough motion to fit a slope through."""
    motion = blocks["framewise_displacement_mm"]
    return (
        len(blocks) >= MINIMUM_BLOCKS
        and float(motion.max() - motion.min()) >= MINIMUM_MOTION_RANGE_MM
    )


def _measure(measurements: pd.DataFrame, name: str) -> pd.DataFrame:
    return measurements[measurements["measure"] == name]


def _placement(transfer: pd.DataFrame, placement: str) -> pd.Series:
    return transfer[transfer["placement"] == placement].groupby("arm")["value"].median()


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))
