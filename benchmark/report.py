"""Write what was measured as one tidy table, one row per thing measured.

Long rather than wide: a row names its arm, its recording, and the one quantity
it carries. Nothing downstream has to know how many arms ran, and an arm that
failed on one recording simply has no rows for it rather than a column of
holes.

Residual is recorded per block against the head motion in that same block,
because the motion axis is fitted at block scale: run means over this cohort
span 0.05 to 0.33 mm where block peaks reach 8 mm. Transfer is recorded per
tone and per channel, keeping the placement that separates a frequency sitting
on a scanner harmonic from one sitting between two.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmark.cohort import RunSpec

if TYPE_CHECKING:
    from benchmark.orchestrator import MeasuredArm

FIELDS = (
    "participant",
    "run_index",
    "motion_stratum",
    "arm",
    "measure",
    "channel",
    "block",
    "frequency_hz",
    "placement",
    "value",
    "framewise_displacement_mm",
)

RESIDUAL = "block_residual_uv"
TRANSFER = "tone_amplitude_ratio"
PHASE = "tone_phase_error_degrees"
WALL_CLOCK = "wall_clock_seconds"
PEAK_MEMORY = "peak_memory_bytes"


def write_rows(measured: Sequence[MeasuredArm], run: RunSpec, path: Path) -> None:
    """Append every measurement from one recording to the tidy table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for arm in measured:
            for row in _rows(arm, run):
                writer.writerow(row)


def _rows(arm: MeasuredArm, run: RunSpec) -> list[dict[str, object]]:
    """Describe one arm's measurements on one recording, one row per value."""
    base = {
        "participant": run.participant,
        "run_index": run.run_index,
        "motion_stratum": run.motion_stratum,
        "arm": arm.arm,
    }
    rows: list[dict[str, object]] = []
    for channel, residuals in zip(
        arm.channel_names, arm.block_residual_microvolts, strict=True
    ):
        for block, value in enumerate(residuals):
            rows.append(
                base
                | {
                    "measure": RESIDUAL,
                    "channel": channel,
                    "block": block,
                    "value": float(value),
                    "framewise_displacement_mm": _displacement(arm, block),
                }
            )
    for tone in arm.tone_transfer:
        shared = {
            "channel": tone.channel,
            "frequency_hz": tone.frequency_hz,
            "placement": tone.placement,
        }
        rows.append(
            base | shared | {"measure": TRANSFER, "value": tone.amplitude_ratio}
        )
        rows.append(
            base | shared | {"measure": PHASE, "value": tone.phase_error_degrees}
        )
    rows.append(base | {"measure": WALL_CLOCK, "value": arm.wall_clock_seconds})
    rows.append(base | {"measure": PEAK_MEMORY, "value": float(arm.peak_memory_bytes)})
    return rows


def _displacement(arm: MeasuredArm, block: int) -> float | None:
    """Return the motion in one block, or nothing where it was not estimated."""
    if block >= arm.block_peak_displacement.size:
        return None
    value = arm.block_peak_displacement[block]
    return None if np.isnan(value) else float(value)
