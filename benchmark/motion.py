"""Read head motion from fMRIPrep confounds, per run and per block.

The run mean is what stratified selection ranks on. The per-block peak is what
the motion axis regresses against: across this cohort the run means span
0.05-0.33 mm, while 30 s block peaks reach 8 mm, so only the block scale has
the dynamic range to fit a slope against.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

DISPLACEMENT_COLUMN = "framewise_displacement"
UNSET_VALUES = frozenset({"n/a", ""})


class MotionError(ValueError):
    """Raised when a confounds file cannot supply framewise displacement."""


def framewise_displacement(confounds_tsv: Path) -> np.ndarray:
    """Return one displacement per volume, unset volumes held as NaN.

    The array keeps its original length so that an index is a volume number.
    fMRIPrep leaves the first volume unset, because displacement is a
    difference between consecutive volumes; dropping it instead would shift
    every later volume by one and misalign the blocks it is compared against.
    """
    with confounds_tsv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or DISPLACEMENT_COLUMN not in rows[0]:
        raise MotionError(f"{confounds_tsv} has no {DISPLACEMENT_COLUMN} column")
    displacement = np.array(
        [
            np.nan
            if row[DISPLACEMENT_COLUMN] in UNSET_VALUES
            else row[DISPLACEMENT_COLUMN]
            for row in rows
        ],
        dtype=float,
    )
    if not np.any(np.isfinite(displacement)):
        raise MotionError(f"{confounds_tsv} has no usable displacement values")
    return displacement


def mean_framewise_displacement(confounds_tsv: Path) -> float:
    """Average framewise displacement over the volumes that define it."""
    displacement = framewise_displacement(confounds_tsv)
    return float(displacement[np.isfinite(displacement)].mean())


def block_peak_displacement(
    confounds_tsv: Path,
    *,
    repetition_time_seconds: float,
    block_seconds: float,
    block_count: int,
) -> np.ndarray:
    """Return the largest displacement inside each residual-QC block.

    The peak rather than the mean, because a correction is damaged by the
    moment the head moves, not by the average over half a minute around it.

    Blocks follow the corrected recording's own grid. A block holding no
    estimated volume is reported as NaN rather than as quiet, so that a
    recording longer than its motion estimates cannot read as motionless.
    """
    if not math.isfinite(block_seconds) or block_seconds <= 0:
        raise MotionError("block duration must be finite and positive")
    if not math.isfinite(repetition_time_seconds) or repetition_time_seconds <= 0:
        raise MotionError("repetition time must be finite and positive")
    if block_count < 0:
        raise MotionError("block count must not be negative")
    volumes_per_block = round(block_seconds / repetition_time_seconds)
    if volumes_per_block < 1:
        raise MotionError("blocks must span at least one volume")
    displacement = framewise_displacement(confounds_tsv)
    peaks = np.full(block_count, np.nan)
    for block in range(block_count):
        volumes = displacement[
            block * volumes_per_block : (block + 1) * volumes_per_block
        ]
        measured = volumes[np.isfinite(volumes)]
        if measured.size:
            peaks[block] = measured.max()
    return peaks
