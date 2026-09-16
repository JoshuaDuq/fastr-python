"""Draw one figure per axis, each one refusing to be read on its own.

The first figure is the one that matters: residual against on-comb transfer,
one point per arm. Every other way of drawing suppression lets a reader rank
the arms without seeing what the ranking cost, and on this data the two orders
are opposite.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from benchmark.analysis import (
    MotionSensitivity,
    cost,
    suppression_and_transfer,
)
from benchmark.probes import BETWEEN_HARMONICS, ON_HARMONIC
from benchmark.report import RESIDUAL, TRANSFER

FIGURE_SIZE = (7.5, 5.0)
PLACEMENT_LABELS = {
    ON_HARMONIC: "on a volume harmonic",
    BETWEEN_HARMONICS: "between harmonics",
}


def write_figures(
    measurements: pd.DataFrame,
    motion: Sequence[MotionSensitivity],
    directory: Path,
) -> tuple[Path, ...]:
    """Draw every figure the benchmark reports, and return where they landed."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        _suppression_against_transfer(measurements, directory / "suppression_cost.png"),
        _residual_spread(measurements, directory / "residual_spread.png"),
        _transfer_by_frequency(measurements, directory / "transfer_by_frequency.png"),
        _cost(measurements, directory / "cost.png"),
    ]
    if motion:
        written.append(_motion(motion, directory / "motion_sensitivity.png"))
    return tuple(written)


def _suppression_against_transfer(measurements: pd.DataFrame, path: Path) -> Path:
    """Plot what each arm removed against what it removed along with it.

    Both placements for every arm, joined, on a log axis. The two sit orders of
    magnitude apart and a linear axis would press the on-comb points flat
    against zero, which is the one number a reader must not lose.
    """
    summary = suppression_and_transfer(measurements)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    for arm, row in summary.iterrows():
        residual = row["residual_uv_median"]
        axes.plot(
            [residual, residual],
            [row["transfer_on_comb"], row["transfer_between_comb"]],
            color="0.7",
            linewidth=1.5,
            zorder=2,
        )
        axes.annotate(
            str(arm),
            (residual, row["transfer_between_comb"]),
            textcoords="offset points",
            xytext=(4, 10),
            ha="left",
            rotation=30,
            fontsize=8,
        )
    axes.scatter(
        summary["residual_uv_median"],
        summary["transfer_between_comb"],
        s=60,
        zorder=3,
        label=PLACEMENT_LABELS[BETWEEN_HARMONICS],
    )
    axes.scatter(
        summary["residual_uv_median"],
        summary["transfer_on_comb"],
        s=60,
        marker="v",
        zorder=3,
        label=PLACEMENT_LABELS[ON_HARMONIC],
    )
    axes.axhline(1.0, linestyle=":", linewidth=1, color="0.6")
    axes.set_yscale("log")
    axes.set_xlabel("median block residual at the scanner harmonics (µV)")
    axes.set_ylabel("injected tone kept")
    axes.set_title("Suppression costs signal at the frequencies it cleans")
    axes.margins(x=0.22, y=0.35)
    axes.legend(loc="center left", fontsize=8)
    axes.grid(alpha=0.3, which="both")
    return _save(figure, path)


def _residual_spread(measurements: pd.DataFrame, path: Path) -> Path:
    """Plot how each arm's residual is distributed over blocks and channels."""
    residual = measurements[measurements["measure"] == RESIDUAL]
    arms = sorted(residual["arm"].unique())
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.boxplot(
        [residual.loc[residual["arm"] == arm, "value"] for arm in arms],
        tick_labels=arms,
        showfliers=False,
    )
    axes.set_ylabel("block residual (µV)")
    axes.set_title("Residual across every block and channel")
    axes.tick_params(axis="x", rotation=20)
    axes.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def _transfer_by_frequency(measurements: pd.DataFrame, path: Path) -> Path:
    """Plot what survived at each probe frequency, the two placements apart."""
    transfer = measurements[measurements["measure"] == TRANSFER]
    figure, axes = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1]), sharey=True
    )
    for panel, placement in zip(axes, (ON_HARMONIC, BETWEEN_HARMONICS), strict=True):
        placed = transfer[transfer["placement"] == placement]
        for arm in sorted(placed["arm"].unique()):
            rows = placed[placed["arm"] == arm]
            curve = rows.groupby("frequency_hz")["value"].median()
            panel.plot(curve.index, curve.to_numpy(), marker="o", label=arm)
        panel.axhline(1.0, linestyle=":", linewidth=1, color="0.6")
        panel.set_title(PLACEMENT_LABELS[placement])
        panel.set_xlabel("frequency (Hz)")
        panel.grid(alpha=0.3)
    axes[0].set_ylabel("injected tone kept")
    axes[0].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    return _save(figure, path)


def _motion(motion: Sequence[MotionSensitivity], path: Path) -> Path:
    """Plot how fast each arm's residual grows with head movement."""
    ordered = sorted(motion, key=lambda arm: arm.slope_uv_per_mm)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.barh(
        [arm.arm for arm in ordered],
        [arm.slope_uv_per_mm for arm in ordered],
        xerr=[arm.standard_error for arm in ordered],
        zorder=3,
    )
    axes.set_xlabel("residual growth per mm of block motion (µV/mm)")
    axes.set_title("Sensitivity to movement, fitted per 30 s block")
    axes.grid(axis="x", alpha=0.3)
    return _save(figure, path)


def _cost(measurements: pd.DataFrame, path: Path) -> Path:
    """Plot what each arm cost in clock time on the same machine."""
    table = cost(measurements)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.barh(list(table.index), table["wall_clock_seconds_median"] / 60.0, zorder=3)
    axes.set_xlabel("median wall clock per recording (minutes)")
    axes.set_title("Processing time, one machine, one correction pass")
    axes.grid(axis="x", alpha=0.3)
    return _save(figure, path)


def _save(figure: Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
