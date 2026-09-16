"""The contract every correction implementation is measured through.

An arm takes one recording and returns a corrected BrainVision file plus what
it cost to produce. Everything that differs between the tools -- MATLAB's MAT
round trip, FACETpy's EDF conversion and separate environment, TorchScript
model loading -- is absorbed here, so that nothing downstream knows which tool
wrote a recording.

Every arm runs as a subprocess, including the one that could run in process.
That is what makes the cost axis fair: the same clock and the same peak
resident set, measured the same way, for a tool invoked the way its users
invoke it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmark.cohort import RunSpec

# getrusage reports the peak resident set in bytes on macOS and in kibibytes
# everywhere else.
_MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024


class ArmError(RuntimeError):
    """Raised when an arm cannot produce a corrected recording."""


@dataclass(frozen=True, slots=True)
class MeasuredRun:
    """What one subprocess cost, and what it said."""

    wall_clock_seconds: float
    peak_memory_bytes: int
    exit_code: int
    output: str


@dataclass(frozen=True, slots=True)
class ArmOutput:
    """One arm's corrected recording, and where its first volume starts.

    The offset is reported rather than discovered downstream because only the
    arm knows it. Arms disagree about where a corrected recording begins, and
    a format like EDF does not carry the markers that would reveal it.
    """

    corrected_vhdr: Path
    sampling_rate: float
    first_volume_sample: int
    volume_count: int
    cost: MeasuredRun


class Arm(Protocol):
    """One correction implementation, measured the same way as the others."""

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording and report what producing it cost."""


def run_measured(
    command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> MeasuredRun:
    """Run a command to completion, timing it and capturing its own peak memory.

    ``os.wait4`` reports resource use for exactly this child, where
    ``getrusage`` would report a high-water mark shared with every other child
    the benchmark has already run.
    """
    started = time.perf_counter()
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        assert process.stdout is not None
        output = process.stdout.read()
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    return MeasuredRun(
        wall_clock_seconds=time.perf_counter() - started,
        peak_memory_bytes=usage.ru_maxrss * _MAXRSS_SCALE,
        exit_code=process.returncode,
        output=output,
    )
