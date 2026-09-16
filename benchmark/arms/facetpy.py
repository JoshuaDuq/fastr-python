"""Correct one recording with FACETpy, in two configurations.

FACETpy runs in its own interpreter. It pins numpy 2.1.3 and MNE 1.10.2 against
this project's 2.4.6 and 1.12.1, so the two cannot share an environment; and it
is GPL-3.0-only where this project is GPL-2.0-only, so the benchmark aggregates
the two rather than combining them.

Two arms, because one number would answer two different questions at once:

`facetpy_slot_matched` averages each acquisition slot against the same slot in
neighbouring volumes, on the same trigger array and the same window this
project's arm uses. What separates it from `fastr_python` is implementation.

`facetpy_volume_averaged` is FACETpy's own documented shape, one artifact per
volume at its default window. What separates it from the slot-matched arm is
method, and reporting only the matched arm would judge the tool on a
configuration its authors never recommended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from benchmark.arms import (
    ArmError,
    ArmOutput,
    input_sampling_rate,
    resolve_geometry,
    run_measured,
)
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings
from fastr_python.correction.timing import AcquisitionGeometry

RUNNER = Path(__file__).with_name("facetpy_runner.py")
SLOT_MATCHED = "slot_matched"
VOLUME_AVERAGED = "volume_averaged"


@dataclass(frozen=True, slots=True)
class FacetpyArm:
    """FACETpy in one of its two configurations, driven in its own environment."""

    settings: MatchedSettings
    interpreter: Path
    mode: str
    window_size: int = 30

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""
        return f"facetpy_{self.mode}"

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording by driving FACETpy's interpreter."""
        if self.mode not in (SLOT_MATCHED, VOLUME_AVERAGED):
            raise ArmError(f"unknown FACETpy mode: {self.mode!r}")
        output_directory.mkdir(parents=True, exist_ok=True)
        geometry = resolve_geometry(run, settings=self.settings)
        sample_count = mne.io.read_raw_brainvision(
            run.raw_vhdr, preload=False, verbose="error"
        ).n_times
        corrected = output_directory / f"{run.raw_vhdr.stem}_{self.name}.vhdr"
        metadata_path = corrected.with_suffix(".meta.json")
        request = output_directory / f"{run.raw_vhdr.stem}_{self.name}_request.json"
        request.write_text(
            json.dumps(
                self._request(
                    run,
                    geometry,
                    corrected,
                    metadata_path,
                    sample_count=sample_count,
                )
            ),
            encoding="utf-8",
        )
        cost = run_measured([str(self.interpreter), str(RUNNER), str(request)])
        if cost.exit_code != 0:
            raise ArmError(
                f"{self.name} failed on {run.raw_vhdr.name} "
                f"(exit {cost.exit_code}):\n{cost.output}",
                cost=cost,
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return ArmOutput(
            corrected_vhdr=corrected,
            sampling_rate=input_sampling_rate(run.raw_vhdr),
            first_volume_sample=metadata["first_volume_sample"],
            volume_count=metadata["volume_count"],
            cost=cost,
        )

    def _request(
        self,
        run: RunSpec,
        geometry: AcquisitionGeometry,
        corrected: Path,
        metadata_path: Path,
        *,
        sample_count: int,
    ) -> dict[str, object]:
        """Describe the whole job, so the runner decides nothing for itself."""
        triggers = (
            _whole_volumes_inside(geometry, sample_count=sample_count)
            if self.mode == SLOT_MATCHED
            else geometry.volume_starts
        )
        return {
            "mode": self.mode,
            "raw_vhdr": str(run.raw_vhdr),
            "triggers": triggers.tolist(),
            "volume_starts": geometry.volume_starts.tolist(),
            "groups_per_volume": geometry.groups_per_volume,
            "neighbor_count": self.settings.neighbor_count,
            "window_size": self.window_size,
            "interpolation_factor": self.settings.interpolation_factor,
            "output_vhdr": str(corrected),
            "output_metadata": str(metadata_path),
        }


def _whole_volumes_inside(
    geometry: AcquisitionGeometry, *, sample_count: int
) -> np.ndarray:
    """Keep the acquisition groups of the volumes that fit in the recording.

    A volume's later groups can start after the recording ends, because a group
    fires part way through the repetition its marker begins. This project's own
    arm drops such a volume whole rather than correcting part of it, so the same
    volumes are dropped here; handing the survivors alone would leave the two
    arms correcting different acquisitions.
    """
    groups = geometry.group_triggers.reshape(-1, geometry.groups_per_volume)
    span = int(np.median(np.diff(geometry.group_triggers)))
    complete = (groups + span <= sample_count).all(axis=1)
    return groups[complete].reshape(-1)
