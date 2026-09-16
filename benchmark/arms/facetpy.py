"""Correct one recording with FACETpy's averaged artifact subtraction.

FACETpy runs in its own interpreter. It pins numpy 2.1.3 and MNE 1.10.2 against
this project's 2.4.6 and 1.12.1, so the two cannot share an environment; and it
is GPL-3.0-only where this project is GPL-2.0-only, so the benchmark aggregates
the two rather than combining them.

One arm, at FACETpy's own documented shape: one artifact per volume, averaged
against its neighbours at its default window.

There is no acquisition-slot arm, because FACETpy cannot represent this
cohort's acquisition. Its slot-matched mode cuts every epoch to one fixed
length, taken from the median trigger spacing, and subtracts each epoch's
template in place and cumulatively. This cohort's multiband slots are spaced
237, 238, 250 and 350 samples apart, so no single length tiles a volume: at the
inferred 250 the epochs overlap and about half the slot boundaries are
subtracted twice, leaving 113.7 uV against 2.4 uV for the other arms; at 237,
the widest length that cannot overlap, the gaps left behind still leave
66.9 uV. Driving it that way would have measured FACETpy's epoch model against
this project's slot geometry and reported a missing capability as a broken
implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
VOLUME_AVERAGED = "volume_averaged"
NAME = "facetpy_volume_averaged"


@dataclass(frozen=True, slots=True)
class FacetpyArm:
    """FACETpy's volume-averaged correction, driven in its own environment."""

    settings: MatchedSettings
    interpreter: Path
    window_size: int = 30

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""
        return NAME

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording by driving FACETpy's interpreter."""
        output_directory.mkdir(parents=True, exist_ok=True)
        geometry = resolve_geometry(run, settings=self.settings)
        corrected = output_directory / f"{run.raw_vhdr.stem}_{self.name}.vhdr"
        metadata_path = corrected.with_suffix(".meta.json")
        request = output_directory / f"{run.raw_vhdr.stem}_{self.name}_request.json"
        request.write_text(
            json.dumps(self._request(run, geometry, corrected, metadata_path)),
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
    ) -> dict[str, object]:
        """Describe the whole job, so the runner decides nothing for itself."""
        return {
            "mode": VOLUME_AVERAGED,
            "raw_vhdr": str(run.raw_vhdr),
            "triggers": geometry.volume_starts.tolist(),
            "volume_starts": geometry.volume_starts.tolist(),
            "window_size": self.window_size,
            "interpolation_factor": self.settings.interpolation_factor,
            "output_vhdr": str(corrected),
            "output_metadata": str(metadata_path),
        }
