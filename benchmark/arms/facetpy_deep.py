"""Correct one recording with one of FACETpy's pretrained networks.

Every one of these models was trained by its author on synthetic data, at a
different sampling rate and montage from this cohort. Running them here
measures how far that training generalises, which is a different question from
how well the architectures work, and the answer is allowed to be that a model
does not run at all: a screening pass reports that as its result rather than
hiding it.

The arm is separate from the classical FACETpy arms because it is configured
differently -- by a checkpoint and the settings that checkpoint was trained
with, rather than by a window and a trigger kind -- even though both are driven
through the same runner and the same interpreter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne

from benchmark.arms import (
    ArmError,
    ArmOutput,
    input_sampling_rate,
    resolve_geometry,
    run_measured,
)
from benchmark.arms.facetpy import RUNNER
from benchmark.arms.model_zoo import ModelExport
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

DEEP_LEARNING = "deep_learning"

# Export names that do not spell their architecture the way FACETpy's blueprint
# registry does. Anything absent here is looked up under its own name, and an
# unknown name is refused rather than guessed at.
BLUEPRINT_ALIASES = {
    "ic_unet": "ic_u_net",
    "dhct_gan_v2": "dhct_gan",
    "cascaded_dae": "dar",
    "cascaded_context_dae": "dar",
}


@dataclass(frozen=True, slots=True)
class FacetpyDeepArm:
    """One pretrained FACETpy network, run as its author published it."""

    settings: MatchedSettings
    interpreter: Path
    export: ModelExport
    checkpoint: Path

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""
        return f"ml_{self.export.name}"

    @property
    def blueprint(self) -> str:
        """Return the architecture blueprint this checkpoint was exported from."""
        return BLUEPRINT_ALIASES.get(self.export.name, self.export.name)

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording by driving FACETpy's interpreter."""
        output_directory.mkdir(parents=True, exist_ok=True)
        geometry = resolve_geometry(run, settings=self.settings)
        channel_count = len(
            mne.io.read_raw_brainvision(
                run.raw_vhdr, preload=False, verbose="error"
            ).ch_names
        )
        corrected = output_directory / f"{run.raw_vhdr.stem}_{self.name}.vhdr"
        metadata_path = corrected.with_suffix(".meta.json")
        request = output_directory / f"{run.raw_vhdr.stem}_{self.name}_request.json"
        request.write_text(
            json.dumps(
                {
                    "mode": DEEP_LEARNING,
                    "raw_vhdr": str(run.raw_vhdr),
                    "triggers": geometry.group_triggers.tolist(),
                    "volume_starts": geometry.volume_starts.tolist(),
                    "blueprint": self.blueprint,
                    "model_name": self.export.name,
                    "checkpoint_path": str(self.checkpoint),
                    "chunk_size_samples": self.export.chunk_size_samples,
                    "channel_count": channel_count,
                    "output_type": self.export.output_type,
                    "interpolation_factor": self.settings.interpolation_factor,
                    "output_vhdr": str(corrected),
                    "output_metadata": str(metadata_path),
                }
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
