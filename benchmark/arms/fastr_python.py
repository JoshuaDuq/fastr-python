"""Correct one recording with this project's own pipeline.

The arm shells out to the installed command rather than calling the library, so
that its cost is measured the way every other arm's is, and is the cost a user
would actually pay.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmark.arms import ArmError, ArmOutput, input_sampling_rate, run_measured
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

NAME = "fastr_python"
TRIM_MODE = "first_to_last_volume"


@dataclass(frozen=True, slots=True)
class FastrPythonArm:
    """This project's acquisition-slot FASTR, held to the matched settings."""

    settings: MatchedSettings

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""
        return NAME

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording through the installed command line."""
        output_directory.mkdir(parents=True, exist_ok=True)
        corrected = output_directory / f"{run.raw_vhdr.stem}_fastr.vhdr"
        config_path = output_directory / f"{run.raw_vhdr.stem}_config.yml"
        sampling_rate = input_sampling_rate(run.raw_vhdr)
        config_path.write_text(
            yaml.safe_dump(
                self._configuration(run, corrected, sampling_rate=sampling_rate),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        cost = run_measured([_executable(), "run", "--config", str(config_path)])
        if cost.exit_code != 0:
            raise ArmError(f"{NAME} failed on {run.raw_vhdr.name}:\n{cost.output}")
        return _read_output(corrected, cost=cost)

    def _configuration(
        self, run: RunSpec, corrected: Path, *, sampling_rate: float
    ) -> dict[str, Any]:
        """Build the YAML the command reads, in the cohort's own settings.

        The output is left at the recorded rate and unfiltered. The benchmark
        applies one anti-alias low-pass and decimation to every arm afterwards,
        so that the comparison is of corrections and not of output filters.
        """
        timing: dict[str, Any] = {
            "marker_type": self.settings.marker_type,
            "marker_description": self.settings.marker_description,
            "marker_kind": "volume",
            "missing_volume_markers": "error",
        }
        if run.marker_block is not None:
            start, count = run.marker_block
            timing["volume_marker_start_index"] = start
            timing["volume_marker_count"] = count
        return {
            "input": {
                "raw_vhdr": str(run.raw_vhdr),
                "fmri_metadata": str(run.protocol_json),
            },
            "output": {"vhdr": str(corrected)},
            "timing": timing,
            "processing": {
                "method": "acquisition_group_fastr",
                "interpolation_factor": self.settings.interpolation_factor,
                "neighbor_count": self.settings.neighbor_count,
                "search_radius_samples": self.settings.search_radius_samples,
                "pre_trigger_fraction": self.settings.pre_trigger_fraction,
                "template_high_pass_hz": self.settings.template_high_pass_hz,
                "lowpass_hz": 0.0,
                "output_sampling_rate_hz": sampling_rate,
                "non_eeg_channels": list(self.settings.non_eeg_channels),
                "channel_batch_size": self.settings.channel_batch_size,
                "reference_channel": self.settings.reference_channel,
                "line_noise_frequencies_hz": list(
                    self.settings.line_noise_frequencies_hz
                ),
                "residual_obs": False,
                "adaptive_noise_cancellation": False,
                "channel_failure_policy": "report",
            },
            "quality_control": {
                "block_seconds": self.settings.block_seconds,
                "mains_frequency_hz": self.settings.mains_frequency_hz,
            },
            "trim": {"mode": TRIM_MODE},
        }


def _read_output(corrected: Path, *, cost: Any) -> ArmOutput:
    """Describe a finished correction from the sidecar it wrote beside itself."""
    sidecar = json.loads(corrected.with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar["trim"]["mode"] != TRIM_MODE:
        raise ArmError(f"{NAME} trimmed to {sidecar['trim']['mode']}, not {TRIM_MODE}")
    sampling_rate = sidecar["output"]["sampling_rate_hz"]
    repetition = sidecar["timing"]["resolved"]["repetition_time_seconds"]
    samples_per_volume = round(repetition * sampling_rate)
    return ArmOutput(
        corrected_vhdr=corrected,
        sampling_rate=sampling_rate,
        # `first_to_last_volume` starts the output on the first volume marker.
        first_volume_sample=0,
        volume_count=sidecar["output"]["sample_count"] // samples_per_volume,
        cost=cost,
    )


def _executable() -> str:
    """Return the command that belongs to the interpreter now running."""
    command = Path(sys.executable).parent / "fastr-python"
    if not command.is_file():
        raise ArmError(f"no fastr-python command beside {sys.executable}")
    return str(command)
