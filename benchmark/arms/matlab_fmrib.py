"""Correct one recording with the original MATLAB FMRIB FASTR.

The volume starts are resolved here rather than in MATLAB, from the same marker
stream and through the same code path this project's own arm uses, so the two
FASTR arms cannot disagree about where the scanner fired.

They are handed over one per volume, which is what this implementation is built
for. Measured on the demo recording, FMRIB given one trigger per volume
suppresses the harmonic comb 284-fold, matching this project's 283-fold; given
one trigger per acquisition group it suppresses nothing measurable. It has no
notion of an acquisition slot, so averaging a group against its neighbours in a
multiband sequence averages different slice sets together. Driving it per group
for the sake of a matched trigger count would report a missing capability as a
broken implementation.

MATLAB writes its samples as raw channel-major float32 beside a small JSON
description, which Python rewrites as the BrainVision recording every arm is
read from. A MAT file of this size would need either a format scipy cannot read
or a compression step costing more than the correction it stores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from benchmark.arms import ArmError, ArmOutput, run_measured
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings
from fastr_python.correction.timing import (
    AcquisitionGeometry,
    load_bids_fmri_timing,
    volume_marker_geometry,
)
from fastr_python.io.brainvision import BrainVisionMarker
from fastr_python.io.recording import (
    read_brainvision_recording,
    select_marker_sample_block,
    select_marker_samples,
    write_brainvision_recording,
)

NAME = "matlab_fmrib"
HARNESS = Path(__file__).with_name("fmrib_fastr_arm.m")
MICROVOLTS_PER_VOLT = 1e6


@dataclass(frozen=True, slots=True)
class MatlabFmribArm:
    """FMRIB FASTR 2.1 under MATLAB, held to the matched settings."""

    settings: MatchedSettings
    matlab: Path
    eeglab_root: Path

    @property
    def name(self) -> str:
        """Return the label this arm is reported under."""
        return NAME

    def correct(self, run: RunSpec, *, output_directory: Path) -> ArmOutput:
        """Correct one recording by driving MATLAB in batch mode."""
        output_directory.mkdir(parents=True, exist_ok=True)
        recording = read_brainvision_recording(run.raw_vhdr)
        raw = mne.io.read_raw_brainvision(run.raw_vhdr, preload=False, verbose="error")
        geometry = self._geometry(run, recording.markers, raw)

        samples = output_directory / f"{run.raw_vhdr.stem}_fmrib.f32"
        metadata_path = samples.with_suffix(".json")
        request = output_directory / f"{run.raw_vhdr.stem}_request.json"
        request.write_text(
            json.dumps(self._request(run, geometry, samples, metadata_path)),
            encoding="utf-8",
        )
        cost = run_measured(
            [
                str(self.matlab),
                "-batch",
                f"addpath('{HARNESS.parent}'); addpath('{self.eeglab_root}'); "
                f"fmrib_fastr_arm('{request}')",
            ]
        )
        if cost.exit_code != 0:
            raise ArmError(f"{NAME} failed on {run.raw_vhdr.name}:\n{cost.output}")

        corrected = _rewrite_as_brainvision(
            samples,
            metadata_path,
            output_vhdr=output_directory / f"{run.raw_vhdr.stem}_fmrib.vhdr",
            markers=recording.markers,
        )
        return ArmOutput(
            corrected_vhdr=corrected,
            sampling_rate=float(raw.info["sfreq"]),
            first_volume_sample=int(geometry.volume_starts[0]),
            # The span between the first and last volume marker, which is what
            # FMRIB corrects and what this project's arm emits. Samples past
            # the last marker are untouched artifact, not a worse correction.
            volume_count=geometry.volume_count - 1,
            cost=cost,
        )

    def _geometry(
        self,
        run: RunSpec,
        markers: tuple[BrainVisionMarker, ...],
        raw: mne.io.BaseRaw,
    ) -> AcquisitionGeometry:
        """Resolve group triggers exactly as this project's own arm resolves them."""
        volume_starts = select_marker_samples(
            markers,
            marker_type=self.settings.marker_type,
            marker_description=self.settings.marker_description,
            sample_count=raw.n_times,
        )
        if run.marker_block is not None:
            start, count = run.marker_block
            volume_starts = select_marker_sample_block(
                volume_starts, start_index=start, count=count
            )
        return volume_marker_geometry(
            volume_starts,
            sampling_rate=float(raw.info["sfreq"]),
            timing=load_bids_fmri_timing(run.protocol_json),
        )

    def _request(
        self,
        run: RunSpec,
        geometry: AcquisitionGeometry,
        samples: Path,
        metadata_path: Path,
    ) -> dict[str, object]:
        """Describe the whole job, so the harness decides nothing for itself."""
        return {
            "raw_vhdr": str(run.raw_vhdr),
            "volume_triggers": geometry.volume_starts.tolist(),
            "non_eeg_channels": list(self.settings.non_eeg_channels),
            "interpolation_factor": self.settings.interpolation_factor,
            "neighbor_count": self.settings.neighbor_count,
            "pre_trigger_fraction": self.settings.pre_trigger_fraction,
            "output_data": str(samples),
            "output_metadata": str(metadata_path),
        }


def _rewrite_as_brainvision(
    samples: Path,
    metadata_path: Path,
    *,
    output_vhdr: Path,
    markers: tuple[BrainVisionMarker, ...],
) -> Path:
    """Rewrite MATLAB's raw samples as the recording every arm is read from."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    channel_names = list(metadata["channel_names"])
    data = np.fromfile(samples, dtype="<f4").reshape(len(channel_names), -1)
    if data.shape[1] != metadata["sample_count"]:
        raise ArmError(
            f"{NAME} wrote {data.shape[1]} samples, not {metadata['sample_count']}"
        )
    write_brainvision_recording(
        data=data.astype(np.float64) / MICROVOLTS_PER_VOLT,
        sampling_rate=metadata["sampling_rate_hz"],
        channel_names=channel_names,
        output_vhdr=output_vhdr,
        markers=markers,
    )
    samples.unlink()
    return output_vhdr
