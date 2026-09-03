"""Write a small synthetic dataset that the pipeline can be run against.

A correction pipeline cannot be tried out on a recording nobody has. This
module writes a complete, self-contained example -- a BrainVision recording
carrying a simulated gradient artifact, both marker conventions, a BIDS sidecar,
and a configuration referring to them -- so that installing the package is
enough to run one correction end to end and read a real provenance sidecar.

The recording is simulated, so its numbers demonstrate the interface rather than
any acquisition. Nothing here is a substitute for validating the correction on
the recordings actually being analysed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pybv import write_brainvision

from .io.brainvision import BrainVisionMarker, write_brainvision_markers
from .simulation import simulate_gradient_artifact

_CHANNEL_NAMES = ("Fz", "Cz", "Pz", "Oz", "ECG")
_INPUT_SAMPLING_RATE_HZ = 5_000.0
_OUTPUT_SAMPLING_RATE_HZ = 1_000.0
_REPETITION_TIME_SECONDS = 0.9
_GROUPS_PER_VOLUME = 18
_MULTIBAND_FACTOR = 2
_VOLUME_COUNT = 40
_READOUT_SECONDS = 0.03
# Deliberately off the volume-harmonic comb. At a 0.9 s repetition time the
# harmonics fall on multiples of 1.111 Hz, so a 10.0 Hz probe would sit exactly
# on the ninth one and template subtraction would remove it along with the
# artifact -- the 1/TR limitation, not a defect. 10.5 Hz is clear of both
# neighbours and survives, which is what the demo should show.
_ALPHA_HZ = 10.5
_ALPHA_MICROVOLTS = 12.0
_SEED = 20260830


@dataclass(frozen=True, slots=True)
class DemoDataset:
    """Where each generated file was written."""

    directory: Path
    raw_vhdr: Path
    fmri_metadata: Path
    config: Path


def write_demo_dataset(directory: str | Path) -> DemoDataset:
    """Write the recording, its metadata, and a ready-to-run configuration.

    The directory must not already contain the generated files, so a demo can
    never quietly overwrite a real dataset.
    """
    output_directory = Path(directory).expanduser().resolve()
    paths = DemoDataset(
        directory=output_directory,
        raw_vhdr=output_directory / "demo.vhdr",
        fmri_metadata=output_directory / "demo_bold.json",
        config=output_directory / "demo.yml",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _reject_existing_files(output_directory, paths)

    volume_starts, group_triggers = _acquisition_samples()
    sample_count = int(volume_starts[-1] + round(
        _REPETITION_TIME_SECONDS * _INPUT_SAMPLING_RATE_HZ
    ))
    data = _simulate_recording(group_triggers, sample_count)

    write_brainvision(
        data=data,
        sfreq=_INPUT_SAMPLING_RATE_HZ,
        ch_names=list(_CHANNEL_NAMES),
        fname_base=paths.raw_vhdr.stem,
        folder_out=output_directory,
        unit="µV",
        events=[],
        overwrite=False,
    )
    marker_path = paths.raw_vhdr.with_suffix(".vmrk")
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        paths.raw_vhdr.with_suffix(".eeg").name,
        _markers(volume_starts, group_triggers),
    )
    paths.fmri_metadata.write_text(
        json.dumps(_bids_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    paths.config.write_text(_configuration(paths), encoding="utf-8")
    return paths


def _acquisition_samples() -> tuple[np.ndarray, np.ndarray]:
    """Volume starts and acquisition-group triggers, in input samples."""
    samples_per_volume = round(_REPETITION_TIME_SECONDS * _INPUT_SAMPLING_RATE_HZ)
    group_stride = samples_per_volume // _GROUPS_PER_VOLUME
    volume_starts = (
        np.arange(_VOLUME_COUNT, dtype=np.int64) * samples_per_volume
    )
    offsets = np.arange(_GROUPS_PER_VOLUME, dtype=np.int64) * group_stride
    return volume_starts, (volume_starts[:, np.newaxis] + offsets).reshape(-1)


def _simulate_recording(
    group_triggers: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Gradient artifact over a known alpha rhythm, in volts.

    The alpha tone is off the volume-harmonic comb, so it is signal the
    correction is expected to keep, and the before/after figures show it.
    """
    artifact = simulate_gradient_artifact(
        group_triggers,
        sample_count=sample_count,
        channel_count=len(_CHANNEL_NAMES),
        sampling_rate=_INPUT_SAMPLING_RATE_HZ,
        readout_seconds=_READOUT_SECONDS,
        groups_per_volume=_GROUPS_PER_VOLUME,
        seed=_SEED,
    )
    times = np.arange(sample_count, dtype=np.float64) / _INPUT_SAMPLING_RATE_HZ
    generator = np.random.default_rng(_SEED)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=len(_CHANNEL_NAMES))
    alpha = _ALPHA_MICROVOLTS * np.sin(
        2.0 * np.pi * _ALPHA_HZ * times + phases[:, np.newaxis]
    )
    return (artifact + alpha) * 1e-6


def _markers(
    volume_starts: np.ndarray,
    group_triggers: np.ndarray,
) -> tuple[BrainVisionMarker, ...]:
    """Mark every volume and every acquisition group, so both modes are usable."""
    volume_samples = set(int(sample) for sample in volume_starts)
    markers = [BrainVisionMarker("New Segment", "", 1, 1, 0)]
    for sample in group_triggers:
        position = int(sample) + 1
        if int(sample) in volume_samples:
            markers.append(
                BrainVisionMarker("Volume", "volume-start", position, 1, 0)
            )
        markers.append(BrainVisionMarker("Slice", "slice-start", position, 1, 0))
    return tuple(markers)


def _bids_metadata() -> dict[str, object]:
    group_offsets = [
        round(index * _REPETITION_TIME_SECONDS / _GROUPS_PER_VOLUME, 6)
        for index in range(_GROUPS_PER_VOLUME)
    ]
    return {
        "RepetitionTime": _REPETITION_TIME_SECONDS,
        # Each acquisition-time slot excites `MultibandAccelerationFactor`
        # slices at once, so every offset appears exactly that many times.
        "SliceTiming": sorted(group_offsets * _MULTIBAND_FACTOR),
        "MultibandAccelerationFactor": _MULTIBAND_FACTOR,
    }


def _configuration(paths: DemoDataset) -> str:
    """Build a commented configuration for this dataset.

    Written as text rather than dumped, because the comments are most of what
    makes a first configuration readable.
    """
    return f"""# Generated by `fastr-python demo`. Paths are relative to this file.
#
# The recording is simulated: a multiband gradient artifact over a {_ALPHA_HZ} Hz
# tone that is deliberately off the volume-harmonic comb, so a correct run
# suppresses the artifact and keeps the tone. Compare the before and after PSD
# figures, then read the provenance sidecar next to the output.

input:
  raw_vhdr: {paths.raw_vhdr.name}
  # Acquisition timing may come from this BIDS sidecar or from an `acquisition:`
  # section here, but not from both. See `examples/configuration.yml`.
  fmri_metadata: {paths.fmri_metadata.name}

output:
  vhdr: demo_corrected.vhdr

timing:
  marker_type: Volume
  marker_description: volume-start
  # This recording is also marked once per acquisition group. To correct from
  # those instead, set marker_kind to slice, name the Slice markers, add
  # `groups_per_volume: {_GROUPS_PER_VOLUME}`, and remove input.fmri_metadata:
  #
  #   marker_type: Slice
  #   marker_description: slice-start
  #   marker_kind: slice
  #   groups_per_volume: {_GROUPS_PER_VOLUME}
  #   expected_repetition_time_seconds: {_REPETITION_TIME_SECONDS}
  marker_kind: volume

processing:
  method: acquisition_group_fastr
  interpolation_factor: 10
  # The template window is a scientific parameter, not a speed setting: it
  # trades residual artifact against loss at the volume harmonics. Revalidate
  # it for a real protocol.
  neighbor_count: 20
  search_radius_samples: 3
  lowpass_hz: 100.0
  output_sampling_rate_hz: {_OUTPUT_SAMPLING_RATE_HZ}
  channel_batch_size: 8
  reference_channel: Cz
  # Simulated data carries no line noise; use [60.0] or [50.0] on a recording.
  line_noise_frequencies_hz: []
  non_eeg_channels: [ECG]
  template_high_pass_hz: 1.0

trim:
  mode: first_to_last_volume
"""


def _reject_existing_files(directory: Path, paths: DemoDataset) -> None:
    generated = (
        paths.raw_vhdr,
        paths.raw_vhdr.with_suffix(".eeg"),
        paths.raw_vhdr.with_suffix(".vmrk"),
        paths.fmri_metadata,
        paths.config,
    )
    existing = [path.name for path in generated if path.exists()]
    if existing:
        raise FileExistsError(
            f"demo files already exist in {directory}: {', '.join(existing)}"
        )
