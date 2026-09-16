"""The settings the benchmark holds every arm to, and where its work goes.

One set of parameters is shared by every arm that has a knob for it, so that a
difference between two arms is a difference between implementations rather
than between the numbers each was handed. The values are the cohort's own, read
from the sidecars of the corrected build in
``sub-*/eeg/fastr_corrected_1khz/``.

An arm with no equivalent knob is not given a substitute. FACETpy's averaged
template has no acquisition-slot geometry to match, so it is reported both held
to what it can honour and left at its own defaults, and the gap between those
two is the part of the comparison that is about method rather than code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MatchedSettings:
    """Correction parameters every capable arm is held to."""

    marker_type: str = "Volume"
    marker_description: str = "V  1"
    reference_channel: str = "Fp1"
    interpolation_factor: int = 10
    neighbor_count: int = 60
    search_radius_samples: int = 3
    pre_trigger_fraction: float = 0.03
    template_high_pass_hz: float = 1.0
    lowpass_hz: float = 100.0
    output_sampling_rate_hz: float = 1000.0
    non_eeg_channels: tuple[str, ...] = ("ECG",)
    # No mains regression: the cohort was corrected without it, and removing
    # power-line interference is not part of gradient correction.
    line_noise_frequencies_hz: tuple[float, ...] = ()
    # A throughput knob rather than a scientific one, matched all the same
    # because it moves the wall clock the cost axis reports.
    channel_batch_size: int = 8
    mains_frequency_hz: float = 60.0
    block_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class ToneSettings:
    """How loud the injected probes are, and where they are allowed to sit."""

    amplitude_volts: float = 5e-6
    mains_guard_hz: float = 1.0


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    """Where the cohort is read from and where every arm's output is written."""

    source_root: Path
    confounds_root: Path
    protocol_root: Path
    output_root: Path

    def arm_directory(self, arm: str, participant: str) -> Path:
        """Return where one arm keeps one participant's corrected recordings."""
        return self.output_root / "corrections" / arm / participant

    @property
    def manifest(self) -> Path:
        """Return the frozen selection every arm reads its recordings from."""
        return self.output_root / "manifest.json"

    @property
    def probes_root(self) -> Path:
        """Return where the tone-injected copies are staged."""
        return self.output_root / "tone_injected"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Everything one benchmark run needs, resolved and validated."""

    paths: BenchmarkPaths
    runs_per_participant: int = 3
    matched: MatchedSettings = field(default_factory=MatchedSettings)
    tones: ToneSettings = field(default_factory=ToneSettings)
