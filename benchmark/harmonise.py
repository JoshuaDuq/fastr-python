"""Put every arm's output on one grid, so a metric compares corrections only.

Two things would otherwise be compared along with the correction.

The first is the output filter. Each tool ships its own: `fmrib_fastr.m` builds
a least-squares FIR and runs it through `filtfilt`, which squares the response
to -4.4 dB at 90 Hz, where this project's once-applied windowed FIR is flat.
Left alone that gap would appear in the results as MATLAB destroying two fifths
of the neural signal at 90 Hz, which is not a correction result at all. So the
arms correct at the recorded rate and do not filter, and one shared anti-alias
low-pass and decimation is applied here, to all of them, afterwards.

The second is the output length. The arms disagree about where a corrected
recording ends: this project stops at the last volume marker, the retrimmed
originals run one repetition past it, and the MATLAB harness pads whole volumes
around the range it was given. The shared span holds an even number of whole
volumes. Whole volumes keep every artifact harmonic on an exact Fourier bin; an
even count does the same for the tones placed halfway between harmonics, which
complete whole cycles only over an even number of repetitions.

A whole volume is dropped from each end of that span. The shared filter smears
its own edge across roughly its half-length, and this project's arm starts its
output on the first volume marker, which would otherwise put that smear inside
the first thing measured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fastr_python.pipeline.io import apply_output_low_pass, make_output_low_pass


class HarmoniseError(ValueError):
    """Raised when the arms share no span that can be measured."""


@dataclass(frozen=True, slots=True)
class CommonWindow:
    """The span, in whole volumes, that every arm is measured over."""

    volume_count: int
    guard_volumes: int
    samples_per_volume: int
    sampling_rate: float

    @property
    def sample_count(self) -> int:
        """Return how many samples the span holds."""
        return self.volume_count * self.samples_per_volume

    @property
    def guard_samples(self) -> int:
        """Return how many samples are dropped at each end of the span."""
        return self.guard_volumes * self.samples_per_volume


def guard_volumes(
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    repetition_time_seconds: float,
) -> int:
    """Return the whole volumes at each end that the shared filter contaminates.

    Derived from the filter actually designed rather than fixed, so that
    changing the cutoff cannot quietly leave its edge inside the measurement.
    """
    if lowpass_hz == 0.0:
        return 0
    taps = make_output_low_pass(
        sampling_rate, lowpass_hz, output_sampling_rate=output_sampling_rate
    )
    half_length_seconds = ((taps.size - 1) // 2) / sampling_rate
    return max(1, math.ceil(half_length_seconds / repetition_time_seconds))


def to_output_rate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
) -> np.ndarray:
    """Low-pass and decimate one arm's output onto the shared grid.

    The whole recording is filtered before anything is cut from it, so the
    filter's edge transient stays outside the span that gets measured.
    """
    decimation = sampling_rate / output_sampling_rate
    if abs(decimation - round(decimation)) > 1e-9:
        raise HarmoniseError(
            f"{sampling_rate} Hz does not decimate to {output_sampling_rate} Hz"
        )
    filtered = apply_output_low_pass(
        data,
        sampling_rate=sampling_rate,
        output_sampling_rate=output_sampling_rate,
        lowpass_hz=lowpass_hz,
    )
    return filtered[:, :: round(decimation)]


def common_window(
    volume_counts: Mapping[str, int],
    *,
    repetition_time_seconds: float,
    sampling_rate: float,
    guard: int,
) -> CommonWindow:
    """Take the longest even run of whole volumes every arm produced, less guards."""
    if not volume_counts:
        raise HarmoniseError("a common window needs at least one arm")
    if not math.isfinite(repetition_time_seconds) or repetition_time_seconds <= 0:
        raise HarmoniseError("repetition time must be finite and positive")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise HarmoniseError("sampling rate must be finite and positive")
    exact = repetition_time_seconds * sampling_rate
    samples_per_volume = round(exact)
    if abs(exact - samples_per_volume) > 1e-9:
        raise HarmoniseError(
            f"a volume spans {exact} samples at {sampling_rate} Hz, which is not whole"
        )
    if guard < 0:
        raise HarmoniseError("the guard must not be negative")
    shared = min(volume_counts.values()) - 2 * guard
    if shared < 1:
        reported = ", ".join(
            f"{arm}={count}" for arm, count in sorted(volume_counts.items())
        )
        raise HarmoniseError(
            f"no arm keeps a whole volume past a {guard}-volume guard: {reported}"
        )
    volume_count = shared - shared % 2
    if volume_count < 2:
        raise HarmoniseError("the shared span must hold at least two volumes")
    return CommonWindow(
        volume_count=volume_count,
        guard_volumes=guard,
        samples_per_volume=samples_per_volume,
        sampling_rate=sampling_rate,
    )


def crop(
    data: np.ndarray, *, first_volume_sample: int, window: CommonWindow
) -> np.ndarray:
    """Cut one arm's output down to the shared span, starting at its volume 0."""
    if data.ndim != 2:
        raise HarmoniseError("cropping expects a channel-by-sample matrix")
    if first_volume_sample < 0:
        raise HarmoniseError("the first volume cannot start before the recording")
    start = first_volume_sample + window.guard_samples
    stop = start + window.sample_count
    if stop > data.shape[1]:
        raise HarmoniseError(
            f"the shared span needs {stop} samples but this arm wrote {data.shape[1]}"
        )
    return data[:, start:stop]


def decimated_offset(first_volume_sample: int, *, decimation: int) -> int:
    """Move an arm's first-volume offset onto the decimated grid.

    An offset that is not a whole number of decimated samples would shift that
    arm against the others by a fraction of a sample, so it is refused rather
    than rounded.
    """
    if decimation < 1:
        raise HarmoniseError("decimation must be at least one")
    if first_volume_sample % decimation:
        raise HarmoniseError(
            f"a first volume at sample {first_volume_sample} does not land on the "
            f"decimated grid of {decimation}"
        )
    return first_volume_sample // decimation
