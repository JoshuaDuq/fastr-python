"""Crop every arm to one span, so a metric compares corrections not lengths.

The arms do not agree on where a corrected recording ends. FASTR-Python stops
at the last volume marker, the retrimmed originals run one repetition past it,
the MATLAB harness pads whole volumes around the requested range, and FACETpy
applies its own convention. Comparing them as written would charge each arm for
its output length as if it were correction quality.

The span holds an even number of whole volumes. Whole volumes keep every
artifact harmonic on an exact Fourier bin; an even count does the same for the
tones placed halfway between harmonics, which complete an integer number of
cycles only over an even number of repetitions. An odd count leaves those tones
leaking into their neighbours.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


class WindowError(ValueError):
    """Raised when the arms share no span that can be measured."""


@dataclass(frozen=True, slots=True)
class CommonWindow:
    """The span, in whole volumes, that every arm is measured over."""

    volume_count: int
    samples_per_volume: int
    sampling_rate: float

    @property
    def sample_count(self) -> int:
        """Return how many samples the span holds."""
        return self.volume_count * self.samples_per_volume


def common_window(
    volume_counts: Mapping[str, int],
    *,
    repetition_time_seconds: float,
    sampling_rate: float,
) -> CommonWindow:
    """Take the longest even run of whole volumes that every arm produced."""
    if not volume_counts:
        raise WindowError("a common window needs at least one arm")
    if not math.isfinite(repetition_time_seconds) or repetition_time_seconds <= 0:
        raise WindowError("repetition time must be finite and positive")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise WindowError("sampling rate must be finite and positive")
    exact = repetition_time_seconds * sampling_rate
    samples_per_volume = round(exact)
    if abs(exact - samples_per_volume) > 1e-9:
        raise WindowError(
            f"a volume spans {exact} samples at {sampling_rate} Hz, which is not whole"
        )
    shared = min(volume_counts.values())
    if shared < 1:
        reported = ", ".join(
            f"{arm}={count}" for arm, count in sorted(volume_counts.items())
        )
        raise WindowError(f"at least one arm produced no whole volume: {reported}")
    volume_count = shared - shared % 2
    if volume_count < 2:
        raise WindowError("the shared span must hold at least two volumes")
    return CommonWindow(
        volume_count=volume_count,
        samples_per_volume=samples_per_volume,
        sampling_rate=sampling_rate,
    )


def crop(
    data: np.ndarray, *, first_volume_sample: int, window: CommonWindow
) -> np.ndarray:
    """Cut one arm's output down to the shared span, starting at its volume 0."""
    if data.ndim != 2:
        raise WindowError("cropping expects a channel-by-sample matrix")
    if first_volume_sample < 0:
        raise WindowError("the first volume cannot start before the recording")
    stop = first_volume_sample + window.sample_count
    if stop > data.shape[1]:
        raise WindowError(
            f"the shared span needs {stop} samples but this arm wrote {data.shape[1]}"
        )
    return data[:, first_volume_sample:stop]
