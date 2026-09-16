"""Add known tones to a recording so signal loss can be measured, not assumed.

Suppression at the scanner harmonics is not evidence of a good correction:
template subtraction, residual PCA, and adaptive cancellation all remove
neural signal at the frequencies they clean. A tone of known amplitude and
phase, added before correction and recovered after it, measures that loss
directly.

Tones sit at two placements relative to the volume comb. One lands exactly on
a harmonic, which is the worst case for any comb-shaped correction. The other
lands halfway between two harmonics, the furthest a frequency can be from the
comb, which is the best case. Reporting both bounds the loss instead of
sampling one arbitrary point between them.

Every tone goes into one recording rather than one recording per tone: the
frequencies are distinct and the projection separates them, which turns what
would be one extra correction pass per tone into one pass overall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from fastr_python.io.recording import (
    read_brainvision_recording,
    write_brainvision_recording,
)

ON_HARMONIC = "on_volume_harmonic"
BETWEEN_HARMONICS = "between_volume_harmonics"

# Spread across the band a gradient correction is expected to preserve, and
# clear of 60 Hz mains. Each becomes two tones, one per placement.
DEFAULT_BAND_CENTRES_HZ = (7.0, 13.0, 23.0, 31.0, 43.0, 53.0, 71.0, 83.0, 93.0)

# Successive multiples of the golden ratio spread phases evenly, so the tones
# never sum into a single transient at the first sample.
_PHASE_STEP = (math.sqrt(5.0) - 1.0) / 2.0


class ProbeError(ValueError):
    """Raised when a tone plan cannot be placed on the requested recording."""


@dataclass(frozen=True, slots=True)
class Tone:
    """One injected frequency, and where it sits relative to the volume comb."""

    frequency_hz: float
    amplitude_volts: float
    placement: str
    phase_radians: float


def plan_tones(
    *,
    repetition_time_seconds: float,
    amplitude_volts: float,
    band_centres_hz: tuple[float, ...] = DEFAULT_BAND_CENTRES_HZ,
    mains_hz: float,
    mains_guard_hz: float = 1.0,
) -> tuple[Tone, ...]:
    """Place one on-harmonic and one between-harmonic tone per band centre.

    Frequencies within the mains guard are dropped rather than moved, so that a
    tone is never silently measured against power-line interference.
    """
    if not math.isfinite(repetition_time_seconds) or repetition_time_seconds <= 0:
        raise ProbeError("repetition time must be finite and positive")
    if not math.isfinite(amplitude_volts) or amplitude_volts <= 0:
        raise ProbeError("tone amplitude must be finite and positive")
    volume_rate_hz = 1.0 / repetition_time_seconds
    planned: list[Tone] = []
    for centre in band_centres_hz:
        harmonic = round(centre / volume_rate_hz)
        for placement, order in (
            (ON_HARMONIC, harmonic),
            (BETWEEN_HARMONICS, harmonic + 0.5),
        ):
            frequency = order * volume_rate_hz
            if abs(frequency - mains_hz) < mains_guard_hz:
                continue
            planned.append(
                Tone(
                    frequency_hz=frequency,
                    amplitude_volts=amplitude_volts,
                    placement=placement,
                    phase_radians=2 * math.pi * ((len(planned) * _PHASE_STEP) % 1.0),
                )
            )
    if not planned:
        raise ProbeError("no tone survived the mains guard")
    return tuple(planned)


def synthesize_tones(
    tones: tuple[Tone, ...], *, sample_count: int, sampling_rate: float
) -> np.ndarray:
    """Build the exact waveform that will be added to the recording."""
    if sample_count < 1:
        raise ProbeError("a tone waveform needs at least one sample")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ProbeError("sampling rate must be finite and positive")
    if any(tone.frequency_hz >= sampling_rate / 2 for tone in tones):
        raise ProbeError("every tone must sit below Nyquist")
    times = np.arange(sample_count, dtype=np.float64) / sampling_rate
    waveform = np.zeros(sample_count)
    for tone in tones:
        waveform += tone.amplitude_volts * np.sin(
            2 * np.pi * tone.frequency_hz * times + tone.phase_radians
        )
    return waveform


def write_tone_injected_recording(
    source_vhdr: Path,
    output_vhdr: Path,
    *,
    tones: tuple[Tone, ...],
    excluded_channels: tuple[str, ...],
) -> None:
    """Copy a recording with tones added to every channel but the excluded ones.

    Markers pass through untouched: the tone-injected copy has to present the
    same acquisition timing as the original, or the arms would be correcting
    two different recordings.
    """
    recording = read_brainvision_recording(source_vhdr)
    raw = mne.io.read_raw_brainvision(source_vhdr, preload=True, verbose="error")
    data = raw.get_data()
    waveform = synthesize_tones(
        tones, sample_count=data.shape[1], sampling_rate=raw.info["sfreq"]
    )
    injected = [name not in excluded_channels for name in raw.ch_names]
    if not any(injected):
        raise ProbeError("every channel is excluded, so no tone would be injected")
    data[injected] += waveform
    write_brainvision_recording(
        data=data,
        sampling_rate=raw.info["sfreq"],
        channel_names=raw.ch_names,
        output_vhdr=output_vhdr,
        markers=recording.markers,
    )
