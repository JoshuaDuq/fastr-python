"""Deterministic simulated recordings with known artifact and signal content.

Used by the tests and by `eegfmri_fastr.demo`; the correction pipeline itself
never imports this module. A simulated recording is a probe with known answers,
never evidence about a real acquisition.
"""

import math

import numpy as np
import numpy.typing as npt

_SPECTRUM_HEADROOM = 0.45
_SLICE_WAVEFORM_SEED = 20260826
_PULSE_WIDTH_SECONDS = 0.02


class SimulationInputError(ValueError):
    """Raised when simulation parameters do not describe a usable recording."""


def simulate_gradient_artifact(
    group_triggers: npt.ArrayLike,
    *,
    sample_count: int,
    channel_count: int,
    sampling_rate: float,
    readout_seconds: float,
    amplitude_microvolts: float = 250.0,
    amplitude_drift: float = 0.02,
    baseline_offset: float = 0.0,
    timing_jitter_seconds: float = 0.0,
    outlier_count: int = 0,
    groups_per_volume: int | None = None,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Simulate the gradient artifact of a multiband EPI acquisition.

    The recorded artifact is an induced voltage, that is the derivative of a
    gradient waveform that starts and ends each readout at baseline. The model
    keeps that physical constraint: the burst vanishes at both ends of the
    readout and integrates to zero across it, so an artifact epoch carries no
    net offset. `baseline_offset` deliberately breaks that constraint, adding a
    smooth same-sign component that models an amplifier offset or a gradient
    that does not fully return, because the size of that offset is what governs
    how much signal an amplitude-fitting correction trades away.

    `groups_per_volume` gives every acquisition-time slot in a volume its own
    waveform, which is what a real multiband acquisition does: only the same slot
    one volume later repeats. Leave it unset to give every group one shared
    waveform, which no real acquisition produces.
    """
    triggers = _validate_triggers(group_triggers)
    _validate_recording_shape(sample_count, channel_count, sampling_rate)
    readout_samples = _validate_readout(
        readout_seconds,
        sampling_rate,
        triggers,
    )
    _validate_drift(amplitude_drift)
    if triggers[-1] + readout_samples > sample_count:
        raise SimulationInputError("the simulated groups extend beyond the recording")

    generator = np.random.default_rng(seed)
    jitter = _make_jitter(
        generator,
        triggers.size,
        timing_jitter_seconds,
        sampling_rate,
    )
    drift = 1.0 + amplitude_drift * np.arange(triggers.size) / max(triggers.size - 1, 1)
    drift *= _make_outlier_scales(generator, triggers.size, outlier_count)

    samples = np.arange(sample_count, dtype=np.float64)
    owners = np.clip(np.searchsorted(triggers, samples, side="right") - 1, 0, None)
    phases = (samples - triggers[owners] - jitter[owners]) / readout_samples
    positions = _slice_positions(owners, triggers.size, groups_per_volume)
    weights = _harmonic_weights(_harmonic_count(readout_samples), groups_per_volume)
    burst = _readout_burst(phases, baseline_offset, positions, weights)

    channel_scales = _make_channel_scales(channel_count)
    return (
        amplitude_microvolts
        * channel_scales[:, np.newaxis]
        * (drift[owners] * burst)
    )


def simulate_pulse_artifact(
    *,
    sample_count: int,
    channel_count: int,
    sampling_rate: float,
    heart_rate_hz: float = 1.2,
    amplitude_microvolts: float = 30.0,
) -> npt.NDArray[np.float64]:
    """Simulate the cardiac pulse artifact that survives gradient correction."""
    _validate_recording_shape(sample_count, channel_count, sampling_rate)
    if not math.isfinite(heart_rate_hz) or heart_rate_hz <= 0.0:
        raise SimulationInputError("heart rate must be finite and positive")

    times = np.arange(sample_count, dtype=np.float64) / sampling_rate
    beat_phase = times * heart_rate_hz
    offsets = (beat_phase - np.round(beat_phase)) / heart_rate_hz
    width = _PULSE_WIDTH_SECONDS
    shape = -(offsets / width) * np.exp(-0.5 * (offsets / width) ** 2)
    channel_scales = _make_channel_scales(channel_count)
    return amplitude_microvolts * channel_scales[:, np.newaxis] * shape




def _readout_burst(
    phases: np.ndarray,
    baseline_offset: float,
    positions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Build a sharp band-limited burst that vanishes and integrates to zero.

    Every term vanishes at both ends of the readout and integrates to zero over
    it, so the burst carries no net offset however its harmonics are weighted.
    """
    harmonics = np.arange(1, weights.shape[0] + 1)[:, np.newaxis]
    clamped = np.clip(phases, 0.0, 1.0)
    burst = np.zeros(phases.size)
    for position in range(weights.shape[1]):
        selected = positions == position
        burst[selected] = np.sum(
            weights[:, position, np.newaxis]
            * np.sin(2 * np.pi * harmonics * clamped[selected]),
            axis=0,
        )
    burst = burst + baseline_offset * np.sin(np.pi * clamped)
    return np.where((phases >= 0.0) & (phases < 1.0), burst, 0.0)


def _harmonic_weights(
    harmonic_count: int,
    groups_per_volume: int | None,
) -> np.ndarray:
    """Weight the readout harmonics, once per acquisition slot when asked.

    One shared waveform falls off as 1/m, the spectrum of a switched gradient.
    Distinct acquisition slots instead get a flat spectrum with a fixed sign
    pattern each, which is what leaves their waveforms as mutually uncorrelated
    as real acquisition slots are. The pattern has its own seed so it never
    changes with the run's jitter and outlier seed.
    """
    if groups_per_volume is None:
        harmonics = np.arange(1, harmonic_count + 1, dtype=np.float64)
        return (1.0 / harmonics)[:, np.newaxis]
    generator = np.random.default_rng(_SLICE_WAVEFORM_SEED)
    return generator.choice([-1.0, 1.0], size=(harmonic_count, groups_per_volume))


def _slice_positions(
    owners: np.ndarray,
    group_count: int,
    groups_per_volume: int | None,
) -> np.ndarray:
    if groups_per_volume is None:
        return np.zeros(owners.size, dtype=np.int64)
    if not isinstance(groups_per_volume, int) or groups_per_volume < 1:
        raise SimulationInputError("groups per volume must be a positive integer")
    if group_count % groups_per_volume:
        raise SimulationInputError("the group count must be a whole number of volumes")
    return owners % groups_per_volume


def _harmonic_count(readout_samples: float) -> int:
    """Every harmonic must stay clear of the recording's Nyquist frequency."""
    return max(int(_SPECTRUM_HEADROOM * readout_samples), 1)


def _make_channel_scales(channel_count: int) -> np.ndarray:
    """Give each channel its own fixed sensitivity, as head geometry does."""
    positions = np.arange(channel_count, dtype=np.float64)
    return 0.5 + 0.5 * np.cos(positions * 2.399963229728653)


def _make_jitter(
    generator: np.random.Generator,
    group_count: int,
    timing_jitter_seconds: float,
    sampling_rate: float,
) -> np.ndarray:
    if not math.isfinite(timing_jitter_seconds) or timing_jitter_seconds < 0.0:
        raise SimulationInputError("timing jitter must be finite and nonnegative")
    if timing_jitter_seconds == 0.0:
        return np.zeros(group_count)
    return generator.normal(
        0.0,
        timing_jitter_seconds * sampling_rate,
        size=group_count,
    )


def _make_outlier_scales(
    generator: np.random.Generator,
    group_count: int,
    outlier_count: int,
) -> np.ndarray:
    if not isinstance(outlier_count, int) or outlier_count < 0:
        raise SimulationInputError("outlier count must be a nonnegative integer")
    if outlier_count > group_count:
        raise SimulationInputError("outlier count cannot exceed the group count")
    scales = np.ones(group_count)
    corrupted = generator.choice(group_count, size=outlier_count, replace=False)
    scales[corrupted] = generator.uniform(2.0, 5.0, size=outlier_count)
    return scales


def _validate_triggers(group_triggers: npt.ArrayLike) -> np.ndarray:
    triggers = np.asarray(group_triggers, dtype=np.float64)
    if triggers.ndim != 1 or triggers.size < 2:
        raise SimulationInputError("group triggers must be one-dimensional")
    if not np.all(np.isfinite(triggers)) or triggers[0] < 0.0:
        raise SimulationInputError("group triggers must be finite and nonnegative")
    if np.any(np.diff(triggers) <= 0.0):
        raise SimulationInputError("group triggers must be strictly increasing")
    return triggers


def _validate_recording_shape(
    sample_count: int,
    channel_count: int,
    sampling_rate: float,
) -> None:
    if not isinstance(sample_count, int) or sample_count < 2:
        raise SimulationInputError("sample count must be an integer of at least two")
    if not isinstance(channel_count, int) or channel_count < 1:
        raise SimulationInputError("channel count must be a positive integer")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise SimulationInputError("sampling rate must be finite and positive")


def _validate_readout(
    readout_seconds: float,
    sampling_rate: float,
    triggers: np.ndarray,
) -> float:
    if not math.isfinite(readout_seconds) or readout_seconds <= 0.0:
        raise SimulationInputError("readout duration must be finite and positive")
    readout_samples = readout_seconds * sampling_rate
    if readout_samples > np.min(np.diff(triggers)):
        raise SimulationInputError(
            "the readout duration must fit inside the group trigger interval"
        )
    return readout_samples


def _validate_drift(amplitude_drift: float) -> None:
    if not math.isfinite(amplitude_drift) or amplitude_drift <= -1.0:
        raise SimulationInputError("amplitude drift must be finite and above -1")
