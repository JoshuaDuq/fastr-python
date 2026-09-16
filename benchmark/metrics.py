"""Measure what a correction removed, and what it removed along with it.

Two measurements, always reported together. Residual at the scanner harmonics
says how much artifact is left. Tone transfer says how much signal survived at
those same frequencies. Either alone is misleading, because template
subtraction, residual PCA, and adaptive cancellation all reduce the first by
damaging the second.

Tone transfer is measured by difference. The corrected recording still holds
residual artifact and real EEG at the probe frequency, so projecting it alone
would measure all three at once. Correcting the same recording twice, once with
the tones and once without, and subtracting leaves only what happened to the
tones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from benchmark.probes import Tone as ToneLike
from fastr_python.quality.harmonics import block_coherent_harmonic_rms
from fastr_python.validation.metrics import ToneTransfer, tone_transfer

VOLTS_PER_MICROVOLT = 1e-6


class MetricError(ValueError):
    """Raised when a measurement is asked for something it cannot measure."""


@dataclass(frozen=True, slots=True)
class ToneResult:
    """How one probe frequency survived one arm, on one channel."""

    frequency_hz: float
    placement: str
    channel: str
    amplitude_ratio: float
    phase_error_degrees: float


def harmonic_orders(
    *,
    repetition_time_seconds: float,
    max_hz: float,
    mains_hz: float,
    mains_guard_hz: float,
) -> np.ndarray:
    """List the volume-harmonic orders a residual is measured at.

    Harmonics landing within the guard of mains are dropped, because power-line
    interference at those frequencies is not artifact any correction here is
    being asked to remove.
    """
    if not math.isfinite(repetition_time_seconds) or repetition_time_seconds <= 0:
        raise MetricError("repetition time must be finite and positive")
    if not math.isfinite(max_hz) or max_hz <= 0:
        raise MetricError("the highest measured frequency must be finite and positive")
    fundamental = 1.0 / repetition_time_seconds
    orders = np.arange(1, int(max_hz / fundamental) + 1)
    clear = np.abs(orders * fundamental - mains_hz) >= mains_guard_hz
    if not clear.any():
        raise MetricError("the mains guard excluded every harmonic")
    return orders[clear]


def coherent_residual_microvolts(
    data: np.ndarray,
    *,
    sampling_rate: float,
    repetition_time_seconds: float,
    orders: np.ndarray,
    block_seconds: float,
) -> np.ndarray:
    """Measure harmonic-locked amplitude per channel in each complete block.

    This is acquisition-locked power, which includes whatever neural activity
    sits at those frequencies. It is a residual measurement, not an estimate of
    artifact alone, which is why nothing here reads it without also reading the
    transfer beside it.
    """
    amplitudes = block_coherent_harmonic_rms(
        data,
        sampling_rate=sampling_rate,
        fundamental_hz=1.0 / repetition_time_seconds,
        harmonic_orders=orders,
        block_seconds=block_seconds,
    )
    return amplitudes / VOLTS_PER_MICROVOLT


def measure_tone_transfer(
    clean: np.ndarray,
    injected: np.ndarray,
    probe: np.ndarray,
    *,
    channel_names: list[str],
    tones: tuple[ToneLike, ...],
    sampling_rate: float,
) -> tuple[ToneResult, ...]:
    """Recover each injected tone from the difference between two corrections.

    ``clean`` and ``injected`` are the same recording corrected by the same arm
    with the same settings, differing only in whether the tones were present.
    ``probe`` is the waveform that was added.
    """
    if clean.shape != injected.shape:
        raise MetricError("the two corrections must have the same shape")
    if clean.shape[0] != len(channel_names):
        raise MetricError("every channel must be named")
    if probe.shape[-1] != clean.shape[1]:
        raise MetricError("the probe waveform must span the measured window")
    difference = injected - clean
    results: list[ToneResult] = []
    for tone in tones:
        for index, name in enumerate(channel_names):
            transfer = _one_tone(
                probe, difference[index], tone.frequency_hz, sampling_rate
            )
            results.append(
                ToneResult(
                    frequency_hz=tone.frequency_hz,
                    placement=tone.placement,
                    channel=name,
                    amplitude_ratio=transfer.amplitude_ratio,
                    phase_error_degrees=transfer.phase_error_degrees,
                )
            )
    return tuple(results)


def _one_tone(
    probe: np.ndarray, recovered: np.ndarray, frequency: float, sampling_rate: float
) -> ToneTransfer:
    """Compare one recovered frequency against the waveform that carried it."""
    return tone_transfer(
        probe, recovered, frequency=frequency, sampling_rate=sampling_rate
    )
