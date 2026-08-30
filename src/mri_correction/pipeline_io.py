"""Input validation and signal preparation for the correction pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
from scipy.signal import oaconvolve

from .config import CorrectionConfig
from .pipeline_types import PipelineInputError
from .window import OutputWindow


def validate_input_files(config: CorrectionConfig) -> None:
    for path, label in (
        (config.input.raw_vhdr, "raw BrainVision header"),
        (config.input.fmri_metadata, "fMRI metadata"),
    ):
        if not path.is_file():
            raise PipelineInputError(f"{label} does not exist: {path}")


def output_paths(output_vhdr: Path) -> dict[str, Path]:
    stem = output_vhdr.with_suffix("")
    return {
        "vhdr": output_vhdr,
        "eeg": output_vhdr.with_suffix(".eeg"),
        "vmrk": output_vhdr.with_suffix(".vmrk"),
        "json": output_vhdr.with_suffix(".json"),
        "psd_before": stem.with_name(f"{stem.name}_psd_before.png"),
        "psd_after": stem.with_name(f"{stem.name}_psd_after.png"),
    }


def validate_output_paths(output_paths: dict[str, Path]) -> None:
    output_paths["vhdr"].parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"correction output already exists: {joined}")


def validate_rates(
    input_rate: float,
    output_rate: float,
    lowpass_hz: float,
) -> tuple[float, int]:
    if output_rate <= 0.0 or not math.isfinite(output_rate):
        raise PipelineInputError("output sampling rate must be finite and positive")
    ratio = input_rate / output_rate
    decimation = round(ratio)
    if not math.isclose(ratio, decimation, rel_tol=0.0, abs_tol=1e-9):
        raise PipelineInputError(
            "input and output sampling rates must have an integer ratio"
        )
    if decimation < 1:
        raise PipelineInputError("output sampling rate cannot exceed input rate")
    if lowpass_hz >= min(input_rate, output_rate) / 2.0:
        raise PipelineInputError(
            "lowpass_hz must be below both input and output Nyquist frequencies"
        )
    return float(output_rate), decimation


def resolve_reference_channel(
    channel_names: list[str],
    reference: str | int,
) -> int:
    if isinstance(reference, str):
        if reference not in channel_names:
            raise PipelineInputError(
                f"reference channel is not present: {reference!r}"
            )
        return channel_names.index(reference)
    if isinstance(reference, bool) or not isinstance(reference, int):
        raise PipelineInputError("reference channel must be a name or index")
    if reference < 0 or reference >= len(channel_names):
        raise PipelineInputError(
            f"reference channel index is outside the channel range: {reference}"
        )
    return reference


def make_output_low_pass(sampling_rate: float, lowpass_hz: float) -> np.ndarray:
    """Design the anti-alias low-pass applied before decimation.

    `fmrib_fastr.m` builds a least-squares FIR and runs it through `filtfilt`,
    which squares the response and so doubles its passband ripple. A windowed
    linear-phase FIR run once is flat to 0.03 dB across the band the cutoff
    names and reaches deeper into the stop band, so `lowpass_hz` describes what
    the output actually keeps. MNE designs it; the symmetric odd-length result
    is zero-phase once its group delay is removed.
    """
    if lowpass_hz >= 0.5 * sampling_rate:
        raise PipelineInputError(
            "lowpass_hz must stay below the input Nyquist frequency"
        )
    return mne.filter.create_filter(
        None,
        sfreq=sampling_rate,
        l_freq=None,
        h_freq=lowpass_hz,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        verbose="ERROR",
    )


def lowpass_and_decimate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    window: OutputWindow,
) -> np.ndarray:
    """Low-pass the whole array, then take the output window and decimate.

    Filtering before slicing keeps the filter's edge transient outside the
    emitted span. Slicing before decimating anchors the decimation phase to the
    window start, so the output sample grid does not shift. Reflecting the
    recording across both ends before convolving stops an untrimmed run, whose
    emitted span reaches sample zero, from being faded in from nothing.
    """
    ratio = round(sampling_rate / output_sampling_rate)
    taps = make_output_low_pass(sampling_rate, lowpass_hz)
    pad = (taps.size - 1) // 2
    reflected = np.pad(
        data,
        ((0, 0), (pad, pad)),
        mode="reflect",
        reflect_type="odd",
    )
    filtered = oaconvolve(reflected, taps[np.newaxis, :], mode="same", axes=1)
    return filtered[:, pad + window.start : pad + window.stop : ratio]


def remove_line_noise(
    data: np.ndarray,
    *,
    sampling_rate: float,
    frequencies_hz: Sequence[float],
) -> np.ndarray:
    """Regress configured stationary sinusoids without widening the notch."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    nyquist = 0.5 * sampling_rate
    if (
        frequencies.ndim != 1
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(frequencies >= nyquist)
    ):
        raise ValueError(
            "line-noise frequencies must be finite, positive, and below "
            "the Nyquist frequency"
        )
    return mne.filter.notch_filter(
        data,
        Fs=sampling_rate,
        freqs=frequencies,
        method="spectrum_fit",
        filter_length="10s",
        verbose="ERROR",
    )
