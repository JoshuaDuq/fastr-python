"""Shared power spectral density diagnostics for corrected recordings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

PSD_MAX_FREQUENCY_HZ = 100.0


def save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    *,
    fmax: float,
    title: str,
    tmin: float,
    tmax: float,
    n_fft: int | None = None,
) -> None:
    """Save a Welch PSD figure with spatial channel colors when possible."""
    plot_raw = prepare_psd_raw(raw)
    fft_length, segment_length = _welch_lengths(plot_raw, tmin=tmin, tmax=tmax)
    if n_fft is not None:
        fft_length = n_fft
        segment_length = min(segment_length, n_fft)

    with mne.use_log_level("ERROR"):
        spectrum = plot_raw.compute_psd(
            method="welch",
            fmin=0.0,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            n_fft=fft_length,
            n_per_seg=segment_length,
            n_jobs=1,
            verbose="ERROR",
        )
        figure = spectrum.plot(
            spatial_colors=plot_raw.get_montage() is not None,
            show=False,
        )
    try:
        for axis in figure.axes:
            if axis.get_xlabel():
                axis.set_xlim(0.0, fmax)
        figure.suptitle(title)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _welch_lengths(
    raw: mne.io.BaseRaw,
    *,
    tmin: float,
    tmax: float,
) -> tuple[int, int]:
    data = raw.get_data(
        tmin=tmin,
        tmax=tmax,
        reject_by_annotation="NaN",
    )
    valid_samples = np.isfinite(data).all(axis=0)
    run_boundaries = np.diff(
        np.pad(valid_samples.astype(np.int8), (1, 1)),
    )
    run_starts = np.flatnonzero(run_boundaries == 1)
    run_stops = np.flatnonzero(run_boundaries == -1)
    if run_starts.size == 0:
        raise ValueError("PSD window contains no usable samples")

    shortest_run = int(np.min(run_stops - run_starts))
    fft_length = min(2048, data.shape[1])
    return fft_length, min(fft_length, shortest_run)


def prepare_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Give PSD plots standard positions when the file has no montage."""
    prepared = raw.copy()
    montage = prepared.get_montage()
    if montage is not None:
        positioned_names = _positioned_channel_names(prepared, montage)
        if len(positioned_names) >= 2:
            prepared.pick(positioned_names)
            return prepared

    standard = mne.channels.make_standard_montage("standard_1020")
    standard_names = {name.casefold() for name in standard.ch_names}
    matched_names = [
        name for name in prepared.ch_names if name.casefold() in standard_names
    ]
    if len(matched_names) >= 2:
        prepared.pick(matched_names)
        prepared.set_montage(
            standard,
            match_case=False,
            on_missing="raise",
            verbose="ERROR",
        )
    return prepared


def _positioned_channel_names(
    raw: mne.io.BaseRaw,
    montage: mne.channels.DigMontage,
) -> list[str]:
    positions = montage.get_positions()["ch_pos"]
    return [
        name
        for name in raw.ch_names
        if name in positions and np.isfinite(positions[name]).all()
    ]
