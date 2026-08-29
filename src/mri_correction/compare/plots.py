"""Uncorrected vs FASTR PSD overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .pairs import RecordingPair


def load_vhdr(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")


class AlignmentError(RuntimeError):
    """The two recordings do not share a common time origin."""


def volume_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    """Onsets of the MR volume markers, in seconds from the file start."""
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    is_volume = np.char.startswith(np.char.lower(descriptions), "volume")
    return np.asarray(raw.annotations.onset, dtype=float)[is_volume]


def verify_shared_origin(
    uncorrected: mne.io.BaseRaw,
    fastr: mne.io.BaseRaw,
    *,
    tolerance_seconds: float = 0.005,
) -> None:
    """Confirm both files are trimmed to the same first volume.

    ``align_to_fastr`` crops from sample 0, which only yields a meaningful
    comparison if sample 0 is the same instant in both recordings. Rather
    than assume that, check it against the volume markers and fail loudly
    when it does not hold.
    """
    left = volume_onsets(uncorrected)
    right = volume_onsets(fastr)
    if left.size == 0 or right.size == 0:
        raise AlignmentError(
            "cannot verify alignment: volume markers are missing from "
            f"{'uncorrected' if left.size == 0 else 'FASTR'} recording"
        )
    shared = min(left.size, right.size)
    deviation = float(np.max(np.abs(left[:shared] - right[:shared])))
    if deviation > tolerance_seconds:
        raise AlignmentError(
            "uncorrected and FASTR volume markers disagree by "
            f"{deviation * 1e3:.1f} ms (tolerance "
            f"{tolerance_seconds * 1e3:.1f} ms); the recordings are not "
            "trimmed to the same first volume"
        )


def align_to_fastr(
    uncorrected: mne.io.BaseRaw, fastr: mne.io.BaseRaw
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    """Crop both recordings to a common, verified time base.

    Returns new objects; neither argument is modified in place.
    """
    aligned = uncorrected
    fastr_fs = float(fastr.info["sfreq"])
    if not np.isclose(float(aligned.info["sfreq"]), fastr_fs):
        aligned = aligned.copy().resample(fastr_fs, verbose="ERROR")
    verify_shared_origin(aligned, fastr)
    n_times = min(aligned.n_times, fastr.n_times)
    tmax = (n_times - 1) / fastr_fs
    if aligned.n_times > n_times:
        aligned = aligned.copy().crop(tmin=0.0, tmax=tmax)
    cropped_fastr = fastr
    if fastr.n_times > n_times:
        cropped_fastr = fastr.copy().crop(tmin=0.0, tmax=tmax)
    return aligned, cropped_fastr


def _eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    names = raw.ch_names
    if "ECG" in names:
        return np.array(
            [index for index, name in enumerate(names) if name != "ECG"]
        )
    return np.arange(len(names))


def eeg_rms(raw: mne.io.BaseRaw) -> float:
    """RMS over EEG channels only.

    ECG carries ~50x the amplitude of EEG here, so including it would make
    the metric report the ECG channel rather than the correction.
    """
    data = raw.get_data(picks=_eeg_indices(raw)) * 1e6
    return float(np.sqrt(np.mean(np.square(data))))


def mean_eeg_psd(
    raw: mne.io.BaseRaw, *, max_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    data = raw.get_data(picks=_eeg_indices(raw)) * 1e6
    fs = float(raw.info["sfreq"])
    nperseg = min(int(fs * 3), data.shape[1])
    freqs, pxx = welch(data, fs=fs, nperseg=nperseg, axis=1)
    keep = freqs <= max_hz
    return freqs[keep], np.mean(pxx[:, keep], axis=0)


def plot_psd(
    traces: dict[str, mne.io.BaseRaw],
    *,
    title: str,
    output: Path,
    max_hz: float,
) -> None:
    styles = {
        "Uncorrected": ("C1-", "Uncorrected"),
        "FASTR": ("C3--", "FASTR"),
    }
    plt.figure(figsize=(6, 6))
    plt.title(title)
    for key, (style, label) in styles.items():
        if key not in traces:
            continue
        freqs, pxx = mean_eeg_psd(traces[key], max_hz=max_hz)
        plt.semilogy(freqs, pxx, style, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD ($\mu V^2/Hz)$")
    plt.xlim(0, max_hz)
    plt.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="png")
    plt.close()


def band_power(freqs: np.ndarray, pxx: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(pxx[mask]))


def metrics_row(
    pair: RecordingPair, traces: dict[str, mne.io.BaseRaw], *, max_hz: float
) -> dict[str, object]:
    row: dict[str, object] = {
        "bids_id": pair.bids_id,
        "key": pair.key,
        "idx_run": pair.idx_run,
    }
    psds = {
        name: mean_eeg_psd(raw, max_hz=max_hz) for name, raw in traces.items()
    }
    uncorr_f, uncorr_p = psds["Uncorrected"]
    row["rms_uncorrected"] = eeg_rms(traces["Uncorrected"])
    if "FASTR" in traces:
        row["rms_fastr"] = eeg_rms(traces["FASTR"])
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "gradient_20hz": (18.0, 22.0),
        "gradient_40hz": (38.0, 42.0),
    }
    for band, (low, high) in bands.items():
        uncorr_band = band_power(uncorr_f, uncorr_p, low, high)
        row[f"{band}_uncorrected"] = uncorr_band
        if "FASTR" not in psds:
            continue
        freqs, pxx = psds["FASTR"]
        value = band_power(freqs, pxx, low, high)
        row[f"{band}_fastr"] = value
        row[f"{band}_fastr_ratio"] = (
            value / uncorr_band if uncorr_band else None
        )
    return row
