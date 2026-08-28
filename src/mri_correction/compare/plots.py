"""Uncorrected vs FASTR overlays, matching the BCG compare figure style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .pairs import RecordingPair


def load_vhdr(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")


def align_to_fastr(
    uncorrected: mne.io.BaseRaw, fastr: mne.io.BaseRaw
) -> mne.io.BaseRaw:
    """Resample and crop uncorrected EEG onto the FASTR time base."""
    aligned = uncorrected
    fastr_fs = float(fastr.info["sfreq"])
    if not np.isclose(float(aligned.info["sfreq"]), fastr_fs):
        aligned = aligned.copy().resample(fastr_fs, verbose="ERROR")
    n_times = min(aligned.n_times, fastr.n_times)
    tmax = (n_times - 1) / fastr_fs
    if aligned.n_times > n_times:
        aligned = aligned.copy().crop(tmin=0.0, tmax=tmax)
    if fastr.n_times > n_times:
        fastr.crop(tmin=0.0, tmax=tmax)
    return aligned


def _eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    names = raw.ch_names
    if "ECG" in names:
        return np.array(
            [index for index, name in enumerate(names) if name != "ECG"]
        )
    return np.arange(len(names))


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


def plot_epoch(
    traces: dict[str, mne.io.BaseRaw],
    *,
    channel: str,
    start: float,
    duration: float,
    title: str,
    output: Path,
) -> None:
    uncorrected = traces["Uncorrected"]
    if channel not in uncorrected.ch_names:
        raise ValueError(
            f"channel {channel!r} is not in {uncorrected.ch_names}"
        )
    fs = float(uncorrected.info["sfreq"])
    start_sample = int(start * fs)
    stop_sample = start_sample + int(duration * fs)
    if stop_sample > uncorrected.n_times:
        start_sample = 0
        stop_sample = min(int(duration * fs), uncorrected.n_times)
    t = np.arange(stop_sample - start_sample) / fs
    ch = uncorrected.ch_names.index(channel)
    before = uncorrected.get_data()[ch, start_sample:stop_sample] * 1e6

    plt.figure(figsize=(8, 10))
    plt.suptitle(title, fontweight="bold")

    plt.subplot(311)
    plt.title("ECG")
    if "ECG" in uncorrected.ch_names:
        ecg = (
            uncorrected.get_data(picks=["ECG"])[0, start_sample:stop_sample]
            * 1e6
        )
        plt.plot(t, ecg, "C0")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")

    plt.subplot(312)
    plt.title("Estimated gradient artifact (uncorrected - FASTR)")
    if "FASTR" in traces:
        n = min(stop_sample, traces["FASTR"].n_times)
        if n > start_sample:
            after = traces["FASTR"].get_data()[ch, start_sample:n] * 1e6
            predicted = before[: after.size] - after
            plt.plot(t[: predicted.size], predicted, "C4")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")

    plt.subplot(313)
    plt.title("Uncorrected and FASTR")
    plt.plot(t, before, "C1", label="Uncorrected")
    if "FASTR" in traces:
        n = min(stop_sample, traces["FASTR"].n_times)
        if n > start_sample:
            plt.plot(
                t[: n - start_sample],
                traces["FASTR"].get_data()[ch, start_sample:n] * 1e6,
                "C3",
                label="FASTR",
            )
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")
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
    row["rms_uncorrected"] = float(
        np.sqrt(np.mean(np.square(traces["Uncorrected"].get_data() * 1e6)))
    )
    if "FASTR" in traces:
        row["rms_fastr"] = float(
            np.sqrt(np.mean(np.square(traces["FASTR"].get_data() * 1e6)))
        )
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
