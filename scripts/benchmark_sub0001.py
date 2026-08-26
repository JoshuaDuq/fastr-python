#!/usr/bin/env python3
"""Reproduce the sub-0001 acquisition-slot FASTR benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt, welch

from mri_correction.fastr import (
    FmriAcquisitionTiming,
    acquisition_group_fastr_with_edges,
    load_bids_fmri_timing,
    make_group_trigger_samples,
)
from mri_correction.metrics import trigger_locked_template

_EXPECTED_RAW_SAMPLING_RATE = 5_000.0
_EXPECTED_ANALYZER_SAMPLING_RATE = 1_000.0
_LOWPASS_CUTOFF_HZ = 100.0
_GROUP_READOUT_SECONDS = 0.045


def main(argv: list[str] | None = None) -> int:
    arguments = _make_parser().parse_args(argv)
    started = time.perf_counter()
    result = run_benchmark(arguments)
    result["runtime_seconds"] = time.perf_counter() - started
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    return 0


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    timing = load_bids_fmri_timing(arguments.fmri_json)
    raw = mne.io.read_raw_brainvision(
        arguments.raw_vhdr,
        preload=False,
        verbose="ERROR",
    )
    analyzer = mne.io.read_raw_brainvision(
        arguments.analyzer_vhdr,
        preload=False,
        verbose="ERROR",
    )
    decimation = _validate_sampling_rates(
        raw.info["sfreq"],
        analyzer.info["sfreq"],
    )
    channels = _select_channels(raw.ch_names, arguments.channels)
    volume_starts = _read_volume_starts(raw)
    _validate_volume_marker_series(
        volume_starts,
        sampling_rate=raw.info["sfreq"],
        timing=timing,
    )
    samples_per_volume = _integer_samples_per_volume(
        timing.repetition_time_seconds,
        raw.info["sfreq"],
    )
    first, last = _benchmark_volume_bounds(
        volume_starts.size,
        arguments.start_volume,
        arguments.volume_count,
    )
    segment_start = int(volume_starts[first])
    segment_stop = int(volume_starts[last])
    segment_stop += samples_per_volume
    segment = raw.get_data(
        picks=channels,
        start=segment_start,
        stop=segment_stop,
    )
    local_volume_starts = volume_starts[first : last + 1] - segment_start
    correction = acquisition_group_fastr_with_edges(
        segment,
        local_volume_starts,
        sampling_rate=raw.info["sfreq"],
        timing=timing,
        neighbor_count=arguments.neighbor_count,
    )
    filtered = _lowpass_and_decimate(
        correction.data,
        raw.info["sfreq"],
        decimation,
    )
    analyzer_data = analyzer.get_data(
        picks=channels,
        start=segment_start // decimation,
        stop=segment_stop // decimation,
    )
    selected_starts = volume_starts[
        arguments.start_volume : arguments.start_volume + arguments.volume_count
    ]
    selected_starts_1k = (selected_starts - segment_start) / decimation
    methods = {
        "raw": _lowpass_and_decimate(
            segment,
            raw.info["sfreq"],
            decimation,
        ),
        "analyzer_volume_aas": analyzer_data,
        "acquisition_slot_fastr": filtered,
    }
    metrics = {
        name: _measure_methods(
            data,
            selected_starts_1k,
            timing.group_offsets_seconds,
            analyzer.info["sfreq"],
            timing.repetition_time_seconds,
        )
        for name, data in methods.items()
    }
    return {
        "method": "acquisition_slot_fastr",
        "raw_vhdr": str(arguments.raw_vhdr.resolve()),
        "analyzer_vhdr": str(arguments.analyzer_vhdr.resolve()),
        "fmri_json": str(arguments.fmri_json.resolve()),
        "channels": channels,
        "start_volume": arguments.start_volume,
        "volume_count": arguments.volume_count,
        "neighbor_count": arguments.neighbor_count,
        "sampling_rate_hz": raw.info["sfreq"],
        "analyzer_sampling_rate_hz": analyzer.info["sfreq"],
        "decimation": decimation,
        "groups_per_volume": timing.groups_per_volume,
        "skipped_group_indices": correction.provenance.skipped_group_indices.tolist(),
        "metrics": metrics,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "filter": {
            "type": "zero-phase Butterworth",
            "order_per_pass": 2,
            "cutoff_hz": 100.0,
            "decimation": decimation,
        },
    }


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the sub-0001 acquisition-slot FASTR benchmark."
    )
    parser.add_argument("--raw-vhdr", type=Path, required=True)
    parser.add_argument("--analyzer-vhdr", type=Path, required=True)
    parser.add_argument("--fmri-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-volume", type=int, default=20)
    parser.add_argument("--volume-count", type=int, default=400)
    parser.add_argument("--neighbor-count", type=int, default=30)
    parser.add_argument(
        "--channels",
        nargs="+",
        help="channels to benchmark; defaults to every channel, including ECG",
    )
    return parser


def _select_channels(
    available: list[str],
    requested: list[str] | None,
) -> list[str]:
    if requested is not None:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested channels are missing: {missing}")
        return requested
    return available


def _validate_sampling_rates(
    raw_sampling_rate: float,
    analyzer_sampling_rate: float,
) -> int:
    if not np.isclose(
        raw_sampling_rate,
        _EXPECTED_RAW_SAMPLING_RATE,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("raw recording must be sampled at 5 kHz")
    if not np.isclose(
        analyzer_sampling_rate,
        _EXPECTED_ANALYZER_SAMPLING_RATE,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Analyzer recording must be sampled at 1 kHz")
    decimation = raw_sampling_rate / analyzer_sampling_rate
    if not np.isclose(decimation, round(decimation), rtol=0.0, atol=1e-9):
        raise ValueError("raw and Analyzer rates must have an integer ratio")
    return round(decimation)


def _read_volume_starts(raw: mne.io.BaseRaw) -> np.ndarray:
    starts = [
        round(float(onset) * raw.info["sfreq"])
        for onset, description in zip(
            raw.annotations.onset,
            raw.annotations.description,
            strict=True,
        )
        if description == "Volume/V  1"
    ]
    if not starts:
        raise ValueError("no exact Volume/V  1 markers found")
    return np.asarray(starts, dtype=np.int64)


def _validate_volume_marker_series(
    volume_starts: np.ndarray,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
) -> None:
    make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )


def _integer_samples_per_volume(repetition_time: float, sampling_rate: float) -> int:
    samples = repetition_time * sampling_rate
    rounded = round(samples)
    if not np.isclose(samples, rounded, rtol=0.0, atol=1e-9):
        raise ValueError("TR does not map to an integer number of samples")
    return rounded


def _benchmark_volume_bounds(
    volume_count: int,
    start_volume: int,
    measured_count: int,
) -> tuple[int, int]:
    if start_volume < 1:
        raise ValueError("start-volume must leave one preceding volume for context")
    if measured_count < 1:
        raise ValueError("volume-count must be positive")
    first = start_volume - 1
    last = start_volume + measured_count
    if last >= volume_count:
        raise ValueError("benchmark range needs one following context volume")
    return first, last


def _lowpass_and_decimate(
    data: np.ndarray,
    sampling_rate: float,
    decimation: int,
) -> np.ndarray:
    coefficients = butter(2, _LOWPASS_CUTOFF_HZ, fs=sampling_rate)
    return filtfilt(*coefficients, data, axis=1)[:, ::decimation]


def _measure_methods(
    data: np.ndarray,
    volume_starts: np.ndarray,
    group_offsets_seconds: tuple[float, ...],
    sampling_rate: float,
    repetition_time_seconds: float,
) -> dict[str, object]:
    volume_template = trigger_locked_template(
        data,
        volume_starts,
        epoch_samples=round(repetition_time_seconds * sampling_rate),
    )
    group_templates = [
        trigger_locked_template(
            data,
            volume_starts + offset * sampling_rate,
            epoch_samples=round(_GROUP_READOUT_SECONDS * sampling_rate),
        )
        for offset in group_offsets_seconds
    ]
    group_template = np.stack(group_templates, axis=1)
    return {
        "volume_locked_rms_uv": _rms_uv(volume_template),
        "group_locked_rms_uv": _rms_uv(group_template),
        "median_1_over_tr_comb_db": _comb_metric(
            data,
            sampling_rate,
            repetition_time_seconds,
        ),
    }


def _rms_uv(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)) * 1e6)


def _comb_metric(
    data: np.ndarray,
    sampling_rate: float,
    repetition_time_seconds: float,
) -> float:
    frequencies, power = welch(
        data,
        fs=sampling_rate,
        nperseg=min(data.shape[1], round(90.0 * sampling_rate)),
        noverlap=min(data.shape[1] // 2, round(45.0 * sampling_rate)),
        axis=1,
        detrend="constant",
    )
    median_power = np.median(power, axis=0)
    harmonics = []
    for harmonic in range(1, 11):
        target = harmonic / repetition_time_seconds
        center = int(np.argmin(abs(frequencies - target)))
        shoulder = (
            (frequencies >= target - 0.05)
            & (frequencies <= target + 0.05)
            & (abs(frequencies - target) >= 0.02)
        )
        if not np.any(shoulder) or median_power[center] <= 0.0:
            raise ValueError("comb metric has no finite local shoulder")
        harmonics.append(
            10.0
            * np.log10(median_power[center] / np.median(median_power[shoulder]))
        )
    return float(np.median(harmonics))


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
