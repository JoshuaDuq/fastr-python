"""Public, configuration-driven BrainVision correction pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, filtfilt

from . import __version__
from .brainvision import BrainVisionMarker
from .brainvision_io import (
    BrainVisionInputError,
    BrainVisionRecording,
    read_brainvision_recording,
    resample_markers,
    select_marker_samples,
    write_brainvision_recording,
)
from .config import CorrectionConfig
from .fastr import (
    FastrAlignment,
    FastrGeometry,
    FastrInputError,
    FmriAcquisitionTiming,
    apply_fastr_batch,
    fit_fastr_alignment,
    load_bids_fmri_timing,
    make_group_trigger_samples,
    prepare_fastr_geometry,
)
from .window import OutputWindow, resolve_output_window


class PipelineInputError(ValueError):
    """Raised when a configured correction run cannot be performed."""


_PSD_MAX_FREQUENCY_HZ = 100.0


@dataclass(frozen=True, slots=True)
class CorrectionSummary:
    """Stable summary of one completed correction run."""

    output_vhdr: Path
    output_eeg: Path
    output_vmrk: Path
    provenance_json: Path
    psd_before: Path
    psd_after: Path
    method: str
    input_sampling_rate_hz: float
    output_sampling_rate_hz: float
    channel_count: int
    input_sample_count: int
    output_sample_count: int
    marker_count: int
    processed_group_count: int
    skipped_group_count: int


def run_correction(config: CorrectionConfig) -> CorrectionSummary:
    """Run the configured FASTR correction and write a BrainVision result."""
    if not isinstance(config, CorrectionConfig):
        raise PipelineInputError("config must be a CorrectionConfig instance")
    started = time.perf_counter()
    try:
        return _run_correction(config, started=started)
    except (BrainVisionInputError, FastrInputError) as error:
        raise PipelineInputError(str(error)) from error


def _run_correction(
    config: CorrectionConfig,
    *,
    started: float,
) -> CorrectionSummary:
    _validate_input_files(config)
    output_paths = _output_paths(config.output.vhdr)
    _validate_output_paths(output_paths)

    recording = read_brainvision_recording(config.input.raw_vhdr)
    raw = mne.io.read_raw_brainvision(
        recording.header_path,
        preload=False,
        verbose="ERROR",
    )
    input_rate = float(raw.info["sfreq"])
    volume_starts = select_marker_samples(
        recording.markers,
        marker_type=config.timing.marker_type,
        marker_description=config.timing.marker_description,
        sample_count=int(raw.n_times),
    )
    timing = load_bids_fmri_timing(config.input.fmri_metadata)
    group_triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=input_rate,
        timing=timing,
    )
    output_rate, decimation = _validate_rates(
        input_rate,
        config.processing.output_sampling_rate_hz,
        config.processing.lowpass_hz,
    )
    window = resolve_output_window(
        volume_starts,
        mode=config.trim.mode,
        input_sample_count=int(raw.n_times),
    )
    reference_index = _resolve_reference_channel(
        raw.ch_names,
        config.processing.reference_channel,
    )
    geometry = prepare_fastr_geometry(
        group_triggers,
        sample_count=int(raw.n_times),
        interpolation_factor=config.processing.interpolation_factor,
        neighbor_count=config.processing.neighbor_count,
        search_radius_samples=config.processing.search_radius_samples,
        groups_per_volume=timing.groups_per_volume,
        allow_edges=True,
    )
    reference_channel = raw.get_data(
        picks=[reference_index],
        start=0,
        stop=raw.n_times,
    )[0]
    alignment = fit_fastr_alignment(reference_channel, geometry)

    channel_count = len(raw.ch_names)
    input_sample_count = int(raw.n_times)
    output_sample_count = (window.length - 1) // decimation + 1
    amplitude_means = np.empty(channel_count, dtype=np.float64)
    amplitude_rms = np.empty(channel_count, dtype=np.float64)
    with tempfile.TemporaryDirectory(
        dir=config.output.vhdr.parent,
        prefix=".mri-correction-",
    ) as temporary_directory:
        output_path = Path(temporary_directory) / "corrected-output.dat"
        corrected_output = np.memmap(
            output_path,
            mode="w+",
            dtype=np.float64,
            shape=(channel_count, output_sample_count),
        )
        for start in range(0, channel_count, config.processing.channel_batch_size):
            stop = min(
                start + config.processing.channel_batch_size,
                channel_count,
            )
            batch = raw.get_data(
                picks=list(range(start, stop)),
                start=0,
                stop=raw.n_times,
            )
            correction = apply_fastr_batch(
                batch,
                geometry,
                alignment,
                template_high_pass_hz=config.processing.template_high_pass_hz,
                sampling_rate=input_rate,
            )
            amplitude_means[start:stop] = correction.provenance.amplitudes.mean(
                axis=1
            )
            amplitude_rms[start:stop] = np.sqrt(
                np.mean(correction.provenance.amplitudes**2, axis=1)
            )
            corrected_output[start:stop] = _lowpass_and_decimate(
                correction.data,
                sampling_rate=input_rate,
                output_sampling_rate=output_rate,
                lowpass_hz=config.processing.lowpass_hz,
                window=window,
            )
        corrected_output.flush()
        transformed_markers = resample_markers(
            recording.markers,
            factor=decimation,
            window=window,
        )
        _validate_marker_output_positions(
            transformed_markers,
            output_sample_count,
        )
        write_brainvision_recording(
            data=corrected_output,
            sampling_rate=output_rate,
            channel_names=raw.ch_names,
            output_vhdr=config.output.vhdr,
            markers=transformed_markers,
        )
        del corrected_output

    psd_max_frequency = min(_PSD_MAX_FREQUENCY_HZ, output_rate / 2.0)
    psd_tmin, psd_tmax = _corrected_psd_window(
        geometry,
        input_sampling_rate=input_rate,
        window=window,
    )
    window_offset_seconds = window.start / input_rate
    _save_psd_plot(
        raw,
        output_paths["psd_before"],
        title="Before scanner-gradient correction (complete epochs)",
        fmax=psd_max_frequency,
        tmin=psd_tmin + window_offset_seconds,
        tmax=psd_tmax + window_offset_seconds,
    )
    corrected_raw = mne.io.read_raw_brainvision(
        output_paths["vhdr"],
        preload=False,
        verbose="ERROR",
    )
    _save_psd_plot(
        corrected_raw,
        output_paths["psd_after"],
        title="After scanner-gradient correction (complete epochs)",
        fmax=psd_max_frequency,
        tmin=psd_tmin,
        tmax=psd_tmax,
    )

    provenance_path = output_paths["json"]
    provenance = _make_provenance(
        config,
        output_paths=output_paths,
        recording=recording,
        raw=raw,
        timing=timing,
        geometry=geometry,
        alignment=alignment,
        amplitude_means=amplitude_means,
        amplitude_rms=amplitude_rms,
        decimation=decimation,
        output_sample_count=output_sample_count,
        window=window,
        psd_tmin=psd_tmin,
        psd_tmax=psd_tmax,
        runtime_seconds=time.perf_counter() - started,
    )
    with provenance_path.open("x", encoding="utf-8") as output:
        json.dump(provenance, output, indent=2)
        output.write("\n")

    return CorrectionSummary(
        output_vhdr=output_paths["vhdr"],
        output_eeg=output_paths["eeg"],
        output_vmrk=output_paths["vmrk"],
        provenance_json=provenance_path,
        psd_before=output_paths["psd_before"],
        psd_after=output_paths["psd_after"],
        method=config.processing.method,
        input_sampling_rate_hz=input_rate,
        output_sampling_rate_hz=output_rate,
        channel_count=channel_count,
        input_sample_count=input_sample_count,
        output_sample_count=output_sample_count,
        marker_count=len(recording.markers),
        processed_group_count=geometry.triggers.size,
        skipped_group_count=geometry.skipped_group_indices.size,
    )


def _validate_input_files(config: CorrectionConfig) -> None:
    for path, label in (
        (config.input.raw_vhdr, "raw BrainVision header"),
        (config.input.fmri_metadata, "fMRI metadata"),
    ):
        if not path.is_file():
            raise PipelineInputError(f"{label} does not exist: {path}")


def _output_paths(output_vhdr: Path) -> dict[str, Path]:
    stem = output_vhdr.with_suffix("")
    return {
        "vhdr": output_vhdr,
        "eeg": output_vhdr.with_suffix(".eeg"),
        "vmrk": output_vhdr.with_suffix(".vmrk"),
        "json": output_vhdr.with_suffix(".json"),
        "psd_before": stem.with_name(f"{stem.name}_psd_before.png"),
        "psd_after": stem.with_name(f"{stem.name}_psd_after.png"),
    }


def _validate_output_paths(output_paths: dict[str, Path]) -> None:
    output_paths["vhdr"].parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"correction output already exists: {joined}")


def _validate_rates(
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


def _resolve_reference_channel(
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


def _lowpass_and_decimate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    window: OutputWindow,
) -> np.ndarray:
    """Low-pass the whole array, then take the output window and decimate.

    Filtering before slicing keeps ``filtfilt``'s edge transient outside the
    emitted span. Slicing before decimating anchors the decimation phase to the
    window start, so the output sample grid does not shift.
    """
    ratio = round(sampling_rate / output_sampling_rate)
    coefficients = butter(2, lowpass_hz, fs=sampling_rate)
    filtered = filtfilt(*coefficients, data, axis=1)
    return filtered[:, window.start : window.stop : ratio]


def _save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    *,
    fmax: float,
    title: str,
    tmin: float,
    tmax: float,
) -> None:
    plot_raw = _prepare_psd_raw(raw)
    with mne.use_log_level("ERROR"):
        figure = mne.viz.plot_raw_psd(
            plot_raw,
            fmin=0.0,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            spatial_colors=plot_raw.get_montage() is not None,
            show=False,
            n_jobs=1,
            verbose="ERROR",
        )
    try:
        for axis in figure.axes:
            if axis.get_xlabel():
                axis.set_xlim(0.0, fmax)
        figure.suptitle(title)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _prepare_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
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


def _corrected_psd_window(
    geometry: FastrGeometry,
    *,
    input_sampling_rate: float,
    window: OutputWindow,
) -> tuple[float, float]:
    """Return a PSD interval, relative to the window, holding only corrected samples.

    Both figures are drawn over the same interval so they can be compared, so
    the bounds are expressed relative to the emitted window rather than to the
    start of the input recording. The before-figure is drawn from the input
    recording and must have ``window.start`` added back.
    """
    first_sample = float(geometry.triggers[0])
    last_sample = float(
        geometry.triggers[-1]
        + geometry.epoch.samples_after / geometry.interpolation_factor
    )
    first_sample = max(first_sample, float(window.start))
    last_sample = min(last_sample, float(window.stop - 1))
    tmin = (first_sample - window.start) / input_sampling_rate
    tmax = (last_sample - window.start) / input_sampling_rate
    if not 0.0 <= tmin < tmax:
        raise PipelineInputError("the corrected PSD interval is empty")
    return tmin, tmax


def _validate_marker_output_positions(
    markers: tuple[BrainVisionMarker, ...],
    output_sample_count: int,
) -> None:
    if any(marker.position > output_sample_count for marker in markers):
        raise PipelineInputError(
            "resampled marker positions extend beyond the output recording"
        )


def _trim_provenance(
    window: OutputWindow,
    *,
    geometry: FastrGeometry,
    input_sample_count: int,
    mode: str,
) -> dict[str, object]:
    """Report the emitted window against the margin the epochs actually need."""
    factor = geometry.interpolation_factor
    required_head = math.ceil(
        (geometry.epoch.samples_before + geometry.search_radius) / factor
    )
    required_tail = math.ceil(
        (geometry.epoch.samples_after + geometry.search_radius) / factor
    )
    return {
        "mode": mode,
        "window_start_sample": window.start,
        "window_stop_sample": window.stop,
        "window_length": window.length,
        "head_margin_samples": window.start,
        "tail_margin_samples": input_sample_count - window.stop,
        "required_head_margin_samples": int(required_head),
        "required_tail_margin_samples": int(required_tail),
    }


def _make_provenance(
    config: CorrectionConfig,
    *,
    output_paths: dict[str, Path],
    recording: BrainVisionRecording,
    raw: mne.io.BaseRaw,
    timing: FmriAcquisitionTiming,
    geometry: FastrGeometry,
    alignment: FastrAlignment,
    amplitude_means: np.ndarray,
    amplitude_rms: np.ndarray,
    decimation: int,
    output_sample_count: int,
    window: OutputWindow,
    psd_tmin: float,
    psd_tmax: float,
    runtime_seconds: float,
) -> dict[str, object]:
    return {
        "package_version": __version__,
        "method": config.processing.method,
        "input": {
            "raw_vhdr": str(recording.header_path),
            "raw_data": str(recording.data_path),
            "raw_vmrk": str(recording.marker_path),
            "fmri_metadata": str(config.input.fmri_metadata),
            "sha256": {
                "vhdr": _sha256(recording.header_path),
                "eeg": _sha256(recording.data_path),
                "vmrk": _sha256(recording.marker_path),
                "fmri_metadata": _sha256(config.input.fmri_metadata),
            },
        },
        "output": {
            "vhdr": str(output_paths["vhdr"]),
            "psd_before": str(output_paths["psd_before"]),
            "psd_after": str(output_paths["psd_after"]),
            "sampling_rate_hz": float(raw.info["sfreq"] / decimation),
            "sample_count": output_sample_count,
            "psd_interval_seconds": {
                "start": psd_tmin,
                "end": psd_tmax,
            },
        },
        "trim": _trim_provenance(
            window,
            geometry=geometry,
            input_sample_count=int(raw.n_times),
            mode=config.trim.mode,
        ),
        "configuration": _jsonable_config(config),
        "timing": {
            "repetition_time_seconds": timing.repetition_time_seconds,
            "slice_timing_seconds": list(timing.slice_timing_seconds),
            "multiband_acceleration_factor": timing.multiband_acceleration_factor,
            "groups_per_volume": timing.groups_per_volume,
        },
        "markers": {
            "count": len(recording.markers),
            "processed_group_count": int(geometry.triggers.size),
            "skipped_group_indices": geometry.skipped_group_indices.tolist(),
        },
        "fastr": {
            "interpolation_factor": geometry.interpolation_factor,
            "samples_before_trigger": geometry.epoch.samples_before,
            "samples_after_trigger": geometry.epoch.samples_after,
            "search_radius_interpolated_samples": geometry.search_radius,
            "alignment": {
                "shifts": alignment.shifts.tolist(),
                "correlations": alignment.correlations.tolist(),
            },
            "amplitude_mean_by_channel": amplitude_means.tolist(),
            "amplitude_rms_by_channel": amplitude_rms.tolist(),
        },
        "runtime_seconds": runtime_seconds,
    }


def _jsonable_config(config: CorrectionConfig) -> dict[str, object]:
    values = asdict(config)
    return _stringify_paths(values)


def _stringify_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
