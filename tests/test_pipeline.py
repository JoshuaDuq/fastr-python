import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
import yaml
from pybv import write_brainvision
from scipy.signal import butter, filtfilt

import mri_correction.pipeline as pipeline_module
from mri_correction.brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from mri_correction.config import load_config
from mri_correction.pipeline import PipelineInputError, run_correction
from mri_correction.window import OutputWindow


def make_fixture(tmp_path: Path, *, gap: bool = False) -> Path:
    data = np.zeros((3, 800), dtype=np.float64)
    data[0, 120:140] = 1e-5
    data[1, 120:140] = 2e-5
    data[2, 120:140] = 3e-5
    write_brainvision(
        data=data,
        sfreq=1_000.0,
        ch_names=["EEG 001", "EEG 002", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        unit="µV",
        events=[],
    )
    marker_positions = [1, 101, 201, 301, 401, 501, 601]
    if gap:
        marker_positions[3:] = [position + 100 for position in marker_positions[3:]]
    markers = (
        BrainVisionMarker("New Segment", "", 1, 1, 0),
        *(
            BrainVisionMarker("Volume", "volume-start", position, 1, 0)
            for position in marker_positions
        ),
        BrainVisionMarker("Comment", "preserve me", 300, 2, 2),
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(marker_path, "source.eeg", markers)

    (tmp_path / "bold.json").write_text(
        json.dumps(
            {
                "RepetitionTime": 0.1,
                "SliceTiming": [0.0, 0.05],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "input": {
            "raw_vhdr": "source.vhdr",
            "fmri_metadata": "bold.json",
        },
        "output": {"vhdr": "corrected.vhdr"},
        "timing": {
            "marker_type": "Volume",
            "marker_description": "volume-start",
        },
        "processing": {
            "method": "acquisition_group_fastr",
            "interpolation_factor": 2,
            "neighbor_count": 2,
            "search_radius_samples": 0,
            "lowpass_hz": 20.0,
            "output_sampling_rate_hz": 500.0,
            "channel_batch_size": 2,
            "reference_channel": "EEG 001",
            "line_noise_frequencies_hz": [],
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_correction_writes_reopenable_output_and_provenance(
    tmp_path: Path,
) -> None:
    config = load_config(make_fixture(tmp_path))

    summary = run_correction(config)

    raw = mne.io.read_raw_brainvision(
        summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    )
    assert raw.info["sfreq"] == 500.0
    assert raw.get_data().shape == (3, 400)
    assert np.all(np.isfinite(raw.get_data()))
    assert summary.channel_count == 3
    assert summary.marker_count == 9
    assert summary.skipped_group_count > 0
    assert summary.provenance_json.is_file()
    assert summary.psd_before == tmp_path / "corrected_psd_before.png"
    assert summary.psd_before.is_file()
    assert summary.psd_after == tmp_path / "corrected_psd_after.png"
    assert summary.psd_after.is_file()

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert provenance["method"] == "acquisition_group_fastr"
    assert provenance["input"]["raw_vhdr"].endswith("source.vhdr")
    assert provenance["output"]["psd_before"] == str(summary.psd_before)
    assert provenance["output"]["psd_after"] == str(summary.psd_after)
    assert provenance["output"]["psd_settings"] == {
        "fmax_hz": 20.0,
        "n_fft": None,
    }
    assert provenance["output"]["psd_interval_seconds"]["start"] > 0.0
    assert provenance["output"]["psd_interval_seconds"]["end"] < 0.8
    assert provenance["fastr"]["alignment"]["shifts"]
    assert provenance["residual_qc"]["block_seconds"] == 30.0
    assert provenance["residual_qc"]["mains_frequency_hz"] == 60.0
    assert provenance["residual_qc"]["mains_exclusion_hz"] == 1.0
    assert "Comment/preserve me" in set(raw.annotations.description)


def test_channel_batch_size_does_not_change_output(tmp_path: Path) -> None:
    first_config_path = make_fixture(tmp_path / "first")
    second_config_path = make_fixture(tmp_path / "second")
    first_config = load_config(first_config_path)
    second_values = yaml.safe_load(second_config_path.read_text(encoding="utf-8"))
    second_values["output"]["vhdr"] = "corrected_one_channel.vhdr"
    second_values["processing"]["channel_batch_size"] = 1
    second_config_path.write_text(
        yaml.safe_dump(second_values),
        encoding="utf-8",
    )
    second_config = load_config(second_config_path)

    first_summary = run_correction(first_config)
    second_summary = run_correction(second_config)
    first = mne.io.read_raw_brainvision(
        first_summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    ).get_data()
    second = mne.io.read_raw_brainvision(
        second_summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    ).get_data()

    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)


def test_run_correction_rejects_marker_gap_before_creating_output(
    tmp_path: Path,
) -> None:
    config = load_config(make_fixture(tmp_path, gap=True))

    with pytest.raises(PipelineInputError, match="acquisition gap"):
        run_correction(config)

    assert not config.output.vhdr.exists()
    assert not config.output.vhdr.with_suffix(".eeg").exists()


def test_run_correction_refuses_existing_sidecar(tmp_path: Path) -> None:
    config = load_config(make_fixture(tmp_path))
    config.output.vhdr.with_suffix(".json").write_text(
        "occupied",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        run_correction(config)


def test_psd_diagnostics_use_only_corrected_time_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(make_fixture(tmp_path))
    calls: list[tuple[float, float]] = []
    fmax_values: list[float] = []
    original = pipeline_module._save_psd_plot

    def capture_window(
        raw: mne.io.BaseRaw,
        output_path: Path,
        *,
        fmax: float,
        title: str,
        tmin: float,
        tmax: float,
        n_fft: int | None = None,
    ) -> None:
        calls.append((tmin, tmax))
        fmax_values.append(fmax)
        original(
            raw,
            output_path,
            fmax=fmax,
            title=title,
            tmin=tmin,
            tmax=tmax,
            n_fft=n_fft,
        )

    monkeypatch.setattr(pipeline_module, "_save_psd_plot", capture_window)

    run_correction(config)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert fmax_values == [20.0, 20.0]
    assert calls[0][0] > 0.0
    assert calls[0][1] < 0.8


def test_pipeline_forwards_configured_psd_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["diagnostics"] = {
        "psd_max_frequency_hz": 400.0,
        "psd_n_fft": 128,
    }
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = load_config(config_path)
    calls: list[tuple[float, int | None]] = []

    def capture_psd(
        raw: mne.io.BaseRaw,
        output_path: Path,
        *,
        fmax: float,
        title: str,
        tmin: float,
        tmax: float,
        n_fft: int | None = None,
    ) -> None:
        calls.append((fmax, n_fft))
        output_path.touch()

    monkeypatch.setattr(pipeline_module, "_save_psd_plot", capture_psd)

    run_correction(config)

    assert calls == [(20.0, 128), (20.0, 128)]


def test_pipeline_applies_configured_line_noise_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"]["line_noise_frequencies_hz"] = [10.0]
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = load_config(config_path)
    calls: list[tuple[float, tuple[float, ...]]] = []

    def capture_line_noise(
        data: np.ndarray,
        *,
        sampling_rate: float,
        frequencies_hz: tuple[float, ...],
    ) -> np.ndarray:
        calls.append((sampling_rate, frequencies_hz))
        return data

    monkeypatch.setattr(
        pipeline_module.pipeline_io,
        "remove_line_noise",
        capture_line_noise,
    )

    run_correction(config)

    assert calls == [(500.0, (10.0,))]


def test_pipeline_forwards_configured_fastr_robustness_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"].update(
        {
            "residual_gate": True,
            "adaptive_window": True,
            "local_neighbor_count": 2,
            "residual_gate_mad_multiplier": 6.0,
            "residual_gate_ratio": 5.0,
            "residual_gate_max_fraction": 0.5,
            "adaptive_improvement_ratio": 0.9,
        }
    )
    values["quality_control"] = {
        "mains_frequency_hz": 50.0,
        "mains_exclusion_hz": 0.5,
    }
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = load_config(config_path)
    seen_gate: dict[str, object] = {}
    seen_adaptive: dict[str, object] = {}
    original_gate = pipeline_module.gate_fastr_geometry
    original_adaptive = pipeline_module.adapt_fastr_geometry

    def capture_gate(*args: object, **kwargs: object) -> object:
        seen_gate.update(kwargs)
        return original_gate(*args, **kwargs)

    def capture_adaptive(*args: object, **kwargs: object) -> object:
        seen_adaptive.update(kwargs)
        return original_adaptive(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "gate_fastr_geometry", capture_gate)
    monkeypatch.setattr(
        pipeline_module,
        "adapt_fastr_geometry",
        capture_adaptive,
    )

    run_correction(config)

    assert seen_gate == {
        "template_high_pass_hz": 1.0,
        "sampling_rate": 1_000.0,
        "residual_gate_mad_multiplier": 6.0,
        "residual_gate_ratio": 5.0,
        "residual_gate_max_fraction": 0.5,
        "mains_frequency_hz": 50.0,
        "mains_exclusion_hz": 0.5,
    }
    assert seen_adaptive == {
        "local_neighbor_count": 2,
        "template_high_pass_hz": 1.0,
        "sampling_rate": 1_000.0,
        "adaptive_improvement_ratio": 0.9,
    }


def test_psd_plot_preparation_assigns_standard_channel_locations() -> None:
    info = mne.create_info(
        ["Fp1", "Fp2", "unmapped"],
        sfreq=1_000.0,
        ch_types=["eeg", "eeg", "eeg"],
    )
    raw = mne.io.RawArray(np.zeros((3, 100)), info, verbose="ERROR")

    prepared = pipeline_module._prepare_psd_raw(raw)

    assert prepared.ch_names == ["Fp1", "Fp2"]
    assert prepared.get_montage() is not None
    assert np.isfinite(prepared.get_montage().get_positions()["ch_pos"]["Fp1"]).all()


def test_psd_plot_requests_spatial_colors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = mne.create_info(["Fp1", "Fp2"], sfreq=1_000.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose="ERROR")
    seen: dict[str, object] = {}

    def capture_plot(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(
        pipeline_module.mne.viz,
        "plot_raw_psd",
        capture_plot,
    )
    pipeline_module._save_psd_plot(
        raw,
        tmp_path / "psd.png",
        fmax=100.0,
        title="test",
        tmin=0.0,
        tmax=0.1,
    )

    assert seen["spatial_colors"] is True
    assert seen["fmax"] == 100.0
    assert "n_fft" not in seen


def test_psd_plot_forwards_a_configured_fft_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = mne.create_info(["Fp1", "Fp2"], sfreq=1_000.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose="ERROR")
    seen: dict[str, object] = {}

    def capture_plot(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(
        pipeline_module.mne.viz,
        "plot_raw_psd",
        capture_plot,
    )
    pipeline_module._save_psd_plot(
        raw,
        tmp_path / "psd.png",
        fmax=100.0,
        title="test",
        tmin=0.0,
        tmax=0.1,
        n_fft=128,
    )

    assert seen["n_fft"] == 128


def test_lowpass_and_decimate_anchors_phase_to_the_window_start() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, 1000))
    coefficients = butter(2, 100.0, fs=5000.0)
    filtered = filtfilt(*coefficients, data, axis=1)

    actual = pipeline_module._lowpass_and_decimate(
        data,
        sampling_rate=5000.0,
        output_sampling_rate=1000.0,
        lowpass_hz=100.0,
        window=OutputWindow(start=13, stop=913),
    )

    assert actual.shape == (2, 180)
    assert np.array_equal(actual, filtered[:, 13:913:5])
    # decimating first and slicing afterwards would start three samples late
    assert not np.array_equal(actual[:, 0], filtered[:, ::5][:, 3])


def test_lowpass_and_decimate_full_window_matches_the_legacy_stride() -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 500))
    coefficients = butter(2, 100.0, fs=5000.0)
    expected = filtfilt(*coefficients, data, axis=1)[:, ::5]

    actual = pipeline_module._lowpass_and_decimate(
        data,
        sampling_rate=5000.0,
        output_sampling_rate=1000.0,
        lowpass_hz=100.0,
        window=OutputWindow(start=0, stop=500),
    )

    assert np.array_equal(actual, expected)


def make_untrimmed_fixture(
    tmp_path: Path,
    *,
    head: int,
    tail: int,
    trim_mode: str = "first_to_last_volume",
) -> Path:
    """A recording that brackets the scan, as the untrimmed sources do."""
    marker_positions = [1 + head + 100 * index for index in range(7)]
    sample_count = head + (marker_positions[-1] - marker_positions[0] + 1) + tail
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, sample_count)) * 1e-6
    for position in marker_positions:
        start = position - 1
        data[:, start : start + 40] += 5e-5
    write_brainvision(
        data=data,
        sfreq=1_000.0,
        ch_names=["EEG 001", "EEG 002", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        unit="µV",
        events=[],
    )
    markers = (
        BrainVisionMarker("New Segment", "", 1, 1, 0),
        *(
            BrainVisionMarker("Volume", "volume-start", position, 1, 0)
            for position in marker_positions
        ),
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(marker_path, "source.eeg", markers)

    (tmp_path / "bold.json").write_text(
        json.dumps(
            {
                "RepetitionTime": 0.1,
                "SliceTiming": [0.0, 0.05],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "input": {"raw_vhdr": "source.vhdr", "fmri_metadata": "bold.json"},
        "output": {"vhdr": "corrected.vhdr"},
        "timing": {
            "marker_type": "Volume",
            "marker_description": "volume-start",
        },
        "processing": {
            "method": "acquisition_group_fastr",
            "interpolation_factor": 2,
            "neighbor_count": 2,
            "search_radius_samples": 0,
            "lowpass_hz": 20.0,
            "output_sampling_rate_hz": 500.0,
            "channel_batch_size": 2,
            "reference_channel": "EEG 001",
            "line_noise_frequencies_hz": [],
        },
        "trim": {"mode": trim_mode},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_trimmed_run_emits_the_first_to_last_volume_span(tmp_path: Path) -> None:
    config = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))

    summary = run_correction(config)

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    trim = provenance["trim"]
    assert trim["mode"] == "first_to_last_volume"
    assert trim["window_start_sample"] == 200
    assert trim["window_stop_sample"] == 801
    assert trim["window_length"] == 601
    assert trim["head_margin_samples"] == 200
    assert trim["tail_margin_samples"] == 200
    assert summary.output_sample_count == (601 - 1) // 2 + 1
    assert summary.input_sample_count == 1001

    raw = mne.io.read_raw_brainvision(
        summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    )
    assert raw.get_data().shape == (3, 301)
    assert np.all(np.isfinite(raw.get_data()))


def test_trimmed_run_puts_the_first_volume_marker_on_the_first_sample(
    tmp_path: Path,
) -> None:
    config = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))

    summary = run_correction(config)

    _, markers = read_brainvision_markers(summary.output_vmrk)
    volume_positions = [
        marker.position for marker in markers if marker.marker_type == "Volume"
    ]
    assert volume_positions[0] == 1
    assert volume_positions == [1 + 50 * index for index in range(7)]


def test_trimmed_run_uses_the_margin_to_correct_the_boundary_volumes(
    tmp_path: Path,
) -> None:
    with_margin = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))
    summary = run_correction(with_margin)

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    trim = provenance["trim"]
    assert trim["head_margin_samples"] >= trim["required_head_margin_samples"]
    assert trim["tail_margin_samples"] >= trim["required_tail_margin_samples"]
    assert provenance["markers"]["skipped_group_indices"] == []


def test_uncorrected_boundary_spans_are_annotated(tmp_path: Path) -> None:
    config = load_config(make_fixture(tmp_path))

    summary = run_correction(config)

    assert summary.skipped_group_count > 0
    _, markers = read_brainvision_markers(summary.output_vmrk)
    bad = [marker for marker in markers if marker.description == "Bad_Gradient"]
    assert bad
    assert all(marker.marker_type == "Bad Interval" for marker in bad)
    assert all(1 <= marker.position <= summary.output_sample_count for marker in bad)
    assert all(
        marker.position + marker.size - 1 <= summary.output_sample_count
        for marker in bad
    )


def test_a_fully_corrected_run_carries_no_bad_gradient_annotation(
    tmp_path: Path,
) -> None:
    config = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))

    summary = run_correction(config)

    assert summary.skipped_group_count == 0
    _, markers = read_brainvision_markers(summary.output_vmrk)
    assert not [
        marker for marker in markers if marker.description == "Bad_Gradient"
    ]


def test_residual_qc_is_reported_in_the_sidecar(tmp_path: Path) -> None:
    config = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))

    summary = run_correction(config)

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    qc = provenance["residual_qc"]
    assert qc["floor_uv"] == 1.0
    assert qc["mad_multiplier"] == 6.0
    assert qc["minimum_channels"] == 4
    assert qc["block_seconds"] == 30.0  # 300 volumes of 0.1 s
    assert qc["volumes_per_block"] == 300
    assert qc["channel_names"] == ["EEG 001", "EEG 002", "ECG"]
    assert len(qc["block_residual_uv"]) == summary.channel_count
    assert len(qc["worst_block_uv"]) == summary.channel_count
    # a 0.3 s output holds no complete 30 s block, so there is nothing to report
    assert qc["block_residual_uv"] == [[], [], []]
    assert qc["flagged_blocks"] == []
    assert qc["flagged_block_count"] == 0


def test_residual_qc_excludes_the_mains_harmonic(tmp_path: Path) -> None:
    config = load_config(make_untrimmed_fixture(tmp_path, head=200, tail=200))

    summary = run_correction(config)

    qc = json.loads(summary.provenance_json.read_text(encoding="utf-8"))["residual_qc"]
    # 2 groups per 0.1 s volume gives a 20 Hz slice rate, whose 3rd harmonic is mains
    assert 20.0 in qc["harmonics_hz"]
    assert 40.0 in qc["harmonics_hz"]
    assert 60.0 not in qc["harmonics_hz"]
