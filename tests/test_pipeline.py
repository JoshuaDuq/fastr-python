import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
import yaml
from pybv import write_brainvision
from scipy.signal import oaconvolve

import fastr_python.pipeline.channels as channels_module
import fastr_python.pipeline.io as pipeline_io
import fastr_python.pipeline.quality as pipeline_quality
import fastr_python.pipeline.runner as pipeline_module
from fastr_python.config import load_config
from fastr_python.io.brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from fastr_python.pipeline import PipelineInputError, run_correction
from fastr_python.quality.psd import prepare_psd_raw
from fastr_python.window import OutputWindow

FIXTURE_OUTPUT = "pipeline_fixture_output.npy"


def expected_fixture_output() -> np.ndarray:
    """The samples ``make_fixture`` produced before the channel path was reused.

    A golden master rather than a recomputation: the point of the extraction was
    that no sample moved, and an expectation derived from the code under test
    could not have shown that.
    """
    return np.load(Path(__file__).resolve().parent / "data" / FIXTURE_OUTPUT)


def make_fixture(
    tmp_path: Path,
    *,
    gap: bool = False,
    channel_names: list[str] | None = None,
    volume_count: int = 7,
    sample_count: int = 800,
) -> Path:
    names = channel_names or ["EEG 001", "EEG 002", "ECG"]
    data = np.zeros((len(names), sample_count), dtype=np.float64)
    for index in range(len(names)):
        data[index, 120:140] = (index + 1) * 1e-5
    write_brainvision(
        data=data,
        sfreq=1_000.0,
        ch_names=names,
        fname_base="source",
        folder_out=tmp_path,
        unit="µV",
        events=[],
    )
    marker_positions = [1 + 100 * index for index in range(volume_count)]
    if gap:
        marker_positions[5:] = [position + 100 for position in marker_positions[5:]]
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


def test_channel_batch_helper_preserves_pipeline_samples(tmp_path: Path) -> None:
    summary = run_correction(load_config(make_fixture(tmp_path)))
    raw = mne.io.read_raw_brainvision(
        summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    )

    actual = raw.get_data()
    expected = expected_fixture_output()
    # The first samples are cancellation residuals; allow one machine-scale
    # rounding unit across platforms while retaining tight relative comparison.
    absolute_tolerance = np.finfo(actual.dtype).eps * np.max(np.abs(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        atol=absolute_tolerance,
    )


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


def test_run_correction_selects_a_volume_marker_block_before_gap_validation(
    tmp_path: Path,
) -> None:
    config_path = make_fixture(tmp_path, gap=True)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["timing"].update(
        volume_marker_start_index=0,
        volume_marker_count=5,
    )
    values["trim"] = {"mode": "first_to_last_volume"}
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    summary = run_correction(load_config(config_path))

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert provenance["markers"]["volume_marker_selection"] == {
        "matching_marker_count": 7,
        "selected_marker_count": 5,
        "start_index": 0,
        "count": 5,
    }
    assert provenance["timing"]["resolved"]["volume_count"] == 5


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
            "neighbor_count": 4,
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


def test_pipeline_applies_channel_adaptive_windows_and_reports_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"].update(
        channel_adaptive_window=True,
        neighbor_count=4,
        local_neighbor_count=2,
        adaptive_improvement_ratio=0.9,
    )
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    calls: list[dict[str, object]] = []
    original = channels_module.apply_channel_adaptive_fastr_batch

    def capture_adaptive_batch(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        channels_module,
        "apply_channel_adaptive_fastr_batch",
        capture_adaptive_batch,
    )

    summary = run_correction(load_config(config_path))

    assert len(calls) == 2
    assert calls[0]["local_neighbor_count"] == 2
    assert calls[0]["adaptive_improvement_ratio"] == 0.9
    assert calls[0]["unscaled_channels"] == []
    assert calls[1]["unscaled_channels"] == [0]
    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    adaptive = provenance["fastr"]["channel_adaptive_window"]
    assert adaptive["enabled"] is True
    assert adaptive["local_neighbor_count"] == 2
    decisions = adaptive["adapted_group_indices_by_channel"]
    assert set(decisions) == {"EEG 001", "EEG 002", "ECG"}
    assert decisions["ECG"] == []
    assert adaptive["adapted_channel_count"] == sum(
        bool(indices) for indices in decisions.values()
    )


def test_pipeline_applies_local_windows_to_named_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"].update(
        neighbor_count=4,
        local_neighbor_count=2,
        local_window_channels=["EEG 002"],
    )
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    calls: list[dict[str, object]] = []
    original = channels_module.apply_selected_local_fastr_batch

    def capture_selected_local(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        channels_module,
        "apply_selected_local_fastr_batch",
        capture_selected_local,
    )

    summary = run_correction(load_config(config_path))

    assert len(calls) == 2
    assert calls[0]["local_channels"] == [1]
    assert calls[1]["local_channels"] == []
    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    local = provenance["fastr"]["local_window_channels"]
    assert local["enabled"] is True
    assert local["channels"] == ["EEG 002"]
    assert local["local_neighbor_count"] == 2
    assert local["corrected_group_count"] > 0
    assert "local_group_indices_by_channel" not in local


def test_pipeline_rejects_an_absent_local_window_channel(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"].update(
        neighbor_count=4,
        local_neighbor_count=2,
        local_window_channels=["AF4"],
    )
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(PipelineInputError, match="AF4"):
        run_correction(load_config(config_path))


def test_psd_plot_preparation_assigns_standard_channel_locations() -> None:
    info = mne.create_info(
        ["Fp1", "Fp2", "unmapped"],
        sfreq=1_000.0,
        ch_types=["eeg", "eeg", "eeg"],
    )
    raw = mne.io.RawArray(np.zeros((3, 100)), info, verbose="ERROR")

    prepared = prepare_psd_raw(raw)

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


def _reference_low_pass(data: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """The production low-pass, spelled out, so slicing order is what is tested."""
    pad = (taps.size - 1) // 2
    reflected = np.pad(
        data,
        ((0, 0), (pad, pad)),
        mode="reflect",
        reflect_type="odd",
    )
    filtered = oaconvolve(reflected, taps[np.newaxis, :], mode="same", axes=1)
    return filtered[:, pad:-pad]


def test_lowpass_and_decimate_anchors_phase_to_the_window_start() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, 1000))
    taps = pipeline_io.make_output_low_pass(5000.0, 100.0)
    filtered = _reference_low_pass(data, taps)

    actual = pipeline_io.lowpass_and_decimate(
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
    taps = pipeline_io.make_output_low_pass(5000.0, 100.0)
    expected = _reference_low_pass(data, taps)[:, ::5]

    actual = pipeline_io.lowpass_and_decimate(
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


RETRY_CHANNELS = ["EEG 001", "EEG 002", "EEG 003", "EEG 004", "ECG"]


def retry_policy_fixture(
    tmp_path: Path,
    *,
    policy: str = "retry_local_and_recommend_bad",
    processing: dict[str, object] | None = None,
    volume_count: int = 7,
    sample_count: int = 800,
) -> Path:
    """A five-channel fixture whose EEG rows can carry a spatial outlier.

    A spatial threshold is a median plus a robust deviation across the EEG
    channels of one block, so it needs at least three of them: with two, the
    median sits halfway between the pair and no member can ever exceed it.
    """
    config_path = make_fixture(
        tmp_path,
        channel_names=list(RETRY_CHANNELS),
        volume_count=volume_count,
        sample_count=sample_count,
    )
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["processing"].update(
        neighbor_count=4,
        local_neighbor_count=2,
        channel_failure_policy=policy,
    )
    values["processing"].update(processing or {})
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return config_path


def residual_measurement(rows: list[list[float]]) -> object:
    """A stand-in measurement, so the correction itself stays real."""
    return pipeline_quality._BlockResidualMeasurement(
        residuals_uv=np.asarray(rows, dtype=np.float64),
        harmonics_hz=(20.0,),
        block_seconds=30.0,
        volumes_per_block=300,
    )


def quiet_rows(
    failing: dict[int, list[float]],
    *,
    blocks: int = 5,
) -> list[list[float]]:
    rows = [[0.5] * blocks for _ in RETRY_CHANNELS]
    rows[-1] = [0.0] * blocks
    for index, values in failing.items():
        rows[index] = values
    return rows


def serve_measurements(
    monkeypatch: pytest.MonkeyPatch,
    measurements: list[object],
) -> None:
    source = iter(measurements)

    def measurement(*args: object, **kwargs: object) -> object:
        return next(source)

    monkeypatch.setattr(
        channels_module,
        "_measure_block_residuals",
        measurement,
    )
    monkeypatch.setattr(pipeline_quality, "_measure_block_residuals", measurement)


def policy_provenance(summary: object) -> dict[str, object]:
    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    return provenance["fastr"]["channel_failure_policy"]


def test_retry_policy_installs_a_materially_better_local_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = retry_policy_fixture(tmp_path)
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(quiet_rows({1: [0.5, 20.0, 15.0, 0.5, 0.5]})),
            residual_measurement([[0.5, 3.0, 4.0, 0.5, 0.5]]),
            residual_measurement(quiet_rows({1: [0.5, 3.0, 4.0, 0.5, 0.5]})),
        ],
    )

    summary = run_correction(load_config(config_path))
    policy = policy_provenance(summary)

    assert policy["enabled"] is True
    assert policy["candidate_channels"] == ["EEG 002"]
    assert policy["candidate_blocks_by_channel"] == {"EEG 002": [1, 2]}
    assert policy["accepted_local_window_channels"] == ["EEG 002"]
    assert policy["final_failed_blocks_by_channel"] == {}
    assert policy["recommended_bad_channels"] == []
    assert policy["reference_channel_recommended_bad"] is False
    retry = policy["retry_by_channel"]["EEG 002"]
    assert retry["accepted"] is True
    assert retry["reason"] == "fewer_failed_blocks_and_lower_maximum"
    assert retry["wide_failed_blocks"] == [1, 2]
    assert retry["local_failed_blocks"] == []
    assert retry["wide_maximum_uv"] == 20.0
    assert retry["local_maximum_uv"] == 4.0


def test_a_rejected_retry_leaves_every_sample_as_the_wide_pass_wrote_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({1: [0.5, 20.0, 15.0, 0.5, 0.5]})
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement(wide),
            residual_measurement([[0.5, 18.0, 14.0, 0.5, 0.5]]),
            residual_measurement(wide),
        ],
    )
    reported = run_correction(
        load_config(retry_policy_fixture(tmp_path / "reported", policy="report"))
    )
    retried = run_correction(load_config(retry_policy_fixture(tmp_path / "retried")))

    policy = policy_provenance(retried)
    assert policy["candidate_channels"] == ["EEG 002"]
    assert policy["accepted_local_window_channels"] == []
    assert policy["retry_by_channel"]["EEG 002"]["accepted"] is False
    assert policy["retry_by_channel"]["EEG 002"]["reason"] == (
        "failed_block_count_not_reduced"
    )
    assert policy["final_failed_blocks_by_channel"] == {"EEG 002": [1, 2]}
    assert policy["recommended_bad_channels"] == ["EEG 002"]

    np.testing.assert_array_equal(
        mne.io.read_raw_brainvision(
            retried.output_vhdr,
            preload=True,
            verbose="ERROR",
        ).get_data(),
        mne.io.read_raw_brainvision(
            reported.output_vhdr,
            preload=True,
            verbose="ERROR",
        ).get_data(),
    )


def test_two_persistent_failures_across_sixteen_blocks_recommend_a_bad_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({1: [0.5] * 16}, blocks=16)
    wide[1][3] = 20.0
    wide[1][9] = 18.0
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement([wide[1]]),
            residual_measurement(wide),
        ],
    )

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    policy = policy_provenance(summary)

    assert policy["final_failed_blocks_by_channel"] == {"EEG 002": [3, 9]}
    assert policy["recommended_bad_channels"] == ["EEG 002"]


def test_a_single_persistent_failure_stays_advisory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({1: [0.5] * 16}, blocks=16)
    wide[1][3] = 20.0
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement([wide[1]]),
            residual_measurement(wide),
        ],
    )

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    policy = policy_provenance(summary)

    assert policy["final_failed_blocks_by_channel"] == {"EEG 002": [3]}
    assert policy["recommended_bad_channels"] == []


def test_a_non_eeg_channel_is_never_a_retry_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ECG trace carries a QRS complex, not evidence about gradient residual."""
    rows = quiet_rows({4: [40.0] * 5})
    serve_measurements(
        monkeypatch,
        [residual_measurement(rows), residual_measurement(rows)],
    )

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    policy = policy_provenance(summary)

    assert policy["candidate_channels"] == []
    assert policy["recommended_bad_channels"] == []


def test_no_candidates_never_reaches_the_local_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = quiet_rows({})
    serve_measurements(
        monkeypatch,
        [residual_measurement(rows), residual_measurement(rows)],
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("a clean recording must not be corrected twice")

    monkeypatch.setattr(channels_module, "_process_local_retry_channel", forbidden)

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    policy = policy_provenance(summary)

    assert policy["candidate_channels"] == []
    assert policy["retry_by_channel"] == {}


def test_an_accepted_retry_updates_every_channel_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({1: [0.5, 20.0, 15.0, 0.5, 0.5]})
    local = [0.5, 3.0, 4.0, 0.5, 0.5]
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement(wide),
            residual_measurement([local]),
            residual_measurement(quiet_rows({1: local})),
        ],
    )
    stages: dict[str, object] = {
        "residual_obs": True,
        "residual_obs_rank": 1,
        "adaptive_noise_cancellation": True,
    }
    reported = run_correction(
        load_config(
            retry_policy_fixture(
                tmp_path / "reported",
                policy="report",
                processing=stages,
                volume_count=30,
                sample_count=3_200,
            )
        )
    )
    retried = run_correction(
        load_config(
            retry_policy_fixture(
                tmp_path / "retried",
                processing=stages,
                volume_count=30,
                sample_count=3_200,
            )
        )
    )

    before = json.loads(reported.provenance_json.read_text(encoding="utf-8"))["fastr"]
    after = json.loads(retried.provenance_json.read_text(encoding="utf-8"))["fastr"]
    anc_before = before["adaptive_noise_cancellation"]
    anc_after = after["adaptive_noise_cancellation"]

    assert policy_provenance(retried)["accepted_local_window_channels"] == ["EEG 002"]
    assert after["amplitude_mean_by_channel"][1] != (
        before["amplitude_mean_by_channel"][1]
    )
    assert after["amplitude_rms_by_channel"][1] != (
        before["amplitude_rms_by_channel"][1]
    )
    assert anc_after["reference_scales"][1] != anc_before["reference_scales"][1]
    assert anc_after["step_sizes"][1] != anc_before["step_sizes"][1]
    assert all(value is not None for value in anc_after["reference_scales"][:4])
    # Every untouched channel keeps exactly the diagnostic the wide pass gave it.
    for index in (0, 2, 3, 4):
        assert after["amplitude_mean_by_channel"][index] == (
            before["amplitude_mean_by_channel"][index]
        )
    assert len(after["residual_obs"]["selected_ranks"]) == len(RETRY_CHANNELS)
    assert all(len(row) == 1 for row in after["residual_obs"]["selected_ranks"])


def test_a_recommended_reference_channel_is_flagged_in_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({0: [20.0, 18.0, 0.5, 0.5, 0.5]})
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement([wide[0]]),
            residual_measurement(wide),
        ],
    )

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    policy = policy_provenance(summary)

    assert policy["candidate_channels"] == ["EEG 001"]
    assert policy["recommended_bad_channels"] == ["EEG 001"]
    assert policy["reference_channel_recommended_bad"] is True


def test_the_retry_policy_never_drops_or_interpolates_a_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = quiet_rows({1: [0.5, 20.0, 15.0, 0.5, 0.5]})
    serve_measurements(
        monkeypatch,
        [
            residual_measurement(wide),
            residual_measurement([[0.5, 3.0, 4.0, 0.5, 0.5]]),
            residual_measurement(quiet_rows({1: [0.5, 3.0, 4.0, 0.5, 0.5]})),
        ],
    )
    forbidden = ("interpolate_bads", "drop_channels")
    for name in forbidden:
        monkeypatch.setattr(
            mne.io.BaseRaw,
            name,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("the policy must not replace or remove data")
            ),
        )

    summary = run_correction(load_config(retry_policy_fixture(tmp_path)))
    data = mne.io.read_raw_brainvision(
        summary.output_vhdr,
        preload=True,
        verbose="ERROR",
    )

    assert summary.channel_count == len(RETRY_CHANNELS)
    assert data.ch_names == RETRY_CHANNELS
    assert data.get_data().shape == (len(RETRY_CHANNELS), 400)
    assert np.all(np.isfinite(data.get_data()))


def test_the_report_policy_leaves_the_sidecar_and_measurement_count_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting must stay a single measurement: the retry pass is opt-in."""
    calls: list[int] = []
    original = pipeline_quality._measure_block_residuals

    def count_measurements(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_quality,
        "_measure_block_residuals",
        count_measurements,
    )

    summary = run_correction(
        load_config(retry_policy_fixture(tmp_path, policy="report"))
    )
    policy = policy_provenance(summary)

    assert len(calls) == 1
    assert policy["policy"] == "report"
    assert policy["enabled"] is False
    assert policy["candidate_channels"] == []
    assert policy["retry_by_channel"] == {}
    assert policy["recommended_bad_channels"] == []
