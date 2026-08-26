import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
import yaml
from pybv import write_brainvision

import mri_correction.pipeline as pipeline_module
from mri_correction.brainvision import BrainVisionMarker, write_brainvision_markers
from mri_correction.config import load_config
from mri_correction.pipeline import PipelineInputError, run_correction


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
    assert provenance["output"]["psd_interval_seconds"]["start"] > 0.0
    assert provenance["output"]["psd_interval_seconds"]["end"] < 0.8
    assert provenance["fastr"]["alignment"]["shifts"]
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
        )

    monkeypatch.setattr(pipeline_module, "_save_psd_plot", capture_window)

    run_correction(config)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert fmax_values == [100.0, 100.0]
    assert calls[0][0] > 0.0
    assert calls[0][1] < 0.8


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
