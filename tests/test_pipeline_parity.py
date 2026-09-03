import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_pipeline import make_fixture

import fastr_python.pipeline.acquisition as acquisition_module
import fastr_python.pipeline.channels as channels_module
import fastr_python.pipeline.runner as pipeline_module
from fastr_python.config import ConfigurationError, load_config
from fastr_python.fastr import AncCorrection, ResidualObsCorrection
from fastr_python.pipeline import run_correction


def update_config(config_path: Path, **processing: object) -> None:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["processing"].update(processing)
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")


def read_provenance(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_repairs_markers_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    config_path = make_fixture(tmp_path, gap=True)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["timing"].update(
        missing_volume_markers="repair",
        expected_volume_count=8,
    )
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    summary = run_correction(load_config(config_path))

    markers = read_provenance(summary.provenance_json)["markers"]
    assert markers["volume_marker_repair"] == {
        "mode": "repair",
        "detected_volume_count": 7,
        "repaired_volume_count": 1,
        "used_volume_count": 8,
    }


def test_default_pipeline_does_not_run_marker_repair_or_anc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acquisition_module,
        "repair_volume_starts",
        lambda *args, **kwargs: pytest.fail("marker repair ran under defaults"),
    )
    monkeypatch.setattr(
        channels_module,
        "adaptive_noise_cancel",
        lambda *args, **kwargs: pytest.fail("ANC ran under defaults"),
    )

    run_correction(load_config(make_fixture(tmp_path)))


def test_pipeline_forwards_pre_trigger_fraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    update_config(config_path, pre_trigger_fraction=0.25)
    observed: list[float] = []
    original = pipeline_module.prepare_fastr_geometry

    def capture_fraction(*args: object, **kwargs: object) -> object:
        observed.append(kwargs["pre_trigger_fraction"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "prepare_fastr_geometry",
        capture_fraction,
    )

    summary = run_correction(load_config(config_path))

    assert observed == [0.25]
    provenance = read_provenance(summary.provenance_json)
    assert provenance["fastr"]["pre_trigger_fraction"] == 0.25


def test_pipeline_runs_obs_before_anc_and_records_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = make_fixture(tmp_path)
    update_config(
        config_path,
        residual_obs=True,
        residual_obs_rank="auto",
        residual_obs_section_seconds=0.3,
        adaptive_noise_cancellation=True,
    )
    calls: list[str] = []
    anc_arguments: list[dict[str, object]] = []

    def fit_obs(
        data: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> ResidualObsCorrection:
        calls.append("obs")
        assert kwargs["rank"] == "auto"
        assert kwargs["section_seconds"] == 0.3
        ranks = np.ones((data.shape[0], 2), dtype=np.int64)
        return ResidualObsCorrection(data=data, selected_ranks=ranks)

    def fit_anc(
        data: np.ndarray,
        artifact: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> AncCorrection:
        calls.append("anc")
        anc_arguments.append(kwargs)
        assert artifact.shape == data.shape
        channel_count = data.shape[0]
        return AncCorrection(
            data=data,
            reference_scales=np.arange(channel_count, dtype=np.float64) + 1.0,
            step_sizes=np.full(channel_count, 0.01),
            filter_order=kwargs["filter_order"],
        )

    monkeypatch.setattr(channels_module, "fit_residual_obs", fit_obs)
    monkeypatch.setattr(channels_module, "adaptive_noise_cancel", fit_anc)

    summary = run_correction(load_config(config_path))

    assert calls == ["obs", "anc", "obs", "anc"]
    assert all(arguments["sample_slice"].step is None for arguments in anc_arguments)
    provenance = read_provenance(summary.provenance_json)
    obs = provenance["fastr"]["residual_obs"]
    assert obs["rank_mode"] == "auto"
    assert obs["section_seconds"] == 0.3
    assert obs["selected_ranks"] == [[1, 1], [1, 1], [1, 1]]
    anc = provenance["fastr"]["adaptive_noise_cancellation"]
    assert anc["enabled"] is True
    assert anc["filter_order"] > 0
    assert anc["reference_scales"] == [1.0, 2.0, 1.0]
    assert anc["step_sizes"] == [0.01, 0.01, 0.01]
    assert provenance["fastr"]["reference"] == {
        "repository": "sccn/fMRIb",
        "commit": "2aa522bc5ec4215f42b3ba8efdb2b84d2a312935",
    }


def test_pipeline_supports_no_low_pass_without_rate_conversion(
    tmp_path: Path,
) -> None:
    config_path = make_fixture(tmp_path)
    update_config(
        config_path,
        lowpass_hz=0.0,
        output_sampling_rate_hz=1_000.0,
    )

    summary = run_correction(load_config(config_path))

    assert summary.output_sampling_rate_hz == 1_000.0
    assert summary.output_sample_count == summary.input_sample_count
    provenance = read_provenance(summary.provenance_json)
    assert provenance["output"]["psd_settings"]["fmax_hz"] == 100.0


def test_anc_cancels_against_a_low_passed_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fmrib_fastr.m` low-passes cleanEEG and Noise before cancelling.

    Cancelling first leaves the LMS reference carrying artifact out to the input
    Nyquist, which shrinks the 0.05/(N*var(refs)) step size and spends the
    filter's taps on content the low-pass then discards.
    """
    config_path = make_fixture(tmp_path)
    update_config(config_path, adaptive_noise_cancellation=True)
    seen: list[np.ndarray] = []

    def capture(
        data: np.ndarray,
        artifact: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> AncCorrection:
        seen.append(np.array(artifact, copy=True))
        channel_count = data.shape[0]
        return AncCorrection(
            data=data,
            reference_scales=np.zeros(channel_count),
            step_sizes=np.zeros(channel_count),
            filter_order=kwargs["filter_order"],
        )

    monkeypatch.setattr(channels_module, "adaptive_noise_cancel", capture)
    config = load_config(config_path)
    run_correction(config)

    assert seen
    rate = 1000.0
    cutoff = config.processing.lowpass_hz
    for reference in seen:
        spectrum = np.abs(np.fft.rfft(reference, axis=1)) ** 2
        freqs = np.fft.rfftfreq(reference.shape[1], 1.0 / rate)
        stop = freqs > cutoff * 1.5
        keep = (freqs > 0.0) & (freqs <= cutoff)
        assert spectrum[:, stop].sum() < 1e-4 * spectrum[:, keep].sum()


def test_anc_requires_an_output_low_pass(tmp_path: Path) -> None:
    """MATLAB forces a 70 Hz cutoff here; overriding a stated one is worse."""
    config_path = make_fixture(tmp_path)
    update_config(
        config_path,
        adaptive_noise_cancellation=True,
        lowpass_hz=0.0,
        output_sampling_rate_hz=1000.0,
    )

    with pytest.raises(ConfigurationError, match="adaptive_noise_cancellation"):
        load_config(config_path)
