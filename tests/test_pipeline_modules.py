from pathlib import Path

import numpy as np
import pytest

import eegfmri_fastr.pipeline as pipeline_module
from eegfmri_fastr import pipeline_io, pipeline_markers, pipeline_provenance
from eegfmri_fastr.pipeline import PipelineInputError
from eegfmri_fastr.pipeline_types import PipelineInputError as SharedPipelineInputError
from eegfmri_fastr.residual_qc import (
    ResidualQcDefaults,
    residual_qc_defaults,
)
from eegfmri_fastr.window import OutputWindow


def test_pipeline_keeps_existing_private_test_seams() -> None:
    assert callable(pipeline_module._lowpass_and_decimate)
    assert callable(pipeline_module._save_psd_plot)
    assert callable(pipeline_module._measure_residual_qc)


def test_pipeline_helpers_are_available_in_focused_modules() -> None:
    assert callable(pipeline_io.lowpass_and_decimate)
    assert callable(pipeline_markers.bad_gradient_markers)
    assert callable(pipeline_provenance.make_provenance)


def test_pipeline_exception_identity_is_preserved() -> None:
    assert PipelineInputError is SharedPipelineInputError


def test_residual_qc_defaults_keep_their_public_values() -> None:
    assert residual_qc_defaults is ResidualQcDefaults
    assert ResidualQcDefaults.MAD_MULTIPLIER == 6.0
    assert ResidualQcDefaults.MINIMUM_CHANNELS == 4
    assert ResidualQcDefaults.FLOOR_UV == 1.0


def test_pipeline_wrapper_delegates_to_lowpass_helper(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        pipeline_io,
        "lowpass_and_decimate",
        lambda *args, **kwargs: sentinel,
    )

    result = pipeline_module._lowpass_and_decimate(
        np.zeros((1, 4)),
        sampling_rate=100.0,
        output_sampling_rate=100.0,
        lowpass_hz=20.0,
        window=OutputWindow(start=0, stop=4),
    )

    assert result is sentinel


def test_marker_helper_matches_the_existing_pipeline_behavior() -> None:
    spans = ((2, 5),)
    window = OutputWindow(start=0, stop=10)

    expected = pipeline_module._bad_gradient_markers(
        spans,
        window=window,
        decimation=1,
        output_sample_count=10,
    )
    actual = pipeline_markers.bad_gradient_markers(
        spans,
        window=window,
        decimation=1,
        output_sample_count=10,
    )

    assert actual == expected


def test_provenance_path_conversion_handles_nested_paths() -> None:
    value = {"path": Path("input.vhdr"), "items": (Path("a"),)}

    assert pipeline_provenance.stringify_paths(value) == {
        "path": "input.vhdr",
        "items": ["a"],
    }


def _tone_amplitude(signal: np.ndarray, frequency: float, rate: float) -> float:
    times = np.arange(signal.size) / rate
    phasor = np.exp(-2j * np.pi * frequency * times)
    return float(2.0 * np.abs(np.vdot(phasor.conj(), signal)) / signal.size)


def test_output_low_pass_keeps_the_passband_it_advertises() -> None:
    """A 100 Hz cutoff must keep the band it names, flat, all the way up.

    A gentle IIR at this corner costs 3 dB by 80 Hz, and a least-squares FIR
    applied twice ripples by 2 dB across the passband. Neither leaves the
    configured cutoff meaning what it says.
    """
    rate = 5_000.0
    cutoff = 100.0
    duration = 4.0
    times = np.arange(int(duration * rate)) / rate

    for frequency, floor in ((10.0, 0.99), (50.0, 0.99), (80.0, 0.99), (95.0, 0.98)):
        tone = np.sin(2.0 * np.pi * frequency * times)[np.newaxis, :]
        filtered = pipeline_io.lowpass_and_decimate(
            tone,
            sampling_rate=rate,
            output_sampling_rate=rate,
            lowpass_hz=cutoff,
            window=OutputWindow(start=0, stop=tone.shape[1]),
        )
        edge = int(0.5 * rate)
        kept = _tone_amplitude(filtered[0, edge:-edge], frequency, rate)
        assert kept > floor, f"{frequency} Hz kept only {kept:.3f}"


def test_output_low_pass_rejects_the_stop_band() -> None:
    rate = 5_000.0
    duration = 4.0
    times = np.arange(int(duration * rate)) / rate

    for frequency, ceiling in ((150.0, 0.02), (600.0, 1e-3)):
        tone = np.sin(2.0 * np.pi * frequency * times)[np.newaxis, :]
        filtered = pipeline_io.lowpass_and_decimate(
            tone,
            sampling_rate=rate,
            output_sampling_rate=rate,
            lowpass_hz=100.0,
            window=OutputWindow(start=0, stop=tone.shape[1]),
        )
        edge = int(0.5 * rate)
        kept = _tone_amplitude(filtered[0, edge:-edge], frequency, rate)
        assert kept < ceiling, f"{frequency} Hz kept {kept:.5f}"


def test_output_low_pass_does_not_taper_the_start_of_the_emitted_span() -> None:
    """An untrimmed run emits sample zero, so the filter must not fade it in.

    A 10 Hz tone sits well inside the passband, so the filtered output should
    track the input from the very first sample. Convolving against implicit
    zeros instead fades the first half filter length in from nothing.
    """
    rate = 5_000.0
    times = np.arange(int(4.0 * rate)) / rate
    tone = np.sin(2.0 * np.pi * 10.0 * times)[np.newaxis, :]

    filtered = pipeline_io.lowpass_and_decimate(
        tone,
        sampling_rate=rate,
        output_sampling_rate=rate,
        lowpass_hz=100.0,
        window=OutputWindow(start=0, stop=tone.shape[1]),
    )

    half = (pipeline_io.make_output_low_pass(rate, 100.0).size - 1) // 2
    np.testing.assert_allclose(filtered[0, :half], tone[0, :half], atol=0.01)
    np.testing.assert_allclose(filtered[0, -half:], tone[0, -half:], atol=0.01)


def test_zero_low_pass_returns_an_unfiltered_window_copy() -> None:
    data = np.arange(20, dtype=np.float64).reshape(2, 10)

    output = pipeline_io.lowpass_and_decimate(
        data,
        sampling_rate=500.0,
        output_sampling_rate=500.0,
        lowpass_hz=0.0,
        window=OutputWindow(start=2, stop=8),
    )

    np.testing.assert_array_equal(output, data[:, 2:8])
    assert not np.shares_memory(output, data)


def test_zero_low_pass_rejects_decimation_without_an_antialias_filter() -> None:
    with pytest.raises(PipelineInputError, match="anti-alias"):
        pipeline_io.validate_rates(
            input_rate=1_000.0,
            output_rate=500.0,
            lowpass_hz=0.0,
        )
