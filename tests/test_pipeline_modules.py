from pathlib import Path

import numpy as np

import mri_correction.pipeline as pipeline_module
from mri_correction import pipeline_io, pipeline_markers, pipeline_provenance
from mri_correction.pipeline import PipelineInputError
from mri_correction.pipeline_types import PipelineInputError as SharedPipelineInputError
from mri_correction.window import OutputWindow


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
