import numpy as np
import pytest

from mri_correction.matlab_comparison import ComparisonError, compare_arrays


def make_recordings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampling_rate = 500.0
    samples = np.arange(5_000, dtype=np.float64)
    neural = np.sin(2.0 * np.pi * 7.0 * samples / sampling_rate)
    scanner = 4.0 * np.sin(2.0 * np.pi * 10.0 * samples / sampling_rate)
    ecg = np.sin(2.0 * np.pi * 1.2 * samples / sampling_rate)
    raw = np.vstack([neural + scanner, ecg + scanner])
    matlab = np.vstack([neural + 0.2 * scanner, ecg + 0.2 * scanner])
    python = np.vstack([neural + 0.1 * scanner, ecg + 0.1 * scanner])
    return raw, matlab, python


def test_comparison_reports_residual_transfer_and_ecg_metrics() -> None:
    raw, matlab, python = make_recordings()

    report = compare_arrays(
        raw=raw,
        matlab=matlab,
        python=python,
        sampling_rate=500.0,
        repetition_time_seconds=0.1,
        ecg_index=1,
    )

    assert set(report) == {
        "sample_rmse",
        "scanner_harmonic_rms",
        "broadband_transfer",
        "ecg_correlation",
    }
    assert report["scanner_harmonic_rms"]["python_uv"] < report[
        "scanner_harmonic_rms"
    ]["raw_uv"]
    assert report["ecg_correlation"]["matlab"] > 0.99
    assert report["ecg_correlation"]["python"] > 0.99


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("python", np.zeros((2, 10)), "same shape"),
        ("sampling_rate", 0.0, "sampling rate"),
        ("repetition_time_seconds", np.nan, "repetition time"),
        ("ecg_index", 2, "ECG index"),
    ],
)
def test_comparison_rejects_incompatible_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    raw, matlab, python = make_recordings()
    arguments = {
        "raw": raw,
        "matlab": matlab,
        "python": python,
        "sampling_rate": 500.0,
        "repetition_time_seconds": 0.1,
        "ecg_index": 1,
    }
    arguments[field] = value

    with pytest.raises(ComparisonError, match=message):
        compare_arrays(**arguments)
