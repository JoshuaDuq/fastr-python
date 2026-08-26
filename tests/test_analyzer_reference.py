import numpy as np
import pytest

from mri_correction.aas import CorrectionInputError, analyzer_reference


def _make_recording() -> tuple[np.ndarray, np.ndarray, int]:
    samples_per_volume = 10
    starts = 3 + np.arange(25) * samples_per_volume
    data = np.zeros((2, starts[-1] + samples_per_volume + 4), dtype=float)
    time = np.arange(samples_per_volume)
    shape = np.stack(
        [
            np.sin(2 * np.pi * time / samples_per_volume),
            np.cos(2 * np.pi * time / samples_per_volume),
        ]
    )
    for index, start in enumerate(starts):
        drifted_artifact = (1.0 + index / 100.0) * shape
        channel_offset = np.array([[10.0 + index], [-4.0 - index]])
        data[:, start : start + samples_per_volume] = drifted_artifact + channel_offset
    return data, starts, samples_per_volume


def _expected_epoch(
    data: np.ndarray,
    starts: np.ndarray,
    target: int,
    window_start: int,
    samples_per_volume: int,
) -> np.ndarray:
    candidates = np.stack(
        [
            data[:, start : start + samples_per_volume]
            for start in starts[window_start : window_start + 21]
        ]
    )
    baselined = candidates - candidates.mean(axis=2, keepdims=True)
    template = baselined.mean(axis=0)
    start = starts[target]
    return data[:, start : start + samples_per_volume] - template


def test_analyzer_reference_uses_shifted_complete_edge_windows() -> None:
    data, starts, samples_per_volume = _make_recording()

    corrected = analyzer_reference(
        data,
        starts,
        samples_per_volume=samples_per_volume,
        window_size=21,
    )

    np.testing.assert_allclose(
        corrected[:, starts[0] : starts[0] + samples_per_volume],
        _expected_epoch(data, starts, 0, 0, samples_per_volume),
    )
    np.testing.assert_allclose(
        corrected[:, starts[12] : starts[12] + samples_per_volume],
        _expected_epoch(data, starts, 12, 2, samples_per_volume),
    )
    np.testing.assert_allclose(
        corrected[:, starts[24] : starts[24] + samples_per_volume],
        _expected_epoch(data, starts, 24, 4, samples_per_volume),
    )


def test_analyzer_reference_preserves_samples_outside_volume_epochs() -> None:
    data, starts, samples_per_volume = _make_recording()

    corrected = analyzer_reference(
        data,
        starts,
        samples_per_volume=samples_per_volume,
        window_size=21,
    )

    np.testing.assert_array_equal(corrected[:, : starts[0]], data[:, : starts[0]])
    np.testing.assert_array_equal(
        corrected[:, starts[-1] + samples_per_volume :],
        data[:, starts[-1] + samples_per_volume :],
    )
    assert not np.shares_memory(corrected, data)


@pytest.mark.parametrize("window_size", [0, 2, 20, 27])
def test_analyzer_reference_rejects_invalid_window_size(window_size: int) -> None:
    data, starts, samples_per_volume = _make_recording()

    with pytest.raises(CorrectionInputError):
        analyzer_reference(
            data,
            starts,
            samples_per_volume=samples_per_volume,
            window_size=window_size,
        )


def test_analyzer_reference_rejects_incomplete_final_epoch() -> None:
    data, starts, samples_per_volume = _make_recording()

    with pytest.raises(CorrectionInputError):
        analyzer_reference(
            data[:, :-5],
            starts,
            samples_per_volume=samples_per_volume,
            window_size=21,
        )
