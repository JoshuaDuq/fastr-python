import numpy as np
import pytest

from mri_correction.bcg import (
    BcgCorrectionConfig,
    BcgInputError,
    correct_bcg,
)


def make_recording() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    sampling_rate_hz = 1_000.0
    sample_count = 5_000
    samples = np.arange(sample_count, dtype=float)
    peak_samples = np.array([800, 1_600, 2_400, 3_200, 4_000], dtype=np.int64)
    artifact_anchors = peak_samples + 200
    clean = np.vstack(
        (
            2e-6 + 2e-7 * np.sin(2.0 * np.pi * samples / 173.0),
            -3e-6 + 3e-7 * np.cos(2.0 * np.pi * samples / 211.0),
            np.zeros(sample_count),
        )
    )
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    data = clean.copy()
    for anchor in artifact_anchors:
        data[0] += 25e-6 * np.roll(artifact, anchor)
        data[1] += 12e-6 * np.roll(artifact, anchor)
    return data, clean, sampling_rate_hz, peak_samples


def correction_config(method: str) -> BcgCorrectionConfig:
    return BcgCorrectionConfig(
        method=method,
        window_seconds=(-0.1, 0.2),
        ecg_to_bcg_delay_seconds=0.2,
        aas_neighbor_count=2,
        pca_obs_components=1,
    )


@pytest.mark.parametrize("method", ["aas", "pca_obs"])
def test_correction_reduces_heartbeat_locked_artifact_and_preserves_boundaries(
    method: str,
) -> None:
    data, clean, sampling_rate_hz, peak_samples = make_recording()

    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config(method),
    )

    corrected = result.data_volts
    samples = result.corrected_samples
    before_error = np.sqrt(np.mean((data[:2, samples] - clean[:2, samples]) ** 2))
    after_error = np.sqrt(
        np.mean((corrected[:2, samples] - clean[:2, samples]) ** 2)
    )
    assert after_error < before_error
    outside = np.ones(data.shape[1], dtype=bool)
    outside[samples] = False
    np.testing.assert_array_equal(corrected[:, outside], data[:, outside])
    np.testing.assert_array_equal(corrected[2], data[2])
    assert result.method == method


def test_pca_obs_restores_input_means_outside_splice() -> None:
    data, _, sampling_rate_hz, peak_samples = make_recording()
    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("pca_obs"),
    )

    outside = np.ones(data.shape[1], dtype=bool)
    outside[result.corrected_samples] = False
    np.testing.assert_allclose(
        result.data_volts[:2, outside].mean(axis=1),
        data[:2, outside].mean(axis=1),
        rtol=0.0,
        atol=0.0,
    )


def test_pca_obs_requires_effective_component_support() -> None:
    data, _, sampling_rate_hz, _ = make_recording()
    with pytest.raises(BcgInputError, match="n_components"):
        correct_bcg(
            data,
            np.array([800, 1_600], dtype=np.int64),
            sampling_rate_hz,
            channel_names=["EEG 001", "EEG 002", "ECG"],
            eeg_picks=np.array([0, 1], dtype=np.int64),
            ecg_channel_index=2,
            config=BcgCorrectionConfig(
                method="pca_obs",
                window_seconds=(-0.1, 0.2),
                ecg_to_bcg_delay_seconds=0.2,
                aas_neighbor_count=2,
                pca_obs_components=2,
            ),
        )
