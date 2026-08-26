import numpy as np
import pytest

from mri_correction.bcg_config import DetectorConfig
from mri_correction.cardiac import CardiacInputError, detect_r_peaks


def make_ecg(
    sampling_rate_hz: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(round(duration_seconds * sampling_rate_hz), dtype=float)
    signal = np.zeros(samples.size, dtype=float)
    peak_seconds = np.array([0.8, 1.65, 2.53, 3.44, 4.37, 5.31])
    for index, peak_second in enumerate(peak_seconds):
        centre = peak_second * sampling_rate_hz
        width = 0.008 * sampling_rate_hz
        sign = -1.0 if index == 4 else 1.0
        signal += sign * np.exp(-0.5 * ((samples - centre) / width) ** 2)
        t_wave = centre + 0.28 * sampling_rate_hz
        signal += 0.65 * np.exp(
            -0.5 * ((samples - t_wave) / (0.035 * sampling_rate_hz)) ** 2
        )
    signal += 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    return signal, np.rint(peak_seconds * sampling_rate_hz).astype(np.int64)


@pytest.fixture
def detector_config() -> DetectorConfig:
    return DetectorConfig(
        ecg_channel="ECG",
        preprocessing_band_hz=(7.0, 40.0),
        teager_emphasis_hz=10.0,
        teager_smoothing_seconds=0.028,
        template_window_seconds=(-0.2, 0.4),
        minimum_rr_seconds=0.4,
        maximum_rr_seconds=1.5,
        candidate_refractory_seconds=0.25,
        candidate_prominence_mad=3.0,
        correlation_threshold=0.5,
        refinement_iterations=2,
    )


def assert_peaks_match(
    detected: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance_samples: int,
) -> None:
    assert detected.shape == expected.shape
    assert np.all(np.abs(detected - expected) <= tolerance_samples)


def test_detector_recovers_known_qrs_positions(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert np.all(np.diff(detection.peak_samples) > 0)
    assert detection.quality.status == "ok"
    assert detection.quality.candidate_count > detection.quality.accepted_count
    assert detection.quality.rejected_double_mark > 0


def test_detector_is_invariant_to_global_ecg_polarity(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(-ecg, 1_000.0, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)


def test_detector_rejects_t_wave_candidates(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)
    t_wave_positions = expected + 280

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert not np.any(
        np.min(np.abs(detection.peak_samples[:, None] - t_wave_positions), axis=1)
        <= 10
    )


def test_detector_is_deterministic_and_annotation_independent(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)
    external_marker_train = np.array([800, 2530, 3440, 4370, 5310])
    first = detect_r_peaks(ecg, 1_000.0, config=detector_config)
    second = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert first.peak_samples.tobytes() == second.peak_samples.tobytes()
    assert first.quality == second.quality
    assert external_marker_train.size < first.peak_samples.size
    assert "annotations" not in detect_r_peaks.__code__.co_varnames


def test_detector_enforces_candidate_refractory_interval(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    minimum_distance = round(
        detector_config.candidate_refractory_seconds * 1_000.0
    )
    assert np.all(np.diff(detection.peak_samples) >= minimum_distance)


def test_detector_handles_amplitude_drift_and_deterministic_noise(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(6_000, dtype=float)
    expected = np.rint(
        np.array([0.8, 1.65, 2.53, 3.44, 4.37, 5.31]) * sampling_rate_hz
    ).astype(np.int64)
    signal = np.zeros(samples.size, dtype=float)
    noise = np.random.default_rng(20260826).normal(0.0, 0.02, samples.size)
    for index, peak in enumerate(expected):
        amplitude = 0.75 + 0.1 * index
        sign = -1.0 if index == 4 else 1.0
        signal += sign * amplitude * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        signal += 0.65 * amplitude * np.exp(
            -0.5 * ((samples - peak - 280.0) / 35.0) ** 2
        )
    signal += 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    signal += noise

    detection = detect_r_peaks(signal, sampling_rate_hz, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)


@pytest.mark.parametrize(
    "ecg",
    [
        np.zeros((2, 1000)),
        np.array([0.0, np.nan, 1.0]),
        np.array([True, False]),
    ],
)
def test_detector_rejects_invalid_ecg(
    detector_config: DetectorConfig,
    ecg: np.ndarray,
) -> None:
    with pytest.raises(CardiacInputError):
        detect_r_peaks(ecg, 1_000.0, config=detector_config)


def test_detector_rejects_invalid_sampling_rate(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)

    with pytest.raises(CardiacInputError, match="sampling rate"):
        detect_r_peaks(ecg, 0.0, config=detector_config)
