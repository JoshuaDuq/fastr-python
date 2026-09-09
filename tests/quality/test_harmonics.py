import numpy as np
import pytest
from scipy.signal import welch

from fastr_python.quality.harmonics import mean_harmonic_psd


def test_exact_harmonic_density_matches_welch_for_multiple_segments():
    rate = 250.0
    data = np.random.default_rng(81).normal(size=(3, 2500))
    frequencies, power = welch(data, rate, window="hann", nperseg=500, noverlap=250)

    exact = mean_harmonic_psd(
        data,
        sampling_rate=rate,
        fundamental_hz=5.0,
        harmonic_count=10,
        segment_samples=500,
    )

    indices = np.searchsorted(frequencies, 5.0 * np.arange(1, 11))
    np.testing.assert_allclose(exact, power[:, indices], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("frequency", [10.0, 10.5])
def test_coherent_blocks_distinguish_volume_harmonics_from_nearby_tones(frequency):
    from fastr_python.quality.harmonics import block_coherent_harmonic_rms

    rate = 250.0
    times = np.arange(30 * 250) / rate
    data = (100.0 * np.sin(2 * np.pi * frequency * times))[np.newaxis, :]

    measured = block_coherent_harmonic_rms(
        data,
        sampling_rate=rate,
        fundamental_hz=1.0,
        harmonic_orders=np.arange(1, 41),
        block_seconds=30.0,
    )

    if frequency == 10.0:
        assert measured[0, 0] == pytest.approx(100 / np.sqrt(2), rel=0.001)
    else:
        assert measured[0, 0] < 0.01


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_rate": np.nan},
        {"sampling_rate": 0.0},
        {"fundamental_hz": 0.0},
        {"harmonic_count": 0},
        {"harmonic_count": 25},
        {"segment_samples": 1},
        {"segment_samples": 501},
    ],
)
def test_harmonic_estimation_rejects_invalid_geometry(overrides):
    arguments = dict(
        sampling_rate=250.0, fundamental_hz=5.0, harmonic_count=10, segment_samples=500
    )
    arguments.update(overrides)
    with pytest.raises(ValueError):
        mean_harmonic_psd(np.ones((1, 500)), **arguments)


def test_harmonic_estimation_rejects_non_finite_data():
    with pytest.raises(ValueError, match="finite"):
        mean_harmonic_psd(
            np.full((1, 500), np.nan),
            sampling_rate=250.0,
            fundamental_hz=5.0,
            harmonic_count=10,
            segment_samples=500,
        )


@pytest.mark.parametrize("block_seconds", [1.0, 2.0, 30.0])
def test_coherent_rms_does_not_count_neighboring_harmonics_twice(block_seconds):
    from fastr_python.quality.harmonics import block_coherent_harmonic_rms

    rate = 250.0
    times = np.arange(round(block_seconds * rate)) / rate
    data = (
        100 * np.sin(2 * np.pi * 10 * times + 0.3)
        + 50 * np.cos(2 * np.pi * 11 * times + 0.8)
    )[np.newaxis, :]

    measured = block_coherent_harmonic_rms(
        data,
        sampling_rate=rate,
        fundamental_hz=1.0,
        harmonic_orders=np.arange(1, 101),
        block_seconds=block_seconds,
    )

    assert measured[0, 0] == pytest.approx(np.sqrt((100**2 + 50**2) / 2), rel=1e-10)


@pytest.mark.parametrize(
    "orders",
    [
        [0, 10],
        [-1, 10],
        [10, 10],
        [10, 9],
        [1.5, 2.0],
        [125],
    ],
)
def test_coherent_rms_rejects_invalid_harmonic_orders(orders):
    from fastr_python.quality.harmonics import block_coherent_harmonic_rms

    with pytest.raises(ValueError, match="harmonic"):
        block_coherent_harmonic_rms(
            np.ones((1, 250)),
            sampling_rate=250.0,
            fundamental_hz=1.0,
            harmonic_orders=np.array(orders),
            block_seconds=1.0,
        )
