import numpy as np
import pytest

from mri_correction.metrics import (
    MetricInputError,
    band_rms_ratio,
    event_locked_rms_ratio,
    tone_transfer,
    trigger_locked_rms,
    trigger_locked_template,
)

SAMPLING_RATE = 1_000.0
VOLUME_SAMPLES = 900
SAMPLE_COUNT = 22 * VOLUME_SAMPLES
TR_HARMONIC = SAMPLING_RATE / VOLUME_SAMPLES


def make_tone(frequency: float, amplitude: float, phase: float) -> np.ndarray:
    times = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLING_RATE
    return amplitude * np.sin(2 * np.pi * frequency * times + phase)


def test_tone_transfer_reports_unit_ratio_for_an_untouched_tone() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.7)

    transfer = tone_transfer(
        injected,
        injected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(1.0, abs=1e-9)
    assert transfer.phase_error_degrees == pytest.approx(0.0, abs=1e-9)


def test_tone_transfer_measures_attenuation_and_phase_shift() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.0)
    corrected = make_tone(TR_HARMONIC, 3.0, np.deg2rad(12.0))

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(0.6, abs=1e-6)
    assert transfer.phase_error_degrees == pytest.approx(12.0, abs=1e-6)


def test_tone_transfer_ignores_energy_at_other_frequencies() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.0)
    corrected = injected + make_tone(5 * TR_HARMONIC, 40.0, 1.0)

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(1.0, abs=1e-9)


def test_tone_transfer_wraps_phase_error_to_the_shortest_rotation() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.7)
    corrected = make_tone(TR_HARMONIC, 5.0, 0.7 + np.deg2rad(-170.0))

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.phase_error_degrees == pytest.approx(-170.0, abs=1e-6)


def test_band_rms_ratio_measures_retention_inside_the_band_only() -> None:
    injected = make_tone(30.0, 4.0, 0.0)
    corrected = 0.5 * injected + make_tone(200.0, 50.0, 0.0)

    ratio = band_rms_ratio(
        injected,
        corrected,
        low=1.0,
        high=100.0,
        sampling_rate=SAMPLING_RATE,
    )

    assert ratio == pytest.approx(0.5, abs=1e-3)


def test_trigger_locked_rms_recovers_a_repeating_artifact_amplitude() -> None:
    triggers = np.arange(98) * 200.0 + 50.0
    data = np.zeros((2, SAMPLE_COUNT))
    epoch = 3.0 * np.sin(2 * np.pi * np.arange(200) / 200.0)
    for start in triggers.astype(int):
        data[:, start : start + 200] = np.stack([epoch, -2.0 * epoch])

    locked = trigger_locked_rms(data, triggers, epoch_samples=200)

    expected = np.sqrt(np.mean(epoch**2))
    np.testing.assert_allclose(locked, [expected, 2 * expected], rtol=1e-9)


def test_trigger_locked_template_returns_the_fractional_aligned_average() -> None:
    signal = np.arange(200, dtype=np.float64)
    data = signal[np.newaxis, :]

    template = trigger_locked_template(
        data,
        np.array([20.5, 60.5]),
        epoch_samples=10,
    )

    np.testing.assert_allclose(template[0], np.arange(40.5, 50.5))


def test_trigger_locked_rms_averages_away_signal_unrelated_to_triggers() -> None:
    triggers = np.arange(98) * 200.0 + 50.0
    noise = np.random.default_rng(0).normal(size=(2, SAMPLE_COUNT))

    locked = trigger_locked_rms(noise, triggers, epoch_samples=200)

    assert np.all(locked < 0.2)


def test_event_locked_rms_ratio_preserves_fractional_event_transfer() -> None:
    injected = np.zeros(SAMPLE_COUNT)
    for start in (100.5, 1_000.5, 1_900.5):
        positions = np.arange(40, dtype=np.float64) + start
        injected[np.floor(positions).astype(int)] += np.sin(
            2.0 * np.pi * (positions - start) / 40.0
        )
    corrected = 0.75 * injected

    ratio = event_locked_rms_ratio(
        injected,
        corrected,
        np.array([100.5, 1_000.5, 1_900.5]),
        epoch_samples=40,
    )

    assert ratio == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: tone_transfer(np.zeros((2, 10)), np.zeros((2, 10)),
                               frequency=1.0, sampling_rate=SAMPLING_RATE),
         "one-dimensional"),
        (lambda: tone_transfer(np.zeros(10), np.zeros(11),
                               frequency=1.0, sampling_rate=SAMPLING_RATE),
         "same length"),
        (lambda: tone_transfer(np.zeros(100), np.zeros(100),
                               frequency=600.0, sampling_rate=SAMPLING_RATE),
         "below the Nyquist"),
        (lambda: band_rms_ratio(np.zeros(100), np.zeros(100),
                                low=50.0, high=10.0, sampling_rate=SAMPLING_RATE),
         "band"),
        (lambda: trigger_locked_rms(np.zeros((2, 100)), np.array([10.0]),
                                    epoch_samples=0),
         "epoch"),
        (lambda: trigger_locked_rms(np.zeros((2, 100)), np.array([10.0, 95.0]),
                                    epoch_samples=50),
         "beyond the recording"),
    ],
)
def test_metrics_reject_invalid_inputs(call, message: str) -> None:
    with pytest.raises(MetricInputError, match=message):
        call()
