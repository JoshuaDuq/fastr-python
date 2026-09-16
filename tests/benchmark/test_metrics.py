import numpy as np
import pytest

from benchmark.metrics import (
    MetricError,
    coherent_residual_microvolts,
    harmonic_orders,
    measure_tone_transfer,
)
from benchmark.probes import plan_tones, synthesize_tones

TR = 0.9
RATE = 1_000.0


def test_harmonic_orders_reach_the_requested_ceiling():
    orders = harmonic_orders(
        repetition_time_seconds=TR, max_hz=100.0, mains_hz=60.0, mains_guard_hz=1.0
    )
    assert orders.max() * (1 / TR) <= 100.0
    assert (orders.max() + 1) * (1 / TR) > 100.0


def test_harmonic_orders_drop_the_lines_that_collide_with_mains():
    orders = harmonic_orders(
        repetition_time_seconds=TR, max_hz=100.0, mains_hz=60.0, mains_guard_hz=1.0
    )
    frequencies = orders / TR
    assert np.abs(frequencies - 60.0).min() >= 1.0
    assert 54 not in orders  # 54 / 0.9 is exactly 60 Hz


def test_harmonic_orders_refuse_a_guard_that_excludes_everything():
    with pytest.raises(MetricError, match="excluded every harmonic"):
        harmonic_orders(
            repetition_time_seconds=TR, max_hz=5.0, mains_hz=3.0, mains_guard_hz=100.0
        )


def test_residual_recovers_the_amplitude_of_a_tone_on_a_harmonic():
    volumes = 20
    samples = round(volumes * TR * RATE)
    times = np.arange(samples) / RATE
    # 10 / 0.9 Hz is the tenth volume harmonic, at 3 uV amplitude.
    signal = 3e-6 * np.sin(2 * np.pi * (10 / TR) * times)
    residual = coherent_residual_microvolts(
        np.stack([signal]),
        sampling_rate=RATE,
        repetition_time_seconds=TR,
        orders=np.array([10]),
        block_seconds=volumes * TR,
    )
    assert residual.shape == (1, 1)
    assert residual[0, 0] == pytest.approx(3.0 / np.sqrt(2), rel=1e-3)


def test_residual_reports_one_value_per_block_per_channel():
    samples = round(60 * TR * RATE)
    data = np.zeros((4, samples))
    residual = coherent_residual_microvolts(
        data,
        sampling_rate=RATE,
        repetition_time_seconds=TR,
        orders=np.array([1, 2]),
        block_seconds=18 * TR,
    )
    assert residual.shape == (4, 3)


def _corrections(gain: float):
    """Build a clean and a tone-injected correction that scale tones by gain."""
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(13.0, 23.0),
        mains_hz=60.0,
    )
    samples = round(20 * TR * RATE)
    probe = synthesize_tones(tones, sample_count=samples, sampling_rate=RATE)
    background = np.random.default_rng(0).normal(0, 1e-6, (2, samples))
    return tones, probe, background, background + gain * probe


def test_an_untouched_tone_transfers_at_unity():
    tones, probe, clean, injected = _corrections(1.0)
    results = measure_tone_transfer(
        clean,
        injected,
        probe,
        channel_names=["Fp1", "Cz"],
        tones=tones,
        sampling_rate=RATE,
    )
    assert len(results) == len(tones) * 2
    assert all(r.amplitude_ratio == pytest.approx(1.0, rel=1e-9) for r in results)
    assert all(abs(r.phase_error_degrees) < 1e-6 for r in results)


def test_a_halved_tone_transfers_at_one_half():
    tones, probe, clean, injected = _corrections(0.5)
    results = measure_tone_transfer(
        clean,
        injected,
        probe,
        channel_names=["Fp1", "Cz"],
        tones=tones,
        sampling_rate=RATE,
    )
    assert all(r.amplitude_ratio == pytest.approx(0.5, rel=1e-9) for r in results)


def test_transfer_carries_the_placement_that_was_measured():
    tones, probe, clean, injected = _corrections(1.0)
    results = measure_tone_transfer(
        clean,
        injected,
        probe,
        channel_names=["Fp1", "Cz"],
        tones=tones,
        sampling_rate=RATE,
    )
    assert {r.placement for r in results} == {
        "on_volume_harmonic",
        "between_volume_harmonics",
    }
    assert {r.channel for r in results} == {"Fp1", "Cz"}


def test_transfer_refuses_two_corrections_of_different_shapes():
    tones, probe, clean, injected = _corrections(1.0)
    with pytest.raises(MetricError, match="same shape"):
        measure_tone_transfer(
            clean,
            injected[:, :-1],
            probe,
            channel_names=["Fp1", "Cz"],
            tones=tones,
            sampling_rate=RATE,
        )


def test_transfer_refuses_a_probe_that_does_not_span_the_window():
    tones, probe, clean, injected = _corrections(1.0)
    with pytest.raises(MetricError, match="span the measured window"):
        measure_tone_transfer(
            clean,
            injected,
            probe[:-1],
            channel_names=["Fp1", "Cz"],
            tones=tones,
            sampling_rate=RATE,
        )
