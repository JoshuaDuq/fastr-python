import numpy as np
import pytest

from mri_correction.diagnostics import (
    DiagnosticInputError,
    estimate_slice_period_candidates,
)

SAMPLE_COUNT = 4_000
TRUE_PERIOD_SAMPLES = 50


def make_periodic_gradient(period_samples: float) -> np.ndarray:
    """Two channels carrying a sharp waveform that repeats at a known period."""
    samples = np.arange(SAMPLE_COUNT, dtype=np.float64)
    harmonics = np.arange(1, 9)[:, np.newaxis]
    shape = np.sum(
        np.sin(2 * np.pi * harmonics * samples / period_samples + harmonics),
        axis=0,
    )
    return np.stack([shape, -2.0 * shape])


def test_estimate_slice_period_candidates_recovers_an_injected_period() -> None:
    data = make_periodic_gradient(TRUE_PERIOD_SAMPLES)

    candidates = estimate_slice_period_candidates(
        data,
        minimum_period=20,
        maximum_period=120,
    )

    assert candidates[0].period_samples == TRUE_PERIOD_SAMPLES
    assert candidates[0].autocorrelation > 0.99


def test_estimate_slice_period_candidates_sees_past_low_frequency_drift() -> None:
    samples = np.arange(SAMPLE_COUNT, dtype=np.float64)
    drift = 50.0 * np.sin(2 * np.pi * samples / 2_000.0)
    data = make_periodic_gradient(TRUE_PERIOD_SAMPLES) + drift

    candidates = estimate_slice_period_candidates(
        data,
        minimum_period=20,
        maximum_period=120,
    )

    assert candidates[0].period_samples == TRUE_PERIOD_SAMPLES


def test_estimate_slice_period_candidates_omits_implied_multiples() -> None:
    data = make_periodic_gradient(TRUE_PERIOD_SAMPLES)

    candidates = estimate_slice_period_candidates(
        data,
        minimum_period=20,
        maximum_period=200,
    )

    reported = {candidate.period_samples for candidate in candidates}
    assert TRUE_PERIOD_SAMPLES in reported
    assert reported.isdisjoint({100, 150, 200})


def test_estimate_slice_period_candidates_ranks_by_descending_evidence() -> None:
    data = np.random.default_rng(1).normal(size=(2, SAMPLE_COUNT))

    candidates = estimate_slice_period_candidates(
        data,
        minimum_period=20,
        maximum_period=200,
        candidate_count=4,
    )

    scores = [candidate.autocorrelation for candidate in candidates]
    assert len(candidates) == 4
    assert scores == sorted(scores, reverse=True)
    assert len({candidate.period_samples for candidate in candidates}) == 4


def test_estimate_slice_period_candidates_reports_weak_aperiodic_evidence() -> None:
    data = np.random.default_rng(0).normal(size=(2, SAMPLE_COUNT))

    candidates = estimate_slice_period_candidates(
        data,
        minimum_period=20,
        maximum_period=120,
    )

    assert max(candidate.autocorrelation for candidate in candidates) < 0.2


@pytest.mark.parametrize(
    ("data", "parameters", "message"),
    [
        (np.zeros(SAMPLE_COUNT), {}, "channels, samples"),
        (np.zeros((2, SAMPLE_COUNT)) + np.nan, {}, "finite"),
        (np.zeros((2, SAMPLE_COUNT)), {"minimum_period": 0}, "period range"),
        (np.zeros((2, SAMPLE_COUNT)), {"maximum_period": 20}, "period range"),
        (np.zeros((2, 30)), {}, "period range"),
        (np.zeros((2, SAMPLE_COUNT)), {"candidate_count": 0}, "candidate count"),
    ],
)
def test_estimate_slice_period_candidates_rejects_invalid_inputs(
    data: np.ndarray,
    parameters: dict[str, int],
    message: str,
) -> None:
    arguments: dict[str, int] = {"minimum_period": 20, "maximum_period": 120}
    arguments.update(parameters)

    with pytest.raises(DiagnosticInputError, match=message):
        estimate_slice_period_candidates(data, **arguments)
