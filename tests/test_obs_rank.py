import numpy as np
import pytest

from eegfmri_fastr.fastr import (
    FastrInputError,
    fit_residual_obs,
    select_obs_rank,
)


def test_select_obs_rank_matches_fmrib_three_criterion_rule() -> None:
    explained = np.array([45.0, 25.0, 12.0, 7.0, 4.0, 3.0, 2.0, 1.0, 0.5])

    assert select_obs_rank(explained) == 3


@pytest.mark.parametrize(
    "explained",
    [
        np.array([60.0, 40.0]),
        np.array([60.0, 20.0, 10.0, 6.0, 4.0]),
        np.array([60.0, 20.0, 10.0, 6.0, np.nan]),
        np.ones((2, 3)),
    ],
)
def test_select_obs_rank_rejects_spectra_without_all_criteria(
    explained: np.ndarray,
) -> None:
    with pytest.raises(FastrInputError, match="automatic OBS rank"):
        select_obs_rank(explained)


def make_residual() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    samples = np.arange(6_000, dtype=np.float64)
    residual = np.vstack(
        [
            np.sin(2.0 * np.pi * samples / 50.0) + 0.05 * rng.standard_normal(6_000),
            np.cos(2.0 * np.pi * samples / 50.0) + 0.05 * rng.standard_normal(6_000),
        ]
    )
    triggers = np.arange(20, 100, dtype=np.int64) * 50
    return residual, triggers


def test_fit_residual_obs_reports_fixed_rank_by_channel_and_section() -> None:
    residual, triggers = make_residual()

    result = fit_residual_obs(
        residual,
        triggers,
        sampling_rate=500.0,
        excluded_channels=(),
        rank=2,
        interpolation_factor=1,
        section_seconds=4.0,
    )

    assert result.data.shape == residual.shape
    assert result.selected_ranks.shape == (2, 2)
    np.testing.assert_array_equal(result.selected_ranks, 2)


def test_fit_residual_obs_selects_automatic_rank() -> None:
    rng = np.random.default_rng(7)
    residual = rng.standard_normal((1, 6_000))
    triggers = np.arange(20, 100, dtype=np.int64) * 50

    result = fit_residual_obs(
        residual,
        triggers,
        sampling_rate=500.0,
        excluded_channels=(),
        rank="auto",
        interpolation_factor=1,
    )

    assert result.selected_ranks.shape == (1, 1)
    assert result.selected_ranks[0, 0] >= 1


def test_fit_residual_obs_records_excluded_channels_as_rank_zero() -> None:
    residual, triggers = make_residual()

    result = fit_residual_obs(
        residual,
        triggers,
        sampling_rate=500.0,
        excluded_channels=(1,),
        rank=2,
        interpolation_factor=1,
        section_seconds=4.0,
    )

    np.testing.assert_array_equal(result.selected_ranks[0], [2, 2])
    np.testing.assert_array_equal(result.selected_ranks[1], [0, 0])
    np.testing.assert_array_equal(result.data[1], residual[1])


@pytest.mark.parametrize("rank", [0, True, "AUTO", "three"])
def test_fit_residual_obs_rejects_invalid_rank(rank: object) -> None:
    residual, triggers = make_residual()

    with pytest.raises(FastrInputError, match="basis rank"):
        fit_residual_obs(
            residual,
            triggers,
            sampling_rate=500.0,
            excluded_channels=(),
            rank=rank,
            interpolation_factor=1,
        )
