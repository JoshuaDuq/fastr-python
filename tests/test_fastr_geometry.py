import numpy as np
import pytest

from fastr_python.fastr import FastrInputError, prepare_fastr_geometry


def make_geometry(fraction: object):
    return prepare_fastr_geometry(
        np.arange(40, dtype=np.float64) * 100.0 + 200.0,
        sample_count=4_400,
        interpolation_factor=10,
        neighbor_count=10,
        search_radius_samples=3,
        pre_trigger_fraction=fraction,
    )


@pytest.mark.parametrize(
    ("fraction", "samples_before", "samples_after"),
    [
        (0.0, 10, 1_000),
        (0.03, 40, 970),
        (1.0, 1_010, 0),
    ],
)
def test_geometry_uses_configured_trigger_fraction(
    fraction: float,
    samples_before: int,
    samples_after: int,
) -> None:
    geometry = make_geometry(fraction)

    assert geometry.pre_trigger_fraction == fraction
    assert geometry.epoch.samples_before == samples_before
    assert geometry.epoch.samples_after == samples_after


@pytest.mark.parametrize("fraction", [-0.01, 1.01, np.inf, np.nan, True, "0.03"])
def test_geometry_rejects_invalid_trigger_fraction(fraction: object) -> None:
    with pytest.raises(FastrInputError, match="pre-trigger fraction"):
        make_geometry(fraction)


def test_geometry_keeps_the_existing_trigger_fraction_default() -> None:
    geometry = prepare_fastr_geometry(
        np.arange(40, dtype=np.float64) * 100.0 + 200.0,
        sample_count=4_400,
        interpolation_factor=10,
        neighbor_count=10,
        search_radius_samples=3,
    )

    assert geometry.pre_trigger_fraction == 0.03
    assert geometry.epoch.samples_before == 40
    assert geometry.epoch.samples_after == 970
