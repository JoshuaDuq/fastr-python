import numpy as np
import pytest

from fastr_python.fastr import FastrInputError, repair_volume_starts


def test_repair_volume_starts_fills_unique_interior_gaps() -> None:
    starts = np.array([100, 200, 400, 500], dtype=np.int64)

    repaired = repair_volume_starts(
        starts,
        samples_per_volume=100,
        expected_volume_count=5,
    )

    np.testing.assert_array_equal(repaired, [100, 200, 300, 400, 500])


def test_repair_volume_starts_accepts_one_sample_clock_ticks() -> None:
    starts = np.array([100, 201, 401, 501], dtype=np.int64)

    repaired = repair_volume_starts(
        starts,
        samples_per_volume=100,
        expected_volume_count=5,
    )

    np.testing.assert_array_equal(repaired, [100, 201, 301, 401, 501])


@pytest.mark.parametrize(
    ("starts", "count", "message"),
    [
        ([100, 250, 350], 4, "integer multiple"),
        ([100, 200, 400], 5, "expected volume count"),
        ([100, 200, 300], 4, "boundary markers"),
    ],
)
def test_repair_volume_starts_rejects_ambiguous_repairs(
    starts: list[int],
    count: int,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        repair_volume_starts(
            np.asarray(starts),
            samples_per_volume=100,
            expected_volume_count=count,
        )


@pytest.mark.parametrize(
    ("samples_per_volume", "expected_volume_count", "message"),
    [
        (True, 3, "samples per volume"),
        (0, 3, "samples per volume"),
        (100, True, "expected volume count"),
        (100, 0, "expected volume count"),
    ],
)
def test_repair_volume_starts_rejects_invalid_counts(
    samples_per_volume: object,
    expected_volume_count: object,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        repair_volume_starts(
            np.array([100, 200, 300]),
            samples_per_volume=samples_per_volume,
            expected_volume_count=expected_volume_count,
        )
