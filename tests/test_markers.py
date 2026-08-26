import numpy as np
import pytest

from mri_correction.markers import (
    MarkerTimingError,
    map_brainvision_position,
    split_volume_blocks,
)


def test_split_volume_blocks_at_non_tr_spacing() -> None:
    samples = np.array([100, 4_600, 9_100, 20_000, 24_500, 29_000])

    blocks = split_volume_blocks(
        samples,
        samples_per_volume=4_500,
        declared_block_starts=np.array([100, 20_000]),
    )

    assert [block.tolist() for block in blocks] == [
        [100, 4_600, 9_100],
        [20_000, 24_500, 29_000],
    ]


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=int),
        np.array([100, 100, 4_600]),
        np.array([4_600, 100]),
        np.array([-1, 4_499]),
    ],
)
def test_split_volume_blocks_rejects_invalid_samples(samples: np.ndarray) -> None:
    with pytest.raises(MarkerTimingError):
        split_volume_blocks(samples, samples_per_volume=4_500)


def test_split_volume_blocks_rejects_undeclared_gap() -> None:
    samples = np.array([100, 4_600, 13_600, 18_100])

    with pytest.raises(MarkerTimingError, match="undeclared"):
        split_volume_blocks(
            samples,
            samples_per_volume=4_500,
            declared_block_starts=np.array([100]),
        )


@pytest.mark.parametrize(
    ("input_position", "expected_position"),
    [
        (1, 1),
        (5, 1),
        (6, 2),
        (4_500, 900),
        (4_501, 901),
    ],
)
def test_map_brainvision_position_matches_analyzer(
    input_position: int,
    expected_position: int,
) -> None:
    assert map_brainvision_position(input_position, factor=5) == expected_position


def test_map_brainvision_position_rejects_invalid_values() -> None:
    with pytest.raises(MarkerTimingError):
        map_brainvision_position(0, factor=5)

    with pytest.raises(MarkerTimingError):
        map_brainvision_position(1, factor=0)
