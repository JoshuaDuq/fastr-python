import numpy as np
import pytest

from benchmark.window import CommonWindow, WindowError, common_window, crop

TR = 0.9
RATE = 1_000.0


def test_window_takes_the_shortest_arm_rounded_down_to_an_even_volume_count():
    window = common_window(
        {"fastr_python": 542, "matlab_fmrib": 545, "facetpy": 543},
        repetition_time_seconds=TR,
        sampling_rate=RATE,
    )
    assert window.volume_count == 542
    assert window.samples_per_volume == 900
    assert window.sample_count == 542 * 900


def test_window_drops_an_odd_trailing_volume():
    window = common_window(
        {"a": 101, "b": 200}, repetition_time_seconds=TR, sampling_rate=RATE
    )
    assert window.volume_count == 100


def test_window_rejects_a_repetition_that_is_not_a_whole_number_of_samples():
    with pytest.raises(WindowError, match="not whole"):
        common_window({"a": 10}, repetition_time_seconds=0.9, sampling_rate=1_111.0)


def test_window_rejects_an_arm_that_produced_nothing():
    with pytest.raises(WindowError, match="no whole volume"):
        common_window(
            {"a": 100, "b": 0}, repetition_time_seconds=TR, sampling_rate=RATE
        )


def test_window_rejects_a_shared_span_shorter_than_two_volumes():
    with pytest.raises(WindowError, match="at least two volumes"):
        common_window({"a": 1}, repetition_time_seconds=TR, sampling_rate=RATE)


def test_window_needs_at_least_one_arm():
    with pytest.raises(WindowError, match="at least one arm"):
        common_window({}, repetition_time_seconds=TR, sampling_rate=RATE)


def test_crop_starts_at_each_arms_own_first_volume():
    window = CommonWindow(volume_count=2, samples_per_volume=900, sampling_rate=RATE)
    data = np.arange(2 * 5_000, dtype=float).reshape(2, 5_000)
    cropped = crop(data, first_volume_sample=137, window=window)
    assert cropped.shape == (2, 1_800)
    assert cropped[0, 0] == data[0, 137]


def test_crop_refuses_an_arm_too_short_for_the_shared_span():
    window = CommonWindow(volume_count=4, samples_per_volume=900, sampling_rate=RATE)
    data = np.zeros((2, 2_000))
    with pytest.raises(WindowError, match="but this arm wrote"):
        crop(data, first_volume_sample=0, window=window)


def test_crop_rejects_a_negative_offset():
    window = CommonWindow(volume_count=2, samples_per_volume=900, sampling_rate=RATE)
    with pytest.raises(WindowError, match="before the recording"):
        crop(np.zeros((1, 5_000)), first_volume_sample=-1, window=window)
