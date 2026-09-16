import numpy as np
import pytest

from benchmark.harmonise import (
    CommonWindow,
    HarmoniseError,
    common_window,
    crop,
    guard_volumes,
    to_output_rate,
)

TR = 0.9
RATE = 1_000.0


def test_window_takes_the_shortest_arm_rounded_down_to_an_even_volume_count():
    window = common_window(
        {"fastr_python": 542, "matlab_fmrib": 545, "facetpy": 543},
        repetition_time_seconds=TR,
        sampling_rate=RATE,
        guard=0,
    )
    assert window.volume_count == 542
    assert window.samples_per_volume == 900
    assert window.sample_count == 542 * 900


def test_window_drops_an_odd_trailing_volume():
    window = common_window(
        {"a": 101, "b": 200}, repetition_time_seconds=TR, sampling_rate=RATE, guard=0
    )
    assert window.volume_count == 100


def test_window_rejects_a_repetition_that_is_not_a_whole_number_of_samples():
    with pytest.raises(HarmoniseError, match="not whole"):
        common_window(
            {"a": 10}, repetition_time_seconds=0.9, sampling_rate=1_111.0, guard=0
        )


def test_window_rejects_an_arm_that_produced_nothing():
    with pytest.raises(HarmoniseError, match="no arm keeps a whole volume"):
        common_window(
            {"a": 100, "b": 0}, repetition_time_seconds=TR, sampling_rate=RATE, guard=0
        )


def test_window_rejects_a_shared_span_shorter_than_two_volumes():
    with pytest.raises(HarmoniseError, match="at least two volumes"):
        common_window({"a": 1}, repetition_time_seconds=TR, sampling_rate=RATE, guard=0)


def test_window_needs_at_least_one_arm():
    with pytest.raises(HarmoniseError, match="at least one arm"):
        common_window({}, repetition_time_seconds=TR, sampling_rate=RATE, guard=0)


def test_crop_starts_at_each_arms_own_first_volume():
    window = CommonWindow(
        volume_count=2, guard_volumes=0, samples_per_volume=900, sampling_rate=RATE
    )
    data = np.arange(2 * 5_000, dtype=float).reshape(2, 5_000)
    cropped = crop(data, first_volume_sample=137, window=window)
    assert cropped.shape == (2, 1_800)
    assert cropped[0, 0] == data[0, 137]


def test_crop_refuses_an_arm_too_short_for_the_shared_span():
    window = CommonWindow(
        volume_count=4, guard_volumes=0, samples_per_volume=900, sampling_rate=RATE
    )
    data = np.zeros((2, 2_000))
    with pytest.raises(HarmoniseError, match="but this arm wrote"):
        crop(data, first_volume_sample=0, window=window)


def test_crop_rejects_a_negative_offset():
    window = CommonWindow(
        volume_count=2, guard_volumes=0, samples_per_volume=900, sampling_rate=RATE
    )
    with pytest.raises(HarmoniseError, match="before the recording"):
        crop(np.zeros((1, 5_000)), first_volume_sample=-1, window=window)


def test_decimation_keeps_a_tone_and_removes_what_is_above_the_new_nyquist():
    rate, out = 5_000.0, 1_000.0
    samples = round(60 * 0.9 * rate)
    times = np.arange(samples) / rate
    keep = np.sin(2 * np.pi * 23.0 * times)
    alias = np.sin(2 * np.pi * 900.0 * times)
    decimated = to_output_rate(
        np.stack([keep + alias]),
        sampling_rate=rate,
        output_sampling_rate=out,
        lowpass_hz=100.0,
    )
    expected = np.sin(2 * np.pi * 23.0 * (np.arange(decimated.shape[1]) / out))
    guard = round(0.9 * out)
    assert decimated.shape[1] == samples // 5
    assert np.abs(decimated[0] - expected)[guard:-guard].max() < 0.01


def test_decimation_refuses_a_non_integer_ratio():
    with pytest.raises(HarmoniseError, match="does not decimate"):
        to_output_rate(
            np.zeros((1, 100)),
            sampling_rate=5_000.0,
            output_sampling_rate=999.0,
            lowpass_hz=100.0,
        )


def test_the_filter_edge_is_what_the_guard_exists_to_exclude():
    rate, out = 5_000.0, 1_000.0
    samples = round(60 * 0.9 * rate)
    times = np.arange(samples) / rate
    signal = np.sin(2 * np.pi * 23.0 * times) + np.sin(2 * np.pi * 900.0 * times)
    decimated = to_output_rate(
        np.stack([signal]),
        sampling_rate=rate,
        output_sampling_rate=out,
        lowpass_hz=100.0,
    )
    expected = np.sin(2 * np.pi * 23.0 * (np.arange(decimated.shape[1]) / out))
    error = np.abs(decimated[0] - expected)
    guard = round(0.9 * out)
    assert error[-1] > 0.1
    assert error[guard:-guard].max() < 0.01


def test_the_guard_is_taken_from_the_filter_that_is_actually_designed():
    assert (
        guard_volumes(
            sampling_rate=5_000.0,
            output_sampling_rate=1_000.0,
            lowpass_hz=100.0,
            repetition_time_seconds=TR,
        )
        == 1
    )


def test_an_unfiltered_arm_needs_no_guard():
    assert (
        guard_volumes(
            sampling_rate=5_000.0,
            output_sampling_rate=5_000.0,
            lowpass_hz=0.0,
            repetition_time_seconds=TR,
        )
        == 0
    )


def test_the_guard_shortens_the_measured_span_at_both_ends():
    window = common_window(
        {"a": 100}, repetition_time_seconds=TR, sampling_rate=RATE, guard=1
    )
    assert window.volume_count == 98
    assert window.guard_samples == 900


def test_crop_skips_the_guard_at_the_leading_edge():
    window = CommonWindow(
        volume_count=2, guard_volumes=1, samples_per_volume=900, sampling_rate=RATE
    )
    data = np.arange(2 * 5_000, dtype=float).reshape(2, 5_000)
    cropped = crop(data, first_volume_sample=100, window=window)
    assert cropped[0, 0] == data[0, 1_000]
