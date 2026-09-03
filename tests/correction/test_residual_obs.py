"""The optimal basis set is FASTR's third stage, wired in as an optional step.

Two properties of the published stage drive these tests. The basis is estimated
from a 70 Hz high-passed copy of the residual, following `fmrib_fastr.m`, so the
stage only reaches artifact structure above that corner. And it is applied at
volume granularity, because the residual it removes is the volume-to-volume
variability of the gradient artifact, which an acquisition-group epoch is too
short to represent.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from fastr_python.correction.processing import obs_trigger_subset, residual_obs
from fastr_python.correction.types import FastrInputError


def test_obs_trigger_subset_drops_triggers_whose_epochs_would_overrun() -> None:
    period = 100
    triggers = np.arange(0, 2000, period)

    kept = obs_trigger_subset(triggers, sample_count=2000, interpolation_factor=10)

    # the leading trigger has no room for the pre-trigger samples
    assert kept[0] > triggers[0]
    assert kept.size < triggers.size

    truncated = obs_trigger_subset(
        triggers,
        sample_count=1910,
        interpolation_factor=10,
    )
    assert truncated[-1] < kept[-1]


def test_obs_trigger_subset_returns_only_usable_triggers() -> None:
    samples = 2000
    kept = obs_trigger_subset(
        np.arange(0, samples, 100),
        sample_count=samples,
        interpolation_factor=10,
    )

    residual_obs(
        np.zeros((1, samples), dtype=np.float64),
        kept,
        sampling_rate=1000.0,
        excluded_channels=[],
        rank=2,
        interpolation_factor=10,
    )


def test_obs_trigger_subset_raises_when_nothing_fits() -> None:
    with pytest.raises(FastrInputError):
        obs_trigger_subset(
            np.array([0, 100]),
            sample_count=120,
            interpolation_factor=10,
        )


def test_residual_obs_removes_repeating_residual_and_keeps_wanted_signal() -> None:
    rate = 5000.0
    period = 500
    count = 80
    samples = period * count
    rng = np.random.default_rng(11)

    # a gradient-like residual: energy well above the stage's 70 Hz corner,
    # repeating each epoch with a varying gain
    within = np.arange(period) / rate
    shape = np.sin(2.0 * np.pi * 260.0 * within) * np.exp(-within * 120.0)
    gains = rng.normal(1.0, 0.4, size=count)
    residual = np.concatenate([gain * shape for gain in gains])

    times = np.arange(samples) / rate
    wanted = 3.0 * np.sin(2.0 * np.pi * 7.3 * times + 0.4)
    recording = (residual + wanted)[np.newaxis, :]

    triggers = obs_trigger_subset(
        np.arange(0, samples, period),
        sample_count=samples,
        interpolation_factor=10,
    )
    corrected = residual_obs(
        recording,
        triggers,
        sampling_rate=rate,
        excluded_channels=[],
        rank=2,
        interpolation_factor=10,
    )

    inner = slice(period * 4, samples - period * 4)
    before = float(np.std(recording[0, inner] - wanted[inner]))
    after = float(np.std(corrected[0, inner] - wanted[inner]))
    assert after < 0.5 * before

    retained = float(
        np.vdot(wanted[inner], corrected[0, inner]).real
        / np.vdot(wanted[inner], wanted[inner]).real
    )
    assert retained == pytest.approx(1.0, abs=0.05)


def test_residual_obs_leaves_excluded_channels_untouched() -> None:
    samples = 6000
    rng = np.random.default_rng(3)
    recording = rng.normal(size=(2, samples))
    triggers = obs_trigger_subset(
        np.arange(0, samples, 100),
        sample_count=samples,
        interpolation_factor=10,
    )

    corrected = residual_obs(
        recording,
        triggers,
        sampling_rate=1000.0,
        excluded_channels=[1],
        rank=2,
        interpolation_factor=10,
    )

    assert np.array_equal(corrected[1], recording[1])
    assert not np.array_equal(corrected[0], recording[0])


def make_changing_residual() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A residual whose shape changes halfway through the recording.

    One basis estimated over the whole recording has to compromise between the
    two shapes. `fmrib_fastr.m` re-estimates the basis every section, which is
    what lets a low-rank basis follow a residual that is not stationary.
    """
    rate = 5000.0
    period = 500
    count = 80
    samples = period * count
    rng = np.random.default_rng(5)

    within = np.arange(period) / rate
    early = np.sin(2.0 * np.pi * 260.0 * within) * np.exp(-within * 120.0)
    late = np.sin(2.0 * np.pi * 410.0 * within + 1.1) * np.exp(-within * 60.0)

    gains = rng.normal(1.0, 0.3, size=count)
    residual = np.concatenate(
        [
            gain * (early if index < count // 2 else late)
            for index, gain in enumerate(gains)
        ]
    )
    times = np.arange(samples) / rate
    wanted = 3.0 * np.sin(2.0 * np.pi * 7.3 * times + 0.4)
    recording = (residual + wanted)[np.newaxis, :]
    return recording, wanted, np.arange(0, samples, period), rate


def test_a_sectioned_basis_follows_a_residual_that_changes_shape() -> None:
    recording, wanted, positions, rate = make_changing_residual()
    triggers = obs_trigger_subset(
        positions,
        sample_count=recording.shape[1],
        interpolation_factor=10,
    )

    whole = residual_obs(
        recording,
        triggers,
        sampling_rate=rate,
        excluded_channels=[],
        rank=1,
        interpolation_factor=10,
    )
    sectioned = residual_obs(
        recording,
        triggers,
        sampling_rate=rate,
        excluded_channels=[],
        rank=1,
        interpolation_factor=10,
        section_seconds=4.0,
    )

    inner = slice(500 * 4, recording.shape[1] - 500 * 4)
    whole_error = float(np.std(whole[0, inner] - wanted[inner]))
    sectioned_error = float(np.std(sectioned[0, inner] - wanted[inner]))
    assert sectioned_error < 0.6 * whole_error


def test_a_section_too_short_for_the_basis_rank_is_rejected() -> None:
    recording, _, positions, rate = make_changing_residual()
    triggers = obs_trigger_subset(
        positions,
        sample_count=recording.shape[1],
        interpolation_factor=10,
    )

    with pytest.raises(FastrInputError, match="section"):
        residual_obs(
            recording,
            triggers,
            sampling_rate=rate,
            excluded_channels=[],
            rank=4,
            interpolation_factor=10,
            section_seconds=0.3,
        )


def test_a_short_trailing_stretch_is_not_given_its_own_basis() -> None:
    """A section with barely more epochs than the rank would gut that stretch.

    A rank-4 basis spans four of the five dimensions a five-epoch section has,
    so the projection takes the signal with the residual. The remainder has to
    join the section before it. The probe is broadband above the stage's 70 Hz
    corner and is not epoch-locked, so nothing but over-fitting can remove it.
    """
    rate = 5000.0
    period = 500
    count = 85
    samples = period * count

    within = np.arange(period) / rate
    shape = np.sin(2.0 * np.pi * 260.0 * within) * np.exp(-within * 120.0)
    gains = np.random.default_rng(7).normal(1.0, 0.3, size=count)
    residual = np.concatenate([gain * shape for gain in gains])
    noise = filtfilt(
        *butter(4, (75.0, 200.0), btype="band", fs=rate),
        np.random.default_rng(1).standard_normal(samples),
    )
    wanted = 3.0 * noise / np.std(noise)
    recording = (residual + wanted)[np.newaxis, :]

    triggers = obs_trigger_subset(
        np.arange(0, samples, period),
        sample_count=samples,
        interpolation_factor=10,
    )

    def kept(section_seconds: float | None) -> float:
        corrected = residual_obs(
            recording,
            triggers,
            sampling_rate=rate,
            excluded_channels=[],
            rank=4,
            interpolation_factor=10,
            section_seconds=section_seconds,
        )
        # the final five epochs, which a 7.9 s section leaves as a remainder
        tail = slice(period * 79 + 250, period * 83)
        return float(
            np.vdot(wanted[tail], corrected[0, tail]).real
            / np.vdot(wanted[tail], wanted[tail]).real
        )

    assert kept(7.9) > 0.8 * kept(None)


def test_a_batch_of_only_excluded_channels_is_returned_untouched() -> None:
    """Channels arrive in batches, so a whole batch can be non-EEG."""
    samples = 6000
    recording = np.random.default_rng(3).normal(size=(2, samples))
    triggers = obs_trigger_subset(
        np.arange(0, samples, 100),
        sample_count=samples,
        interpolation_factor=10,
    )

    corrected = residual_obs(
        recording,
        triggers,
        sampling_rate=1000.0,
        excluded_channels=[0, 1],
        rank=2,
        interpolation_factor=10,
    )

    np.testing.assert_array_equal(corrected, recording)
