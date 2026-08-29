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

from mri_correction.fastr_processing import obs_trigger_subset, residual_obs
from mri_correction.fastr_types import FastrInputError


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
