import numpy as np
import pytest

from eegfmri_fastr.simulation import (
    SimulationInputError,
    simulate_gradient_artifact,
    simulate_pulse_artifact,
)

SAMPLING_RATE = 1_000.0
VOLUME_SAMPLES = 900
GROUPS_PER_VOLUME = 18
GROUP_INTERVAL = 47.5
VOLUME_COUNT = 12
FIRST_TRIGGER = 20.0


def make_triggers() -> np.ndarray:
    offsets = np.arange(GROUPS_PER_VOLUME) * GROUP_INTERVAL
    starts = FIRST_TRIGGER + np.arange(VOLUME_COUNT) * VOLUME_SAMPLES
    return (starts[:, np.newaxis] + offsets).reshape(-1)


def sample_count() -> int:
    return VOLUME_COUNT * VOLUME_SAMPLES + int(FIRST_TRIGGER) + 100


def simulate(**overrides) -> np.ndarray:
    arguments = {
        "sample_count": sample_count(),
        "channel_count": 2,
        "sampling_rate": SAMPLING_RATE,
        "readout_seconds": 0.045,
    }
    arguments.update(overrides)
    return simulate_gradient_artifact(make_triggers(), **arguments)


def test_gradient_artifact_repeats_at_every_group_trigger() -> None:
    triggers = make_triggers()

    artifact = simulate(amplitude_drift=0.0)

    # Groups two apart share a sub-sample trigger phase, so they sample the
    # same burst at the same offsets; adjacent groups legitimately do not.
    for index in (10, 11):
        first = artifact[0, int(triggers[index]) : int(triggers[index]) + 45]
        second = artifact[0, int(triggers[index + 2]) : int(triggers[index + 2]) + 45]
        np.testing.assert_allclose(first, second, atol=1e-9)


def test_gradient_artifact_is_silent_in_volume_dead_time() -> None:
    triggers = make_triggers()

    artifact = simulate()

    dead = slice(
        int(triggers[GROUPS_PER_VOLUME - 1] + 0.046 * SAMPLING_RATE) + 3,
        int(triggers[GROUPS_PER_VOLUME]) - 3,
    )
    assert dead.stop - dead.start > 20
    burst = artifact[0, int(triggers[5]) : int(triggers[5]) + 45]
    active = np.sqrt(np.mean(burst**2))
    assert np.max(np.abs(artifact[:, dead])) < 0.01 * active


def test_gradient_artifact_returns_to_baseline_within_each_group() -> None:
    """The induced voltage is a derivative, so each group integrates to zero."""
    triggers = make_triggers()

    artifact = simulate(amplitude_drift=0.0)

    epoch = artifact[0, int(triggers[40]) : int(triggers[40]) + 47]
    assert abs(epoch.mean()) / np.sqrt(np.mean(epoch**2)) < 0.05


def test_gradient_artifact_applies_amplitude_drift_across_the_run() -> None:
    triggers = make_triggers()

    artifact = simulate(amplitude_drift=0.2)

    def volume_rms(first_group: int) -> float:
        span = slice(int(triggers[first_group]), int(triggers[first_group + 17]) + 45)
        return float(np.sqrt(np.mean(artifact[0, span] ** 2)))

    assert volume_rms(len(triggers) - 18) / volume_rms(0) == pytest.approx(
        1.2, rel=0.05
    )


def test_gradient_artifact_is_deterministic_for_one_seed() -> None:
    first = simulate(timing_jitter_seconds=2e-4, seed=7)
    second = simulate(timing_jitter_seconds=2e-4, seed=7)
    third = simulate(timing_jitter_seconds=2e-4, seed=8)

    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, third)


def test_gradient_artifact_scales_each_channel_differently() -> None:
    artifact = simulate(channel_count=4, amplitude_drift=0.0)

    per_channel = np.sqrt(np.mean(artifact**2, axis=1))
    assert len(np.unique(np.round(per_channel, 6))) == 4


def test_pulse_artifact_repeats_at_the_heart_rate() -> None:
    pulse = simulate_pulse_artifact(
        sample_count=sample_count(),
        channel_count=2,
        sampling_rate=SAMPLING_RATE,
        heart_rate_hz=1.2,
    )

    lag = int(SAMPLING_RATE / 1.2)
    repetition = np.corrcoef(pulse[0, :-lag], pulse[0, lag:])[0, 1]
    assert repetition > 0.99


def test_gradient_artifact_baseline_offset_creates_a_net_epoch_offset() -> None:
    triggers = make_triggers()
    epoch = slice(int(triggers[40]), int(triggers[40]) + 47)

    balanced = simulate(amplitude_drift=0.0)[0, epoch]
    offset = simulate(amplitude_drift=0.0, baseline_offset=1.0)[0, epoch]

    assert abs(balanced.mean()) / np.sqrt(np.mean(balanced**2)) < 0.05
    assert abs(offset.mean()) / np.sqrt(np.mean(offset**2)) > 0.2


def test_gradient_artifact_corrupts_the_requested_number_of_groups() -> None:
    triggers = make_triggers()

    artifact = simulate(amplitude_drift=0.0, outlier_count=3, seed=1)

    per_group = np.array(
        [np.sqrt(np.mean(artifact[0, int(t) : int(t) + 45] ** 2)) for t in triggers]
    )
    assert np.count_nonzero(per_group > 1.5 * np.median(per_group)) == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sample_count": 0}, "sample count"),
        ({"channel_count": 0}, "channel count"),
        ({"sampling_rate": 0.0}, "sampling rate"),
        ({"readout_seconds": 0.0}, "readout"),
        ({"readout_seconds": 1.0}, "readout"),
        ({"amplitude_drift": -1.5}, "drift"),
        ({"sample_count": 500}, "beyond the recording"),
    ],
)
def test_simulate_gradient_artifact_rejects_invalid_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SimulationInputError, match=message):
        simulate(**overrides)


def test_gradient_artifact_gives_each_acquisition_slot_its_own_waveform() -> None:
    """Real groups differ by acquisition slot; only the same slot repeats."""
    triggers = make_triggers()

    artifact = simulate_gradient_artifact(
        triggers,
        sample_count=sample_count(),
        channel_count=2,
        sampling_rate=SAMPLING_RATE,
        readout_seconds=0.045,
        amplitude_drift=0.0,
        groups_per_volume=GROUPS_PER_VOLUME,
    )

    def epoch(index: int) -> np.ndarray:
        start = int(triggers[index])
        return artifact[0, start : start + 45]

    def correlation(first: int, second: int) -> float:
        a, b = epoch(first) - epoch(first).mean(), epoch(second) - epoch(second).mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    assert correlation(20, 20 + GROUPS_PER_VOLUME) > 0.999
    assert abs(correlation(20, 21)) < 0.5


def test_gradient_artifact_shares_one_waveform_without_slice_positions() -> None:
    triggers = make_triggers()

    artifact = simulate_gradient_artifact(
        triggers,
        sample_count=sample_count(),
        channel_count=2,
        sampling_rate=SAMPLING_RATE,
        readout_seconds=0.045,
        amplitude_drift=0.0,
    )

    first = artifact[0, int(triggers[20]) : int(triggers[20]) + 45]
    second = artifact[0, int(triggers[22]) : int(triggers[22]) + 45]
    np.testing.assert_allclose(first, second, atol=1e-9)
