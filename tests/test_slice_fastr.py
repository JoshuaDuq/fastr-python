import math

import numpy as np
import pytest

from mri_correction.fastr import (
    FastrInputError,
    FastrProvenance,
    FmriAcquisitionTiming,
    acquisition_group_fastr,
    residual_obs,
    slice_fastr,
)
from mri_correction.simulation import simulate_gradient_artifact

SAMPLING_RATE = 1_000.0
GROUP_INTERVAL_SAMPLES = 47.5
GROUP_COUNT = 80
FIRST_TRIGGER_SAMPLE = 20.0
GROUPS_PER_VOLUME = 18
VOLUME_SAMPLES = 900
VOLUME_COUNT = 4
SLOT_OFFSETS_SECONDS = tuple(np.arange(GROUPS_PER_VOLUME) * 0.0475)


def make_group_triggers() -> np.ndarray:
    """Uniform half-sample group triggers, as multiband timing produces."""
    return FIRST_TRIGGER_SAMPLE + np.arange(GROUP_COUNT) * GROUP_INTERVAL_SAMPLES


def make_multiband_triggers() -> np.ndarray:
    """Group triggers of an acquisition-slot volume series with dead time."""
    offsets = np.arange(GROUPS_PER_VOLUME) * GROUP_INTERVAL_SAMPLES
    volume_starts = FIRST_TRIGGER_SAMPLE + np.arange(VOLUME_COUNT) * VOLUME_SAMPLES
    return (volume_starts[:, np.newaxis] + offsets).reshape(-1)


def make_acquisition_slot_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.9,
        slice_timing_seconds=SLOT_OFFSETS_SECONDS,
        multiband_acceleration_factor=1,
    )


def make_multiband_volume_starts() -> np.ndarray:
    return np.int64(FIRST_TRIGGER_SAMPLE) + np.arange(
        VOLUME_COUNT,
        dtype=np.int64,
    ) * VOLUME_SAMPLES


def constant_amplitudes(*channel_scales: float) -> np.ndarray:
    """Per-group artifact amplitudes that never drift."""
    return np.outer(channel_scales, np.ones(GROUP_COUNT))


def make_gradient_artifact(
    group_amplitudes: np.ndarray,
    *,
    triggers: np.ndarray | None = None,
    group_delays: np.ndarray | None = None,
) -> np.ndarray:
    """Band-limited artifact burst repeating at each group trigger.

    `group_amplitudes` is (channels, groups); `group_delays` offsets individual
    groups from their declared trigger, as scanner timing jitter does. Each
    burst is confined to one group interval, so volume dead time stays silent.
    """
    group_triggers = make_group_triggers() if triggers is None else triggers
    samples = np.arange(int(group_triggers[-1]) + 60, dtype=np.float64)
    owners = np.searchsorted(group_triggers, samples, side="right") - 1
    owners = np.clip(owners, 0, None)
    delays = (
        np.zeros(group_triggers.size) if group_delays is None else group_delays
    )
    phases = (
        samples - group_triggers[owners] - delays[owners]
    ) / GROUP_INTERVAL_SAMPLES
    harmonics = np.arange(1, 9)[:, np.newaxis]
    shape = np.sum(
        np.sin(2 * np.pi * harmonics * phases + harmonics) / harmonics,
        axis=0,
    ) * np.sin(np.pi * np.clip(phases, 0.0, 1.0)) ** 2
    return np.where(
        (phases >= 0.0) & (phases < 1.0),
        group_amplitudes[:, owners] * shape,
        0.0,
    )


def fitted_span(triggers: np.ndarray) -> slice:
    """Samples the correction is expected to touch."""
    return slice(int(triggers[0]) + 1, int(triggers[-1]))


def volume_dead_time(
    triggers: np.ndarray,
    provenance: FastrProvenance,
) -> slice:
    """The gap between volumes that no searched artifact epoch can reach."""
    factor = provenance.interpolation_factor
    reach_before = (
        provenance.samples_before_trigger + provenance.search_radius
    ) / factor
    reach_after = (
        provenance.samples_after_trigger + provenance.search_radius
    ) / factor
    return slice(
        math.ceil(triggers[GROUPS_PER_VOLUME - 1] + reach_after),
        math.ceil(triggers[GROUPS_PER_VOLUME] - reach_before),
    )


def tone_amplitude(signal: np.ndarray, period: float, span: slice) -> float:
    """Least-squares amplitude of one exact tone over the fitted samples."""
    times = np.arange(signal.size, dtype=np.float64)[span]
    basis = np.stack(
        [np.sin(2 * np.pi * times / period), np.cos(2 * np.pi * times / period)],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, signal[span], rcond=None)
    return float(np.hypot(*coefficients))


def test_slice_fastr_suppresses_a_repeating_group_artifact() -> None:
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0))
    triggers = make_group_triggers()

    correction = slice_fastr(data, triggers)

    fitted = fitted_span(triggers)
    residual_rms = np.sqrt(np.mean(correction.data[:, fitted] ** 2))
    artifact_rms = np.sqrt(np.mean(data[:, fitted] ** 2))
    assert residual_rms / artifact_rms < 0.01


def test_slice_fastr_builds_templates_from_opposite_parity_neighbors() -> None:
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0))

    neighbors = slice_fastr(data, make_group_triggers()).provenance.neighbor_indices

    targets = np.arange(GROUP_COUNT)[:, np.newaxis]
    assert neighbors.shape == (GROUP_COUNT, 30)
    assert np.all(neighbors % 2 != targets % 2)
    assert np.all(np.diff(neighbors, axis=1) == 2)
    assert neighbors.min() == 0
    assert neighbors.max() == GROUP_COUNT - 1


def test_slice_fastr_centers_neighbor_windows_and_shifts_them_at_edges() -> None:
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0))

    neighbors = slice_fastr(data, make_group_triggers()).provenance.neighbor_indices

    np.testing.assert_array_equal(neighbors[40], np.arange(11, 71, 2))
    np.testing.assert_array_equal(neighbors[0], np.arange(1, 61, 2))
    np.testing.assert_array_equal(neighbors[GROUP_COUNT - 1], np.arange(20, 80, 2))


def test_slice_fastr_fits_the_sub_sample_shift_of_a_delayed_group() -> None:
    delays = np.zeros(GROUP_COUNT)
    delays[40] = 1.3
    data = make_gradient_artifact(
        constant_amplitudes(1.0, -2.0),
        group_delays=delays,
    )

    shifts = slice_fastr(data, make_group_triggers()).provenance.shifts

    assert shifts[40] == 13
    assert np.all(shifts[np.arange(GROUP_COUNT) != 40] == 0)


def test_slice_fastr_uses_aligned_neighbors_for_channel_fitting() -> None:
    """Template amplitudes must be fitted against the same aligned epochs."""
    group_number = np.arange(GROUP_COUNT, dtype=np.float64)
    delays = 1.0 * np.sin(2.0 * np.pi * group_number / 17.0)
    data = make_gradient_artifact(
        constant_amplitudes(1.0, -2.0),
        group_delays=delays,
    )

    correction = slice_fastr(data, make_group_triggers())

    fitted = fitted_span(make_group_triggers())
    residual_rms = np.sqrt(np.mean(correction.data[:, fitted] ** 2))
    artifact_rms = np.sqrt(np.mean(data[:, fitted] ** 2))
    assert residual_rms / artifact_rms < 0.02


def test_slice_fastr_fits_template_amplitudes_per_channel() -> None:
    amplitudes = constant_amplitudes(1.0, -2.0)
    amplitudes[0, 40] *= 1.5
    data = make_gradient_artifact(amplitudes)

    fitted = slice_fastr(data, make_group_triggers()).provenance.amplitudes

    assert fitted[0, 40] == pytest.approx(1.5, abs=0.01)
    assert fitted[1, 40] == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(fitted[:, 42::2], 1.0, atol=0.01)


def test_slice_fastr_transfers_most_of_an_exact_volume_harmonic() -> None:
    """Pins how much exact-harmonic transfer an unbalanced artifact epoch costs.

    This fixture's burst carries a large net offset per epoch, |mean|/RMS near
    0.5, and at that ratio the per-channel amplitude fit trades away roughly a
    fifth of an exact 1/TR harmonic. Real sub-0001 epochs sit near 0.05, where the
    same code retains essentially all of it, so this bound characterises the
    mechanism rather than the acquisition. Measured in
    docs/results/2026-08-26-slice-fastr-transfer.md.
    """
    triggers = make_multiband_triggers()
    artifact = make_gradient_artifact(
        np.outer([1.0, -2.0], np.ones(triggers.size)),
        triggers=triggers,
    )
    samples = np.arange(artifact.shape[1], dtype=np.float64)
    injected = 0.05 * np.sin(2 * np.pi * samples / VOLUME_SAMPLES)

    correction = slice_fastr(artifact + injected, triggers)

    fitted = fitted_span(triggers)
    retained = tone_amplitude(correction.data[0], VOLUME_SAMPLES, fitted)
    ratio = retained / tone_amplitude(injected, VOLUME_SAMPLES, fitted)
    assert 0.70 < ratio < 0.85


def test_slice_fastr_leaves_dead_time_and_recording_edges_untouched() -> None:
    triggers = make_multiband_triggers()
    data = make_gradient_artifact(
        np.outer([1.0, -2.0], np.ones(triggers.size)),
        triggers=triggers,
    )

    correction = slice_fastr(data, triggers)

    dead_time = volume_dead_time(triggers, correction.provenance)
    assert dead_time.stop - dead_time.start > 30
    np.testing.assert_array_equal(correction.data[:, dead_time], data[:, dead_time])
    head = slice(0, math.floor(triggers[0]) - 5)
    np.testing.assert_array_equal(correction.data[:, head], data[:, head])


def test_slice_fastr_is_exact_when_triggers_need_no_interpolation() -> None:
    """Whole-sample triggers and an exactly repeated artifact cancel completely.

    Only once the neighbour window has slid past the first group, whose epoch has
    no preceding artifact to contain, as at any recording edge.
    """
    triggers = FIRST_TRIGGER_SAMPLE + np.arange(GROUP_COUNT) * 50.0
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0), triggers=triggers)

    correction = slice_fastr(data, triggers, interpolation_factor=1)

    interior = slice(int(triggers[32]), int(triggers[-1]))
    assert np.max(np.abs(correction.data[:, interior])) < 1e-12
    assert correction.provenance.search_radius == 3


def test_slice_fastr_rejects_alignment_that_reorders_groups() -> None:
    """A fitted epoch that overtakes its neighbour makes the fit ambiguous."""
    data = np.random.default_rng(0).normal(size=(2, 600))
    triggers = 20.0 + np.arange(100) * 5.0

    with pytest.raises(FastrInputError, match="overlap"):
        slice_fastr(
            data,
            triggers,
            interpolation_factor=1,
            search_radius_samples=3,
        )


def test_slice_fastr_returns_provenance_for_every_group_and_channel() -> None:
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0))

    provenance = slice_fastr(data, make_group_triggers()).provenance

    assert provenance.interpolation_factor == 10
    assert provenance.search_radius == 30
    assert provenance.samples_before_trigger == 19
    assert provenance.samples_after_trigger == 461
    assert provenance.shifts.shape == (GROUP_COUNT,)
    assert provenance.correlations.shape == (GROUP_COUNT,)
    assert provenance.amplitudes.shape == (2, GROUP_COUNT)
    assert np.all(provenance.correlations > 0.99)


@pytest.mark.parametrize(
    ("data", "triggers", "message"),
    [
        (np.zeros(4_000), make_group_triggers(), "channels, samples"),
        (np.zeros((2, 4_000)) + np.inf, make_group_triggers(), "finite"),
        (np.zeros((2, 4_000)), make_group_triggers()[:1], "one-dimensional"),
        (np.zeros((2, 4_000)), make_group_triggers()[::-1], "strictly increasing"),
        (np.zeros((2, 4_000)), make_group_triggers() + 0.03, "interpolated sample"),
        (np.zeros((2, 4_000)), make_group_triggers()[:40], "opposite trigger parity"),
        (np.zeros((2, 3_000)), make_group_triggers(), "beyond the recording"),
        (np.zeros((2, 4_000)), make_group_triggers() - 20.0, "beyond the recording"),
    ],
)
def test_slice_fastr_rejects_invalid_inputs(
    data: np.ndarray,
    triggers: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        slice_fastr(data, triggers)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"interpolation_factor": 0}, "interpolation factor"),
        ({"neighbor_count": 31}, "even"),
        ({"neighbor_count": 1}, "at least two"),
        ({"search_radius_samples": -1}, "search radius"),
    ],
)
def test_slice_fastr_rejects_invalid_parameters(
    parameters: dict[str, int],
    message: str,
) -> None:
    data = make_gradient_artifact(constant_amplitudes(1.0, -2.0))

    with pytest.raises(FastrInputError, match=message):
        slice_fastr(data, make_group_triggers(), **parameters)


def make_residual_artifact(group_amplitudes: np.ndarray) -> np.ndarray:
    """Group-locked residual above the OBS high-pass that no one template fits."""
    triggers = make_group_triggers()
    samples = np.arange(int(triggers[-1]) + 60, dtype=np.float64)
    owners = np.clip(np.searchsorted(triggers, samples, side="right") - 1, 0, None)
    harmonics = np.arange(5, 9)[:, np.newaxis]
    phases = samples / GROUP_INTERVAL_SAMPLES
    shape = np.sum(np.sin(2 * np.pi * harmonics * phases + harmonics), axis=0)
    return group_amplitudes[:, owners] * shape


def drifting_amplitudes(*channel_scales: float) -> np.ndarray:
    """Per-group amplitudes that no single average template can represent."""
    drift = 1.0 + 0.6 * np.sin(2 * np.pi * np.arange(GROUP_COUNT) / 13.0)
    return np.outer(channel_scales, drift)


def test_residual_obs_removes_group_locked_residual_variance() -> None:
    triggers = make_group_triggers()
    residual = make_residual_artifact(drifting_amplitudes(1.0, -2.0))

    cleaned = residual_obs(
        residual,
        triggers,
        sampling_rate=SAMPLING_RATE,
        excluded_channels=(),
    )

    fitted = fitted_span(triggers)
    before = np.sqrt(np.mean(residual[:, fitted] ** 2))
    after = np.sqrt(np.mean(cleaned[:, fitted] ** 2))
    assert after / before < 0.2


def test_residual_obs_preserves_signal_below_the_high_pass() -> None:
    triggers = make_group_triggers()
    residual = make_residual_artifact(drifting_amplitudes(1.0, -2.0))
    samples = np.arange(residual.shape[1], dtype=np.float64)
    injected = np.sin(2 * np.pi * 10.0 * samples / SAMPLING_RATE)

    cleaned = residual_obs(
        residual + injected,
        triggers,
        sampling_rate=SAMPLING_RATE,
        excluded_channels=(),
    )

    fitted = fitted_span(triggers)
    period = SAMPLING_RATE / 10.0
    retained = tone_amplitude(cleaned[0], period, fitted)
    assert retained / tone_amplitude(injected, period, fitted) > 0.95


def test_residual_obs_leaves_excluded_channels_untouched() -> None:
    triggers = make_group_triggers()
    residual = make_residual_artifact(drifting_amplitudes(1.0, -2.0))

    cleaned = residual_obs(
        residual,
        triggers,
        sampling_rate=SAMPLING_RATE,
        excluded_channels=(1,),
    )

    np.testing.assert_array_equal(cleaned[1], residual[1])
    assert not np.allclose(cleaned[0], residual[0])


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"sampling_rate": 0.0}, "sampling rate"),
        ({"rank": 0}, "rank"),
        ({"rank": GROUP_COUNT + 1}, "rank"),
        ({"excluded_channels": (2,)}, "excluded channels"),
        ({"excluded_channels": (0, 1)}, "excluded channels"),
    ],
)
def test_residual_obs_rejects_invalid_parameters(
    parameters: dict[str, object],
    message: str,
) -> None:
    residual = make_residual_artifact(drifting_amplitudes(1.0, -2.0))
    arguments: dict[str, object] = {
        "sampling_rate": SAMPLING_RATE,
        "excluded_channels": (),
    }
    arguments.update(parameters)

    with pytest.raises(FastrInputError, match=message):
        residual_obs(residual, make_group_triggers(), **arguments)


def make_position_dependent_artifact(triggers: np.ndarray) -> np.ndarray:
    """An artifact whose waveform differs per acquisition slot."""
    return simulate_gradient_artifact(
        triggers,
        sample_count=int(triggers[-1]) + 180,
        channel_count=2,
        sampling_rate=1_000.0,
        readout_seconds=0.045,
        amplitude_drift=0.0,
        groups_per_volume=GROUPS_PER_VOLUME,
    )


def test_acquisition_group_fastr_matches_acquisition_slots() -> None:
    triggers = make_multiband_triggers()
    data = make_position_dependent_artifact(triggers)

    neighbors = acquisition_group_fastr(
        data,
        make_multiband_volume_starts(),
        sampling_rate=SAMPLING_RATE,
        timing=make_acquisition_slot_timing(),
        neighbor_count=2,
    ).provenance.neighbor_indices

    targets = np.arange(triggers.size)[:, np.newaxis]
    assert neighbors.shape == (triggers.size, 2)
    assert np.all(neighbors % GROUPS_PER_VOLUME == targets % GROUPS_PER_VOLUME)
    assert not np.any(neighbors == targets)


def test_acquisition_group_fastr_suppresses_slot_dependent_artifact() -> None:
    """Slot waveforms differ, so only same-slot neighbours cancel them."""
    triggers = make_multiband_triggers()
    data = make_position_dependent_artifact(triggers)
    fitted = fitted_span(triggers)
    raw_rms = np.sqrt(np.mean(data[:, fitted] ** 2))

    parity = slice_fastr(data, triggers, neighbor_count=2)
    matched = acquisition_group_fastr(
        data,
        make_multiband_volume_starts(),
        sampling_rate=SAMPLING_RATE,
        timing=make_acquisition_slot_timing(),
        neighbor_count=2,
    )

    parity_rms = np.sqrt(np.mean(parity.data[:, fitted] ** 2))
    matched_rms = np.sqrt(np.mean(matched.data[:, fitted] ** 2))
    assert parity_rms / raw_rms > 0.5
    assert matched_rms / raw_rms < 0.01


def test_acquisition_group_fastr_corrects_volume_dead_time() -> None:
    """Matched neighbours share a gap structure, so the gap can be corrected."""
    triggers = make_multiband_triggers()
    data = make_position_dependent_artifact(triggers)

    correction = acquisition_group_fastr(
        data,
        make_multiband_volume_starts(),
        sampling_rate=SAMPLING_RATE,
        timing=make_acquisition_slot_timing(),
        neighbor_count=2,
    )

    provenance = correction.provenance
    largest_gap = np.max(np.diff(triggers)) * provenance.interpolation_factor
    assert provenance.samples_after_trigger == int(largest_gap)
    gap = slice(
        math.ceil(triggers[GROUPS_PER_VOLUME - 1] + 0.046 * SAMPLING_RATE),
        math.floor(triggers[GROUPS_PER_VOLUME]),
    )
    assert gap.stop - gap.start > 20
    assert not np.array_equal(correction.data[:, gap], data[:, gap])


def test_acquisition_group_fastr_keeps_a_target_out_of_its_own_template() -> None:
    triggers = make_multiband_triggers()
    data = make_position_dependent_artifact(triggers)
    perturbed = triggers.size // 2
    # The epoch reaches the largest gap, so scale everything the fit sees.
    span = slice(int(triggers[perturbed]), int(triggers[perturbed + 2]))
    data = data.copy()
    data[0, span] *= 1.5

    fitted = acquisition_group_fastr(
        data,
        make_multiband_volume_starts(),
        sampling_rate=SAMPLING_RATE,
        timing=make_acquisition_slot_timing(),
        neighbor_count=2,
    ).provenance.amplitudes

    assert fitted[0, perturbed] == pytest.approx(1.5, abs=0.02)
    assert fitted[1, perturbed] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"neighbor_count": 4}, "volumes"),
    ],
)
def test_acquisition_group_fastr_rejects_an_impossible_window(
    parameters: dict[str, int],
    message: str,
) -> None:
    triggers = make_multiband_triggers()
    data = make_position_dependent_artifact(triggers)
    arguments: dict[str, int] = {"neighbor_count": 2}
    arguments.update(parameters)

    with pytest.raises(FastrInputError, match=message):
        acquisition_group_fastr(
            data,
            make_multiband_volume_starts(),
            sampling_rate=SAMPLING_RATE,
            timing=make_acquisition_slot_timing(),
            **(arguments | parameters),
        )
