import math
from pathlib import Path

import mne
import numpy as np
import pytest
from pybv import write_brainvision

from benchmark.probes import (
    BETWEEN_HARMONICS,
    ON_HARMONIC,
    ProbeError,
    Tone,
    plan_tones,
    synthesize_tones,
    write_tone_injected_recording,
)
from fastr_python.io.brainvision import BrainVisionMarker, write_brainvision_markers
from fastr_python.io.recording import read_brainvision_recording
from fastr_python.validation.metrics import tone_transfer

SAMPLING_RATE = 1_000.0
TR = 0.9


def _make_recording(tmp_path: Path, *, channels: list[str], samples: int) -> Path:
    write_brainvision(
        data=np.zeros((len(channels), samples)),
        sfreq=SAMPLING_RATE,
        ch_names=channels,
        fname_base="source",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "source.eeg",
        (BrainVisionMarker("Volume", "V  1", 1, 1, 0),),
    )
    return tmp_path / "source.vhdr"


def test_plan_places_one_tone_on_and_one_between_volume_harmonics():
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(23.0,),
        mains_hz=60.0,
    )
    assert [tone.placement for tone in tones] == [ON_HARMONIC, BETWEEN_HARMONICS]
    volume_rate = 1.0 / TR
    on, between = (tone.frequency_hz for tone in tones)
    assert on % volume_rate == pytest.approx(0.0, abs=1e-9)
    assert (between % volume_rate) == pytest.approx(volume_rate / 2, abs=1e-9)


def test_plan_drops_tones_colliding_with_mains():
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(60.0,),
        mains_hz=60.0,
        mains_guard_hz=0.3,
    )
    assert [tone.placement for tone in tones] == [BETWEEN_HARMONICS]
    assert all(abs(tone.frequency_hz - 60.0) >= 0.3 for tone in tones)


def test_plan_rejects_a_plan_with_nothing_left():
    with pytest.raises(ProbeError, match="no tone survived"):
        plan_tones(
            repetition_time_seconds=TR,
            amplitude_volts=5e-6,
            band_centres_hz=(60.0,),
            mains_hz=60.0,
            mains_guard_hz=10.0,
        )


def test_plan_spreads_phases_so_tones_do_not_sum_into_one_transient():
    tones = plan_tones(repetition_time_seconds=TR, amplitude_volts=1.0, mains_hz=60.0)
    waveform = synthesize_tones(
        tones, sample_count=int(SAMPLING_RATE), sampling_rate=SAMPLING_RATE
    )
    assert abs(waveform[0]) < 0.5 * len(tones)


def test_synthesized_tone_is_recovered_at_its_exact_amplitude():
    tone = Tone(
        frequency_hz=23.0,
        amplitude_volts=5e-6,
        placement=ON_HARMONIC,
        phase_radians=0.4,
    )
    waveform = synthesize_tones(
        (tone,), sample_count=10_000, sampling_rate=SAMPLING_RATE
    )
    transfer = tone_transfer(
        waveform, waveform, frequency=tone.frequency_hz, sampling_rate=SAMPLING_RATE
    )
    assert transfer.amplitude_ratio == pytest.approx(1.0, rel=1e-9)
    assert math.isclose(transfer.phase_error_degrees, 0.0, abs_tol=1e-6)


def test_tones_are_exactly_separable_over_an_even_number_of_volumes():
    tones = plan_tones(repetition_time_seconds=TR, amplitude_volts=5e-6, mains_hz=60.0)
    samples = round(66 * TR * SAMPLING_RATE)
    combined = synthesize_tones(
        tones, sample_count=samples, sampling_rate=SAMPLING_RATE
    )
    for tone in tones:
        alone = synthesize_tones(
            (tone,), sample_count=samples, sampling_rate=SAMPLING_RATE
        )
        transfer = tone_transfer(
            alone, combined, frequency=tone.frequency_hz, sampling_rate=SAMPLING_RATE
        )
        assert transfer.amplitude_ratio == pytest.approx(1.0, rel=1e-9)


def test_tones_leak_into_each_other_over_an_odd_number_of_volumes():
    tones = plan_tones(repetition_time_seconds=TR, amplitude_volts=5e-6, mains_hz=60.0)
    samples = round(67 * TR * SAMPLING_RATE)
    combined = synthesize_tones(
        tones, sample_count=samples, sampling_rate=SAMPLING_RATE
    )
    alone = synthesize_tones(
        (tones[0],), sample_count=samples, sampling_rate=SAMPLING_RATE
    )
    transfer = tone_transfer(
        alone, combined, frequency=tones[0].frequency_hz, sampling_rate=SAMPLING_RATE
    )
    assert abs(transfer.amplitude_ratio - 1.0) > 1e-6


def test_synthesis_rejects_a_tone_at_or_above_nyquist():
    tone = Tone(
        frequency_hz=600.0,
        amplitude_volts=1.0,
        placement=ON_HARMONIC,
        phase_radians=0.0,
    )
    with pytest.raises(ProbeError, match="Nyquist"):
        synthesize_tones((tone,), sample_count=100, sampling_rate=SAMPLING_RATE)


def test_injection_adds_tones_to_eeg_and_leaves_excluded_channels_alone(tmp_path):
    source = _make_recording(tmp_path, channels=["Fp1", "Cz", "ECG"], samples=20_000)
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(23.0,),
        mains_hz=60.0,
    )
    output = tmp_path / "injected.vhdr"
    write_tone_injected_recording(
        source, output, tones=tones, excluded_channels=("ECG",)
    )

    raw = mne.io.read_raw_brainvision(output, preload=True, verbose="error")
    data = raw.get_data()
    expected = synthesize_tones(
        tones, sample_count=data.shape[1], sampling_rate=SAMPLING_RATE
    )
    assert data[raw.ch_names.index("Fp1")] == pytest.approx(expected, abs=1e-12)
    assert data[raw.ch_names.index("ECG")] == pytest.approx(0.0, abs=1e-12)


def test_injection_preserves_the_marker_stream(tmp_path):
    source = _make_recording(tmp_path, channels=["Fp1", "ECG"], samples=5_000)
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(23.0,),
        mains_hz=60.0,
    )
    output = tmp_path / "injected.vhdr"
    write_tone_injected_recording(
        source, output, tones=tones, excluded_channels=("ECG",)
    )
    assert (
        read_brainvision_recording(output).markers
        == read_brainvision_recording(source).markers
    )


def test_injection_refuses_to_exclude_every_channel(tmp_path):
    source = _make_recording(tmp_path, channels=["ECG"], samples=1_000)
    tones = plan_tones(
        repetition_time_seconds=TR,
        amplitude_volts=5e-6,
        band_centres_hz=(23.0,),
        mains_hz=60.0,
    )
    with pytest.raises(ProbeError, match="every channel is excluded"):
        write_tone_injected_recording(
            source, tmp_path / "out.vhdr", tones=tones, excluded_channels=("ECG",)
        )
