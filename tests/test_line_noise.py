import numpy as np

from eegfmri_fastr import pipeline_io
from eegfmri_fastr.metrics import tone_transfer


def test_spectrum_fit_removes_only_the_configured_line() -> None:
    sampling_rate = 1_000.0
    times = np.arange(int(20 * sampling_rate)) / sampling_rate
    nearby = np.sin(2 * np.pi * 59.7 * times)
    line = 3.0 * np.sin(2 * np.pi * 60.0 * times)
    data = (nearby + line)[np.newaxis, :]

    cleaned = pipeline_io.remove_line_noise(
        data,
        sampling_rate=sampling_rate,
        frequencies_hz=(60.0,),
    )

    line_transfer = tone_transfer(
        line,
        cleaned[0],
        frequency=60.0,
        sampling_rate=sampling_rate,
    )
    nearby_transfer = tone_transfer(
        nearby,
        cleaned[0],
        frequency=59.7,
        sampling_rate=sampling_rate,
    )
    assert line_transfer.amplitude_ratio < 0.05
    assert nearby_transfer.amplitude_ratio > 0.99


def test_line_noise_frequencies_must_stay_below_nyquist() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "below the Nyquist frequency",
    ):
        pipeline_io.remove_line_noise(
            np.zeros((1, 1_000)),
            sampling_rate=1_000.0,
            frequencies_hz=(500.0,),
        )
