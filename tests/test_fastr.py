import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mri_correction.fastr import (
    FastrInputError,
    FmriAcquisitionTiming,
    load_bids_fmri_timing,
    make_group_trigger_samples,
)

GROUP_OFFSETS_SECONDS = (
    0.0,
    0.0475,
    0.0975,
    0.145,
    0.195,
    0.2425,
    0.2925,
    0.34,
    0.39,
    0.4375,
    0.4875,
    0.535,
    0.585,
    0.6325,
    0.6825,
    0.73,
    0.78,
    0.83,
)


def make_real_acquisition_timing() -> FmriAcquisitionTiming:
    slice_timing = tuple(
        offset
        for _multiband_slice in range(3)
        for offset in GROUP_OFFSETS_SECONDS
    )
    return FmriAcquisitionTiming(
        repetition_time_seconds=0.9,
        slice_timing_seconds=slice_timing,
        multiband_acceleration_factor=3,
    )


def test_acquisition_timing_exposes_immutable_sorted_groups() -> None:
    timing = make_real_acquisition_timing()

    assert timing.group_offsets_seconds == GROUP_OFFSETS_SECONDS
    assert timing.groups_per_volume == 18
    with pytest.raises(FrozenInstanceError):
        timing.repetition_time_seconds = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("repetition_time", "slice_timing", "multiband_factor", "message"),
    [
        (0.0, (0.0,), 1, "repetition time"),
        (float("nan"), (0.0,), 1, "repetition time"),
        (True, (0.0,), 1, "repetition time"),
        (0.9, (), 1, "slice timing"),
        (0.9, (float("inf"),), 1, "slice timing"),
        (0.9, (True,), 1, "slice timing"),
        (0.9, (-0.01,), 1, "slice timing"),
        (0.9, (0.9,), 1, "slice timing"),
        (0.9, (0.0,), 0, "multiband"),
        (0.9, (0.0,), True, "multiband"),
        (0.9, (0.0, 0.1, 0.1), 2, "divisible"),
        (0.9, (0.0, 0.0, 0.1, 0.2), 2, "exactly"),
    ],
)
def test_acquisition_timing_rejects_invalid_metadata(
    repetition_time: object,
    slice_timing: tuple[object, ...],
    multiband_factor: object,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        FmriAcquisitionTiming(
            repetition_time_seconds=repetition_time,  # type: ignore[arg-type]
            slice_timing_seconds=slice_timing,  # type: ignore[arg-type]
            multiband_acceleration_factor=multiband_factor,  # type: ignore[arg-type]
        )


def test_load_bids_fmri_timing_reads_required_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "bold.json"
    metadata_path.write_text(
        json.dumps(
            {
                "RepetitionTime": 0.9,
                "SliceTiming": list(
                    make_real_acquisition_timing().slice_timing_seconds
                ),
                "MultibandAccelerationFactor": 3,
            }
        ),
        encoding="utf-8",
    )

    timing = load_bids_fmri_timing(metadata_path)

    assert timing.repetition_time_seconds == 0.9
    assert timing.group_offsets_seconds == GROUP_OFFSETS_SECONDS


@pytest.mark.parametrize(
    "metadata",
    [
        {"SliceTiming": [0.0], "MultibandAccelerationFactor": 1},
        {"RepetitionTime": 0.9, "MultibandAccelerationFactor": 1},
        {"RepetitionTime": 0.9, "SliceTiming": [0.0]},
    ],
)
def test_load_bids_fmri_timing_rejects_missing_fields(
    tmp_path, metadata: dict[str, object]
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FastrInputError, match="missing required field"):
        load_bids_fmri_timing(metadata_path)


def test_load_bids_fmri_timing_chains_json_errors(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(FastrInputError, match="valid JSON") as error:
        load_bids_fmri_timing(metadata_path)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_load_bids_fmri_timing_chains_io_errors(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FastrInputError, match="read fMRI metadata") as error:
        load_bids_fmri_timing(missing_path)

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {
            "RepetitionTime": "0.9",
            "SliceTiming": [0.0],
            "MultibandAccelerationFactor": 1,
        },
        {
            "RepetitionTime": 0.9,
            "SliceTiming": "0.0",
            "MultibandAccelerationFactor": 1,
        },
    ],
)
def test_load_bids_fmri_timing_chains_type_errors(
    tmp_path, metadata: object
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FastrInputError) as error:
        load_bids_fmri_timing(metadata_path)

    assert isinstance(error.value.__cause__, TypeError)


def test_make_group_triggers_preserves_fractional_sample_positions() -> None:
    timing = make_real_acquisition_timing()

    triggers = make_group_trigger_samples(
        np.array([100, 4600], dtype=np.int64),
        sampling_rate=5000.0,
        timing=timing,
    )

    assert triggers.dtype == np.float64
    assert triggers.shape == (36,)
    assert triggers[0] == 100.0
    assert triggers[1] == 337.5
    assert triggers[18] == 4600.0
    np.testing.assert_allclose(
        triggers[:18],
        100.0 + np.asarray(GROUP_OFFSETS_SECONDS) * 5000.0,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("volume_starts", "sampling_rate", "message"),
    [
        (np.array([], dtype=np.int64), 5000.0, "nonempty"),
        (np.array([[0, 4500]], dtype=np.int64), 5000.0, "one-dimensional"),
        (np.array([0, 4500], dtype=np.float64), 5000.0, "integer"),
        (np.array([False, True]), 5000.0, "boolean"),
        (np.array([4500, 0]), 5000.0, "strictly increasing"),
        (np.array([0, 4500, 9001]), 5000.0, "jitter"),
        (np.array([0, 4500, 22500]), 5000.0, "acquisition gap"),
        (np.array([0, 4500]), True, "sampling rate"),
        (np.array([0, 4500]), float("inf"), "sampling rate"),
        (np.array([0, 4500]), 5000.5, "integer number"),
    ],
)
def test_make_group_triggers_rejects_invalid_inputs(
    volume_starts: np.ndarray,
    sampling_rate: object,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        make_group_trigger_samples(
            volume_starts,
            sampling_rate=sampling_rate,  # type: ignore[arg-type]
            timing=make_real_acquisition_timing(),
        )


@pytest.mark.parametrize(
    ("volume_starts", "message"),
    [
        (np.array([0, 4_500, 9_001]), "jitter"),
        (np.array([0, 4_500, 8_999]), "jitter"),
        (np.array([0, 4_500, 9_000, 326_672]), "acquisition gap"),
        (np.array([0, 4_500, 13_500]), "acquisition gap"),
    ],
)
def test_make_group_triggers_names_jitter_and_gaps_apart(
    volume_starts: np.ndarray,
    message: str,
) -> None:
    """Both fail, but a missing marker and a jittered one need different fixes."""
    with pytest.raises(FastrInputError, match=message):
        make_group_trigger_samples(
            volume_starts,
            sampling_rate=5000.0,
            timing=make_real_acquisition_timing(),
        )


def test_make_group_triggers_reports_where_the_spacing_breaks() -> None:
    with pytest.raises(FastrInputError, match="markers 2 and 3") as error:
        make_group_trigger_samples(
            np.array([0, 4_500, 9_001, 13_501]),
            sampling_rate=5000.0,
            timing=make_real_acquisition_timing(),
        )

    assert "+1" in str(error.value)
