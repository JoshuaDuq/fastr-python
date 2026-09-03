"""Both marker conventions, and the single source of truth each one needs."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pybv import write_brainvision

from fastr_python.config import ConfigurationError, load_config
from fastr_python.fastr import (
    FastrInputError,
    FmriAcquisitionTiming,
    slice_marker_geometry,
    volume_marker_geometry,
)
from fastr_python.io.brainvision import (
    BrainVisionMarker,
    write_brainvision_markers,
)
from fastr_python.pipeline import run_correction

SAMPLING_RATE = 1_000.0
SAMPLES_PER_VOLUME = 100
GROUPS_PER_VOLUME = 2
GROUP_STRIDE = SAMPLES_PER_VOLUME // GROUPS_PER_VOLUME
VOLUME_COUNT = 7


def group_markers(
    *,
    volume_count: int = VOLUME_COUNT,
    groups_per_volume: int = GROUPS_PER_VOLUME,
) -> np.ndarray:
    """Zero-based samples of one marker per acquisition group, evenly spaced."""
    stride = SAMPLES_PER_VOLUME // groups_per_volume
    volumes = np.arange(volume_count, dtype=np.int64) * SAMPLES_PER_VOLUME
    offsets = np.arange(groups_per_volume, dtype=np.int64) * stride
    return (volumes[:, np.newaxis] + offsets).reshape(-1)


def declared_timing() -> FmriAcquisitionTiming:
    return FmriAcquisitionTiming(
        repetition_time_seconds=SAMPLES_PER_VOLUME / SAMPLING_RATE,
        slice_timing_seconds=(0.0, GROUP_STRIDE / SAMPLING_RATE),
        multiband_acceleration_factor=1,
    )


# --- measuring geometry from acquisition-group markers ----------------------


def test_group_markers_supply_their_own_timing() -> None:
    markers = group_markers()

    acquisition = slice_marker_geometry(
        markers,
        sampling_rate=SAMPLING_RATE,
        groups_per_volume=GROUPS_PER_VOLUME,
    )

    assert acquisition.source == "measured_group_markers"
    assert acquisition.groups_per_volume == GROUPS_PER_VOLUME
    assert acquisition.volume_count == VOLUME_COUNT
    assert acquisition.repetition_time_seconds == pytest.approx(0.1)
    assert acquisition.group_offsets_seconds == pytest.approx((0.0, 0.05))
    # Recorded, not derived: the triggers are the markers themselves.
    assert np.array_equal(acquisition.group_triggers, markers.astype(np.float64))
    assert np.array_equal(
        acquisition.volume_starts,
        markers[::GROUPS_PER_VOLUME],
    )


def test_declared_timing_and_measured_markers_agree() -> None:
    """The two conventions describe the same acquisition, so they must match."""
    markers = group_markers()

    measured = slice_marker_geometry(
        markers,
        sampling_rate=SAMPLING_RATE,
        groups_per_volume=GROUPS_PER_VOLUME,
    )
    derived = volume_marker_geometry(
        markers[::GROUPS_PER_VOLUME],
        sampling_rate=SAMPLING_RATE,
        timing=declared_timing(),
    )

    assert derived.source == "declared_slice_timing"
    assert np.array_equal(measured.group_triggers, derived.group_triggers)
    assert measured.groups_per_volume == derived.groups_per_volume
    assert measured.repetition_time_seconds == pytest.approx(
        derived.repetition_time_seconds
    )


def test_a_partial_volume_of_group_markers_is_rejected() -> None:
    with pytest.raises(FastrInputError, match="whole 2-group volumes"):
        slice_marker_geometry(
            group_markers()[:-1],
            sampling_rate=SAMPLING_RATE,
            groups_per_volume=GROUPS_PER_VOLUME,
        )


def test_a_dropped_group_marker_moves_every_later_volume_boundary() -> None:
    """Deleting one marker keeps the count even but shifts every boundary."""
    markers = np.delete(group_markers(), 4)

    with pytest.raises(FastrInputError, match="samples apart instead of"):
        slice_marker_geometry(
            np.append(markers, markers[-1] + GROUP_STRIDE),
            sampling_rate=SAMPLING_RATE,
            groups_per_volume=GROUPS_PER_VOLUME,
        )


def test_drifting_within_volume_offsets_are_rejected() -> None:
    markers = group_markers()
    markers[5] += 20

    with pytest.raises(FastrInputError, match="offsets have to repeat"):
        slice_marker_geometry(
            markers,
            sampling_rate=SAMPLING_RATE,
            groups_per_volume=GROUPS_PER_VOLUME,
        )


def test_one_volume_cannot_yield_a_repetition_time() -> None:
    with pytest.raises(FastrInputError, match="at least two volumes"):
        slice_marker_geometry(
            group_markers(volume_count=1),
            sampling_rate=SAMPLING_RATE,
            groups_per_volume=GROUPS_PER_VOLUME,
        )


def test_a_wrong_group_count_is_self_consistent_on_its_own() -> None:
    """Counting whole volumes together measures a multiple of the repetition time.

    The markers cannot refute this reading, which is exactly why the group
    count has to be declared rather than inferred.
    """
    acquisition = slice_marker_geometry(
        group_markers(volume_count=8),
        sampling_rate=SAMPLING_RATE,
        groups_per_volume=2 * GROUPS_PER_VOLUME,
    )

    assert acquisition.repetition_time_seconds == pytest.approx(0.2)


def test_a_declared_repetition_time_catches_a_wrong_group_count() -> None:
    with pytest.raises(FastrInputError, match="count excitations or slices"):
        slice_marker_geometry(
            group_markers(volume_count=8),
            sampling_rate=SAMPLING_RATE,
            groups_per_volume=2 * GROUPS_PER_VOLUME,
            expected_repetition_time_seconds=0.1,
        )


def test_a_matching_declared_repetition_time_is_accepted() -> None:
    acquisition = slice_marker_geometry(
        group_markers(),
        sampling_rate=SAMPLING_RATE,
        groups_per_volume=GROUPS_PER_VOLUME,
        expected_repetition_time_seconds=0.1,
    )

    assert acquisition.repetition_time_seconds == pytest.approx(0.1)


# --- one source of truth, enforced in the configuration ---------------------


def write_config(tmp_path: Path, document: dict) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def base_document(**timing: object) -> dict:
    return {
        "input": {"raw_vhdr": "source.vhdr"},
        "output": {"vhdr": "corrected.vhdr"},
        "timing": {
            "marker_type": "Volume",
            "marker_description": "volume-start",
            **timing,
        },
        "processing": {
            "method": "acquisition_group_fastr",
            "interpolation_factor": 2,
            "neighbor_count": 2,
            "search_radius_samples": 0,
            "lowpass_hz": 20.0,
            "output_sampling_rate_hz": 500.0,
            "channel_batch_size": 2,
            "reference_channel": "EEG 001",
            "line_noise_frequencies_hz": [],
        },
    }


def inline_acquisition() -> dict:
    return {
        "repetition_time_seconds": 0.1,
        "slice_timing_seconds": [0.0, 0.05],
        "multiband_acceleration_factor": 1,
    }


def test_inline_acquisition_replaces_the_bids_sidecar(tmp_path: Path) -> None:
    document = base_document()
    document["acquisition"] = inline_acquisition()

    config = load_config(write_config(tmp_path, document))

    assert config.input.fmri_metadata is None
    assert config.acquisition == declared_timing()


def test_two_declared_sources_are_rejected(tmp_path: Path) -> None:
    document = base_document()
    document["input"]["fmri_metadata"] = "bold.json"
    document["acquisition"] = inline_acquisition()

    with pytest.raises(ConfigurationError, match="declare the acquisition timing once"):
        load_config(write_config(tmp_path, document))


def test_volume_markers_require_a_declared_source(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="need declared slice timing"):
        load_config(write_config(tmp_path, base_document()))


def test_group_markers_reject_a_declared_source(tmp_path: Path) -> None:
    document = base_document(marker_kind="slice", groups_per_volume=2)
    document["acquisition"] = inline_acquisition()

    with pytest.raises(ConfigurationError, match="record their own timing"):
        load_config(write_config(tmp_path, document))


def test_group_markers_require_a_group_count(tmp_path: Path) -> None:
    document = base_document(marker_kind="slice")

    with pytest.raises(ConfigurationError, match="groups_per_volume is required"):
        load_config(write_config(tmp_path, document))


def test_volume_markers_reject_a_group_count(tmp_path: Path) -> None:
    document = base_document(groups_per_volume=2)
    document["acquisition"] = inline_acquisition()

    with pytest.raises(ConfigurationError, match="only valid when"):
        load_config(write_config(tmp_path, document))


def test_group_markers_cannot_be_repaired(tmp_path: Path) -> None:
    document = base_document(
        marker_kind="slice",
        groups_per_volume=2,
        missing_volume_markers="repair",
        expected_volume_count=7,
    )

    with pytest.raises(ConfigurationError, match="must be 'error' when"):
        load_config(write_config(tmp_path, document))


def test_a_declared_repetition_time_needs_group_markers(tmp_path: Path) -> None:
    document = base_document(expected_repetition_time_seconds=0.1)
    document["acquisition"] = inline_acquisition()

    with pytest.raises(ConfigurationError, match="repetition time is declared"):
        load_config(write_config(tmp_path, document))


def test_an_unknown_marker_kind_is_rejected(tmp_path: Path) -> None:
    document = base_document(marker_kind="event")
    document["acquisition"] = inline_acquisition()

    with pytest.raises(ConfigurationError, match="marker_kind must be one of"):
        load_config(write_config(tmp_path, document))


def test_an_incomplete_acquisition_section_is_rejected(tmp_path: Path) -> None:
    document = base_document()
    document["acquisition"] = {"repetition_time_seconds": 0.1}

    with pytest.raises(ConfigurationError, match="missing required field"):
        load_config(write_config(tmp_path, document))


def test_invalid_inline_timing_reports_the_timing_error(tmp_path: Path) -> None:
    document = base_document()
    document["acquisition"] = {
        "repetition_time_seconds": 0.1,
        "slice_timing_seconds": [0.0, 0.5],
        "multiband_acceleration_factor": 1,
    }

    with pytest.raises(ConfigurationError, match="invalid acquisition section"):
        load_config(write_config(tmp_path, document))


# --- end to end, over one recording carrying both marker kinds --------------


def make_dual_marked_recording(tmp_path: Path) -> None:
    """Write one recording marked both per volume and per acquisition group."""
    data = np.zeros((3, 800), dtype=np.float64)
    data[0, 120:140] = 1e-5
    data[1, 120:140] = 2e-5
    data[2, 120:140] = 3e-5
    write_brainvision(
        data=data,
        sfreq=SAMPLING_RATE,
        ch_names=["EEG 001", "EEG 002", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        unit="µV",
        events=[],
    )
    markers = [BrainVisionMarker("New Segment", "", 1, 1, 0)]
    for sample in group_markers():
        position = int(sample) + 1
        if sample % SAMPLES_PER_VOLUME == 0:
            markers.append(
                BrainVisionMarker("Volume", "volume-start", position, 1, 0)
            )
        markers.append(BrainVisionMarker("Slice", "slice-start", position, 1, 0))
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(marker_path, "source.eeg", tuple(markers))


def run_variant(tmp_path: Path, name: str, document: dict) -> Path:
    """Run one configuration into its own output name and return the data file."""
    document["output"] = {"vhdr": f"{name}.vhdr"}
    config_path = tmp_path / f"{name}.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return run_correction(load_config(config_path)).output_eeg


def test_group_markers_and_declared_timing_correct_identically(
    tmp_path: Path,
) -> None:
    """The same acquisition, described two ways, must correct to the same data."""
    make_dual_marked_recording(tmp_path)
    (tmp_path / "bold.json").write_text(
        json.dumps(
            {
                "RepetitionTime": 0.1,
                "SliceTiming": [0.0, 0.05],
                "MultibandAccelerationFactor": 1,
            }
        ),
        encoding="utf-8",
    )

    from_sidecar = base_document()
    from_sidecar["input"]["fmri_metadata"] = "bold.json"
    from_inline = base_document()
    from_inline["acquisition"] = inline_acquisition()
    from_markers = base_document(
        marker_type="Slice",
        marker_description="slice-start",
        marker_kind="slice",
        groups_per_volume=2,
    )

    sidecar_eeg = run_variant(tmp_path, "sidecar", from_sidecar)
    inline_eeg = run_variant(tmp_path, "inline", from_inline)
    markers_eeg = run_variant(tmp_path, "markers", from_markers)

    assert sidecar_eeg.read_bytes() == inline_eeg.read_bytes()
    assert sidecar_eeg.read_bytes() == markers_eeg.read_bytes()


def test_measured_timing_is_recorded_in_the_sidecar(tmp_path: Path) -> None:
    make_dual_marked_recording(tmp_path)
    document = base_document(
        marker_type="Slice",
        marker_description="slice-start",
        marker_kind="slice",
        groups_per_volume=2,
    )
    document["output"] = {"vhdr": "corrected.vhdr"}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    summary = run_correction(load_config(config_path))

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    timing = provenance["timing"]
    assert timing["marker_kind"] == "slice"
    assert timing["group_position_source"] == "measured_group_markers"
    assert timing["declared_timing_source"] is None
    assert timing["declared"] is None
    assert timing["resolved"] == {
        "repetition_time_seconds": 0.1,
        "groups_per_volume": 2,
        "group_offsets_seconds": [0.0, 0.05],
        "volume_count": VOLUME_COUNT,
    }
    assert provenance["input"]["fmri_metadata"] is None
    assert provenance["input"]["sha256"]["fmri_metadata"] is None


def test_declared_timing_is_recorded_alongside_the_resolved_geometry(
    tmp_path: Path,
) -> None:
    make_dual_marked_recording(tmp_path)
    document = base_document()
    document["acquisition"] = inline_acquisition()
    document["output"] = {"vhdr": "corrected.vhdr"}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    summary = run_correction(load_config(config_path))

    timing = json.loads(summary.provenance_json.read_text(encoding="utf-8"))["timing"]
    assert timing["declared_timing_source"] == "configuration"
    assert timing["declared"] == {
        "repetition_time_seconds": 0.1,
        "slice_timing_seconds": [0.0, 0.05],
        "multiband_acceleration_factor": 1,
    }
    assert timing["group_position_source"] == "declared_slice_timing"
