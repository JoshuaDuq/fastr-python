from pathlib import Path

import mne
import numpy as np
import pytest
from pybv import write_brainvision

from mri_correction.brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from mri_correction.brainvision_io import (
    BrainVisionInputError,
    read_brainvision_recording,
    resample_markers,
    select_marker_samples,
    write_brainvision_recording,
)


def make_markers() -> tuple[BrainVisionMarker, ...]:
    return (
        BrainVisionMarker(
            "New Segment",
            "",
            1,
            1,
            0,
            "20260826123456123456",
        ),
        BrainVisionMarker("Volume", "volume-start", 1, 1, 0),
        BrainVisionMarker("Comment", "comma, preserved", 501, 10, 2),
        BrainVisionMarker("Volume", "volume-start", 901, 1, 0),
    )


def make_source_recording(tmp_path: Path) -> Path:
    data = np.zeros((2, 1_000), dtype=np.float64)
    write_brainvision(
        data=data,
        sfreq=1_000.0,
        ch_names=["EEG 001", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(marker_path, "source.eeg", make_markers())
    return tmp_path / "source.vhdr"


def test_read_recording_resolves_header_references_and_markers(tmp_path: Path) -> None:
    vhdr_path = make_source_recording(tmp_path)

    recording = read_brainvision_recording(vhdr_path)

    assert recording.data_path == tmp_path / "source.eeg"
    assert recording.marker_path == tmp_path / "source.vmrk"
    assert recording.markers == make_markers()


def test_read_recording_accepts_recorder_header_identifier(tmp_path: Path) -> None:
    vhdr_path = make_source_recording(tmp_path)
    header = vhdr_path.read_text(encoding="utf-8")
    vhdr_path.write_text(
        header.replace(
            "Brain Vision Data Exchange Header File Version 1.0",
            "BrainVision Data Exchange Header File Version 1.0",
            1,
        ),
        encoding="utf-8",
    )

    recording = read_brainvision_recording(vhdr_path)

    assert recording.data_path == tmp_path / "source.eeg"


def test_select_marker_samples_requires_exact_configured_match() -> None:
    markers = make_markers()

    samples = select_marker_samples(
        markers,
        marker_type="Volume",
        marker_description="volume-start",
        sample_count=1_000,
    )

    np.testing.assert_array_equal(samples, np.array([0, 900], dtype=np.int64))


def test_select_marker_samples_rejects_missing_or_duplicate_positions() -> None:
    markers = (*make_markers(),
        BrainVisionMarker("Volume", "volume-start", 1, 1, 0),
    )

    with pytest.raises(BrainVisionInputError, match="duplicate"):
        select_marker_samples(
            markers,
            marker_type="Volume",
            marker_description="volume-start",
            sample_count=1_000,
        )

    with pytest.raises(BrainVisionInputError, match="no markers"):
        select_marker_samples(
            markers,
            marker_type="Volume",
            marker_description="missing",
            sample_count=1_000,
        )


def test_resample_markers_maps_positions_and_sizes_without_losing_fields() -> None:
    markers = make_markers()

    transformed = resample_markers(markers, factor=2)

    assert transformed == (
        BrainVisionMarker(
            "New Segment",
            "",
            1,
            1,
            0,
            "20260826123456123456",
        ),
        BrainVisionMarker("Volume", "volume-start", 1, 1, 0),
        BrainVisionMarker("Comment", "comma, preserved", 251, 5, 2),
        BrainVisionMarker("Volume", "volume-start", 451, 1, 0),
    )


def test_write_recording_reopens_in_mne_and_preserves_marker_descriptions(
    tmp_path: Path,
) -> None:
    vhdr_path = tmp_path / "source.vhdr"
    data = np.arange(2_000, dtype=np.float64).reshape(2, 1_000) * 1e-8
    output_markers = resample_markers(make_markers(), factor=2)

    write_brainvision_recording(
        data=data[:, ::2],
        sampling_rate=500.0,
        channel_names=["EEG 001", "ECG"],
        output_vhdr=vhdr_path,
        markers=output_markers,
    )

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    assert raw.info["sfreq"] == 500.0
    assert raw.get_data().shape == (2, 500)
    np.testing.assert_allclose(raw.get_data(), data[:, ::2], rtol=0.0, atol=1e-12)
    _, markers = read_brainvision_markers(vhdr_path.with_suffix(".vmrk"))
    assert markers == output_markers
    assert set(raw.annotations.description) >= {
        "Volume/volume-start",
        "Comment/comma, preserved",
    }


def test_write_recording_refuses_existing_output_stem(tmp_path: Path) -> None:
    output = tmp_path / "result.vhdr"
    output.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_brainvision_recording(
            data=np.zeros((1, 10)),
            sampling_rate=100.0,
            channel_names=["EEG 001"],
            output_vhdr=output,
            markers=(),
        )
