"""Exact marker timing operations."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt

_MARKER_FILE_IDENTIFIER = "BrainVision Data Exchange Marker File Version 1.0"


class BrainVisionMarkerError(ValueError):
    """Raised when BrainVision marker data or syntax is invalid."""


def _validate_marker_text(value: str, *, field_name: str, allow_empty: bool) -> None:
    if not isinstance(value, str):
        raise BrainVisionMarkerError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise BrainVisionMarkerError(f"{field_name} cannot be empty")
    if any(character in value for character in ("\n", "\r", "\0")):
        raise BrainVisionMarkerError(f"{field_name} contains an invalid character")
    if r"\1" in value:
        message = f"{field_name} contains the ambiguous BrainVision comma placeholder"
        raise BrainVisionMarkerError(message)


def _validate_marker_integer(value: int, *, field_name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise BrainVisionMarkerError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )


def _validate_marker_date(value: str | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 20
        or not value.isascii()
        or not value.isdigit()
    ):
        raise BrainVisionMarkerError(
            "date must use the 20-digit YYYYMMDDhhmmssuuuuuu format"
        )
    try:
        datetime.strptime(value, "%Y%m%d%H%M%S%f")
    except ValueError as error:
        raise BrainVisionMarkerError("date is not a valid calendar date") from error


@dataclass(frozen=True, slots=True)
class BrainVisionMarker:
    """One losslessly represented BrainVision marker."""

    marker_type: str
    description: str
    position: int
    size: int
    channel: int
    date: str | None = None

    def __post_init__(self) -> None:
        _validate_marker_text(
            self.marker_type,
            field_name="marker_type",
            allow_empty=False,
        )
        _validate_marker_text(
            self.description,
            field_name="description",
            allow_empty=True,
        )
        _validate_marker_integer(self.position, field_name="position", minimum=1)
        _validate_marker_integer(self.size, field_name="size", minimum=1)
        _validate_marker_integer(self.channel, field_name="channel", minimum=0)
        _validate_marker_date(self.date)


def _parse_marker_integer_field(value: str, *, field_name: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise BrainVisionMarkerError(
            f"marker {field_name} must contain only decimal digits"
        )
    return int(value)


def _parse_marker_line(line: str) -> tuple[int, BrainVisionMarker]:
    key, separator, value = line.partition("=")
    index_text = key.removeprefix("Mk")
    if (
        not separator
        or not index_text.isascii()
        or not index_text.isdigit()
        or index_text != str(int(index_text))
    ):
        raise BrainVisionMarkerError(f"malformed marker declaration: {line}")

    fields = value.split(",")
    if len(fields) not in (5, 6):
        raise BrainVisionMarkerError(f"malformed marker declaration: {line}")

    marker_type, description, position, size, channel = fields[:5]
    date = (fields[5] or None) if len(fields) == 6 else None
    try:
        marker = BrainVisionMarker(
            marker_type=marker_type.replace(r"\1", ","),
            description=description.replace(r"\1", ","),
            position=_parse_marker_integer_field(position, field_name="position"),
            size=_parse_marker_integer_field(size, field_name="size"),
            channel=_parse_marker_integer_field(channel, field_name="channel"),
            date=date,
        )
    except ValueError as error:
        raise BrainVisionMarkerError(f"invalid marker declaration: {line}") from error
    return int(index_text), marker


def _parse_common_info_line(line: str, common_infos: dict[str, str]) -> None:
    key, separator, value = line.partition("=")
    if not separator or key not in ("Codepage", "DataFile"):
        raise BrainVisionMarkerError(f"unsupported Common Infos line: {line}")
    if key in common_infos:
        raise BrainVisionMarkerError(f"duplicate Common Infos {key} declaration")
    common_infos[key] = value


def _enter_marker_file_section(line: str, sections: list[str]) -> str:
    expected_section = "[Common Infos]" if not sections else "[Marker Infos]"
    if line != expected_section:
        raise BrainVisionMarkerError(f"unexpected or duplicate section: {line}")
    sections.append(line)
    return line


def _validate_data_file_name(data_file_name: str) -> None:
    if not isinstance(data_file_name, str) or not data_file_name:
        raise BrainVisionMarkerError("data_file_name must be a nonempty string")
    if any(character in data_file_name for character in ("\n", "\r", "\0")):
        raise BrainVisionMarkerError("data_file_name contains an invalid character")


def read_brainvision_markers(
    path: str | PathLike[str],
) -> tuple[str, tuple[BrainVisionMarker, ...]]:
    """Read a strict BrainVision marker file without losing marker fields."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != _MARKER_FILE_IDENTIFIER:
        raise BrainVisionMarkerError("invalid BrainVision marker-file identifier")

    section: str | None = None
    sections: list[str] = []
    common_infos: dict[str, str] = {}
    indexed_markers: list[tuple[int, BrainVisionMarker]] = []
    for line in lines[1:]:
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            section = _enter_marker_file_section(line, sections)
        elif section == "[Common Infos]":
            _parse_common_info_line(line, common_infos)
        elif section == "[Marker Infos]":
            indexed_markers.append(_parse_marker_line(line))
        else:
            raise BrainVisionMarkerError(f"content outside a supported section: {line}")

    if sections != ["[Common Infos]", "[Marker Infos]"]:
        raise BrainVisionMarkerError(
            "marker file must contain one Common Infos and one Marker Infos section"
        )
    if common_infos.get("Codepage") != "UTF-8":
        raise BrainVisionMarkerError("Common Infos Codepage must be declared as UTF-8")
    data_file_name = common_infos.get("DataFile")
    if not data_file_name:
        raise BrainVisionMarkerError("missing Common Infos DataFile declaration")
    _validate_data_file_name(data_file_name)
    indexed_markers.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed_markers]
    if indices != list(range(1, len(indices) + 1)):
        raise BrainVisionMarkerError("marker indices must be unique and contiguous")
    return data_file_name, tuple(marker for _, marker in indexed_markers)


def _format_marker_line(index: int, marker: BrainVisionMarker) -> str:
    marker_type = marker.marker_type.replace(",", r"\1")
    description = marker.description.replace(",", r"\1")
    fields = [
        marker_type,
        description,
        str(marker.position),
        str(marker.size),
        str(marker.channel),
    ]
    if marker.date is not None:
        fields.append(marker.date)
    return f"Mk{index}={','.join(fields)}"


def write_brainvision_markers(
    path: str | PathLike[str],
    data_file_name: str,
    markers: Iterable[BrainVisionMarker],
) -> None:
    """Write a lossless BrainVision marker file without overwriting existing data."""
    _validate_data_file_name(data_file_name)
    marker_values = tuple(markers)
    if any(not isinstance(marker, BrainVisionMarker) for marker in marker_values):
        raise BrainVisionMarkerError("markers must contain BrainVisionMarker instances")

    lines = [
        _MARKER_FILE_IDENTIFIER,
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data_file_name}",
        "",
        "[Marker Infos]",
    ]
    lines.extend(
        _format_marker_line(index, marker)
        for index, marker in enumerate(marker_values, start=1)
    )
    content = "\n".join(lines) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as marker_file:
        marker_file.write(content)


class MarkerTimingError(ValueError):
    """Raised when marker timing is invalid or ambiguous."""


def split_volume_blocks(
    volume_samples: npt.ArrayLike,
    *,
    samples_per_volume: int,
    declared_block_starts: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Split zero-based volume samples at explicitly declared discontinuities."""
    samples = np.asarray(volume_samples)
    _validate_volume_samples(samples, samples_per_volume)

    boundary_indices = np.flatnonzero(np.diff(samples) != samples_per_volume) + 1
    observed_starts = samples[np.concatenate(([0], boundary_indices))]
    if declared_block_starts is None:
        if boundary_indices.size:
            message = "volume markers contain an undeclared acquisition gap"
            raise MarkerTimingError(message)
    else:
        declared_starts = np.asarray(declared_block_starts)
        _validate_declared_starts(declared_starts)
        if not np.array_equal(declared_starts, observed_starts):
            message = "block declarations mismatch; a marker gap may be undeclared"
            raise MarkerTimingError(message)
    return tuple(np.split(samples.astype(np.int64, copy=False), boundary_indices))


def map_brainvision_position(input_position: int, *, factor: int) -> int:
    """Map a one-based BrainVision position after integer-factor resampling."""
    if not isinstance(input_position, int) or input_position < 1:
        raise MarkerTimingError("BrainVision positions must be positive integers")
    if not isinstance(factor, int) or factor < 1:
        raise MarkerTimingError("resampling factor must be a positive integer")
    return (input_position - 1) // factor + 1


def _validate_volume_samples(samples: np.ndarray, samples_per_volume: int) -> None:
    if samples.ndim != 1 or samples.size == 0:
        message = "volume samples must be a non-empty one-dimensional array"
        raise MarkerTimingError(message)
    if not np.issubdtype(samples.dtype, np.integer):
        raise MarkerTimingError("volume samples must contain integers")
    if np.any(samples < 0):
        raise MarkerTimingError("volume samples cannot be negative")
    if not isinstance(samples_per_volume, int) or samples_per_volume < 1:
        raise MarkerTimingError("samples_per_volume must be a positive integer")
    if np.any(np.diff(samples) <= 0):
        raise MarkerTimingError("volume samples must be strictly increasing")


def _validate_declared_starts(starts: np.ndarray) -> None:
    if starts.ndim != 1 or starts.size == 0:
        raise MarkerTimingError("declared block starts must be a non-empty array")
    if not np.issubdtype(starts.dtype, np.integer):
        raise MarkerTimingError("declared block starts must contain integers")
    if np.any(starts < 0) or np.any(np.diff(starts) <= 0):
        raise MarkerTimingError("declared block starts must be strictly increasing")
