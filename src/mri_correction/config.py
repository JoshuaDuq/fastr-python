"""Validated configuration for the public correction pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    """Raised when a YAML configuration does not describe a valid run."""


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Input files required for one correction run."""

    raw_vhdr: Path
    fmri_metadata: Path


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output file requested for one correction run."""

    vhdr: Path


@dataclass(frozen=True, slots=True)
class TimingConfig:
    """Marker definition used to identify volume starts."""

    marker_type: str
    marker_description: str


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Algorithm and resource settings for one correction run."""

    method: str
    interpolation_factor: int
    neighbor_count: int
    search_radius_samples: int
    lowpass_hz: float
    output_sampling_rate_hz: float
    channel_batch_size: int
    reference_channel: str | int


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """How the pipeline restricts its output to the scanning period."""

    mode: str = "none"
    minimum_epoch_coverage: float = 0.75

    def __post_init__(self) -> None:
        if self.mode not in _TRIM_MODES:
            raise ConfigurationError(
                f"trim.mode must be one of {sorted(_TRIM_MODES)}"
            )
        if not 0.0 < self.minimum_epoch_coverage <= 1.0:
            raise ConfigurationError(
                "trim.minimum_epoch_coverage must be greater than 0 and at most 1"
            )


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Complete immutable configuration for a correction run."""

    input: InputConfig
    output: OutputConfig
    timing: TimingConfig
    processing: ProcessingConfig
    trim: TrimConfig


_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing", "trim"})
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing"})
_TRIM_MODES = frozenset({"none", "first_to_last_volume"})
_TRIM_KEYS = frozenset({"mode", "minimum_epoch_coverage"})
_INPUT_KEYS = frozenset({"raw_vhdr", "fmri_metadata"})
_OUTPUT_KEYS = frozenset({"vhdr"})
_TIMING_KEYS = frozenset({"marker_type", "marker_description"})
_PROCESSING_KEYS = frozenset(
    {
        "method",
        "interpolation_factor",
        "neighbor_count",
        "search_radius_samples",
        "lowpass_hz",
        "output_sampling_rate_hz",
        "channel_batch_size",
        "reference_channel",
    }
)
_SUPPORTED_METHODS = frozenset({"acquisition_group_fastr"})


def load_config(path: str | Path) -> CorrectionConfig:
    """Load and validate one YAML configuration document.

    Relative paths are resolved against the directory containing ``path``. The
    loader validates the configuration structure and scalar values, but does not
    create outputs or require input files to exist.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"configuration file does not exist: {config_path}"
        raise ConfigurationError(message) from error
    except yaml.YAMLError as error:
        message = f"invalid YAML in configuration: {config_path}"
        raise ConfigurationError(message) from error

    root = _require_mapping(document, "configuration")
    _reject_unknown_keys(root, _TOP_LEVEL_KEYS, "configuration")
    for section in sorted(_REQUIRED_TOP_LEVEL_KEYS):
        if section not in root:
            raise ConfigurationError(f"missing required field: {section}")

    input_values = _section(root, "input", _INPUT_KEYS)
    output_values = _section(root, "output", _OUTPUT_KEYS)
    timing_values = _section(root, "timing", _TIMING_KEYS)
    processing_values = _section(root, "processing", _PROCESSING_KEYS)

    base_directory = config_path.parent
    return CorrectionConfig(
        input=InputConfig(
            raw_vhdr=_path_value(input_values, "raw_vhdr", base_directory),
            fmri_metadata=_path_value(
                input_values,
                "fmri_metadata",
                base_directory,
            ),
        ),
        output=OutputConfig(
            vhdr=_path_value(output_values, "vhdr", base_directory),
        ),
        timing=TimingConfig(
            marker_type=_string_value(timing_values, "marker_type"),
            marker_description=_string_value(
                timing_values,
                "marker_description",
            ),
        ),
        processing=_processing_config(processing_values),
        trim=_trim_config(root),
    )


def _trim_config(root: Mapping[str, object]) -> TrimConfig:
    """Read the optional trim section, defaulting each absent field."""
    if "trim" not in root:
        return TrimConfig()
    values = _require_mapping(root["trim"], "trim")
    _reject_unknown_keys(values, _TRIM_KEYS, "trim")
    defaults = TrimConfig()
    mode = _string_value(values, "mode") if "mode" in values else defaults.mode
    coverage = (
        _finite_number(values, "minimum_epoch_coverage", minimum=0.0)
        if "minimum_epoch_coverage" in values
        else defaults.minimum_epoch_coverage
    )
    return TrimConfig(mode=mode, minimum_epoch_coverage=coverage)


def _processing_config(values: Mapping[str, object]) -> ProcessingConfig:
    method = _string_value(values, "method")
    if method not in _SUPPORTED_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_METHODS))
        raise ConfigurationError(
            f"processing.method must be one of {supported}; got {method!r}"
        )

    interpolation_factor = _integer_value(
        values,
        "interpolation_factor",
        minimum=1,
    )
    neighbor_count = _integer_value(values, "neighbor_count", minimum=2)
    if neighbor_count % 2:
        raise ConfigurationError("processing.neighbor_count must be even")

    return ProcessingConfig(
        method=method,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=_integer_value(
            values,
            "search_radius_samples",
            minimum=0,
        ),
        lowpass_hz=_finite_number(values, "lowpass_hz", minimum=0.0),
        output_sampling_rate_hz=_finite_number(
            values,
            "output_sampling_rate_hz",
            minimum=0.0,
        ),
        channel_batch_size=_integer_value(
            values,
            "channel_batch_size",
            minimum=1,
        ),
        reference_channel=_reference_channel(values),
    )


def _reference_channel(values: Mapping[str, object]) -> str | int:
    value = values.get("reference_channel")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ConfigurationError(
            "processing.reference_channel must be a nonempty channel name "
            "or nonnegative channel index"
        )
    if isinstance(value, str) and not value:
        raise ConfigurationError("processing.reference_channel cannot be empty")
    if isinstance(value, int) and value < 0:
        raise ConfigurationError(
            "processing.reference_channel must be nonnegative"
        )
    return value


def _section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, object]:
    values = _require_mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    for field_name in sorted(expected_keys):
        if field_name not in values:
            raise ConfigurationError(
                f"missing required field: {name}.{field_name}"
            )
    return values


def _require_mapping(value: object, field_path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_path} must be a mapping")
    return value


def _reject_unknown_keys(
    values: Mapping[str, object],
    expected_keys: frozenset[str],
    field_path: str,
) -> None:
    unknown = sorted(set(values) - expected_keys)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field_path}: {', '.join(unknown)}"
        )


def _required_value(values: Mapping[str, object], name: str) -> object:
    if name not in values:
        raise ConfigurationError(f"missing required field: {name}")
    return values[name]


def _string_value(values: Mapping[str, object], name: str) -> str:
    value = _required_value(values, name)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
    if any(character in value for character in ("\0", "\n", "\r")):
        raise ConfigurationError(f"{name} contains an invalid character")
    return value


def _path_value(
    values: Mapping[str, object],
    name: str,
    base_directory: Path,
) -> Path:
    value = _string_value(values, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _integer_value(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = _required_value(values, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_number(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: float,
) -> float:
    value = _required_value(values, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number <= minimum:
        raise ConfigurationError(f"{name} must be greater than {minimum}")
    return number
