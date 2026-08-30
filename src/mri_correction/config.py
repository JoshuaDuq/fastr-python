"""Validated configuration for the public correction pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import yaml

from .residual_qc import residual_qc_defaults


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
    line_noise_frequencies_hz: tuple[float, ...]
    non_eeg_channels: tuple[str, ...] = ("ECG",)
    template_high_pass_hz: float = 1.0
    residual_threshold_uv: float = 1.0
    residual_gate: bool = False
    residual_obs: bool = False
    residual_obs_rank: int = 4
    adaptive_window: bool = False
    local_neighbor_count: int = 20
    residual_gate_mad_multiplier: float = 8.0
    residual_gate_ratio: float = 8.0
    residual_gate_max_fraction: float = 0.02
    adaptive_improvement_ratio: float = 0.85


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """How the pipeline restricts its output to the scanning period."""

    mode: str = "none"

    def __post_init__(self) -> None:
        if self.mode not in _TRIM_MODES:
            raise ConfigurationError(
                f"trim.mode must be one of {sorted(_TRIM_MODES)}"
            )


@dataclass(frozen=True, slots=True)
class QualityControlConfig:
    """Settings for residual-gradient measurements and annotations."""

    block_seconds: float = 30.0
    mains_frequency_hz: float = 60.0
    mains_exclusion_hz: float = 1.0
    # How far above a channel's own median residual a block must sit, and on
    # how many channels at once, before it is annotated. See residual_qc.
    residual_mad_multiplier: float = residual_qc_defaults.MAD_MULTIPLIER
    residual_minimum_channels: int = residual_qc_defaults.MINIMUM_CHANNELS


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Settings for generated PSD diagnostics."""

    psd_max_frequency_hz: float = 100.0
    psd_n_fft: int | None = None


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Complete immutable configuration for a correction run."""

    input: InputConfig
    output: OutputConfig
    timing: TimingConfig
    processing: ProcessingConfig
    trim: TrimConfig
    quality_control: QualityControlConfig = field(
        default_factory=QualityControlConfig,
    )
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


_TOP_LEVEL_KEYS = frozenset(
    {
        "input",
        "output",
        "timing",
        "processing",
        "trim",
        "quality_control",
        "diagnostics",
    }
)
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing"})
_TRIM_MODES = frozenset({"none", "first_to_last_volume"})
_TRIM_KEYS = frozenset({"mode"})
_INPUT_KEYS = frozenset({"raw_vhdr", "fmri_metadata"})
_OUTPUT_KEYS = frozenset({"vhdr"})
_TIMING_KEYS = frozenset({"marker_type", "marker_description"})
_QUALITY_CONTROL_KEYS = frozenset(
    {
        "block_seconds",
        "mains_frequency_hz",
        "mains_exclusion_hz",
        "residual_mad_multiplier",
        "residual_minimum_channels",
    }
)
_DIAGNOSTICS_KEYS = frozenset({"psd_max_frequency_hz", "psd_n_fft"})
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
        "line_noise_frequencies_hz",
        "non_eeg_channels",
        "template_high_pass_hz",
        "residual_threshold_uv",
        "residual_gate",
        "residual_obs",
        "residual_obs_rank",
        "adaptive_window",
        "local_neighbor_count",
        "residual_gate_mad_multiplier",
        "residual_gate_ratio",
        "residual_gate_max_fraction",
        "adaptive_improvement_ratio",
    }
)
_OPTIONAL_PROCESSING_KEYS = frozenset(
    {
        "non_eeg_channels",
        "template_high_pass_hz",
        "residual_threshold_uv",
        "residual_gate",
        "residual_obs",
        "residual_obs_rank",
        "adaptive_window",
        "local_neighbor_count",
        "residual_gate_mad_multiplier",
        "residual_gate_ratio",
        "residual_gate_max_fraction",
        "adaptive_improvement_ratio",
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
    processing_values = _section(
        root,
        "processing",
        _PROCESSING_KEYS,
        optional_keys=_OPTIONAL_PROCESSING_KEYS,
    )

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
        quality_control=_quality_control_config(root),
        diagnostics=_diagnostics_config(root),
    )


def _trim_config(root: Mapping[str, object]) -> TrimConfig:
    """Read the optional trim section, defaulting each absent field."""
    if "trim" not in root:
        return TrimConfig()
    values = _require_mapping(root["trim"], "trim")
    _reject_unknown_keys(values, _TRIM_KEYS, "trim")
    defaults = TrimConfig()
    mode = _string_value(values, "mode") if "mode" in values else defaults.mode
    return TrimConfig(mode=mode)


def _quality_control_config(
    root: Mapping[str, object],
) -> QualityControlConfig:
    values = _optional_section(root, "quality_control", _QUALITY_CONTROL_KEYS)
    return QualityControlConfig(
        block_seconds=_optional_finite_number(
            values,
            "block_seconds",
            default=30.0,
            minimum=0.0,
        ),
        mains_frequency_hz=_optional_finite_number(
            values,
            "mains_frequency_hz",
            default=60.0,
            minimum=0.0,
        ),
        residual_mad_multiplier=_optional_finite_number(
            values,
            "residual_mad_multiplier",
            default=residual_qc_defaults.MAD_MULTIPLIER,
            minimum=0.0,
            inclusive=True,
        ),
        residual_minimum_channels=_optional_positive_integer(
            values,
            "residual_minimum_channels",
            default=residual_qc_defaults.MINIMUM_CHANNELS,
        ),
        mains_exclusion_hz=_optional_finite_number(
            values,
            "mains_exclusion_hz",
            default=1.0,
            minimum=0.0,
            inclusive=True,
        ),
    )


def _diagnostics_config(root: Mapping[str, object]) -> DiagnosticsConfig:
    values = _optional_section(root, "diagnostics", _DIAGNOSTICS_KEYS)
    psd_n_fft = _optional_integer_or_none(values, "psd_n_fft")
    return DiagnosticsConfig(
        psd_max_frequency_hz=_optional_finite_number(
            values,
            "psd_max_frequency_hz",
            default=100.0,
            minimum=0.0,
        ),
        psd_n_fft=psd_n_fft,
    )


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

    defaults = ProcessingConfig.__dataclass_fields__["template_high_pass_hz"].default
    template_high_pass_hz = (
        _finite_number(
            values,
            "template_high_pass_hz",
            minimum=0.0,
            inclusive=True,
        )
        if "template_high_pass_hz" in values
        else defaults
    )

    residual_threshold_uv = (
        _finite_number(
            values,
            "residual_threshold_uv",
            minimum=0.0,
            inclusive=True,
        )
        if "residual_threshold_uv" in values
        else ProcessingConfig.__dataclass_fields__["residual_threshold_uv"].default
    )
    residual_gate = (
        _boolean_value(values, "residual_gate")
        if "residual_gate" in values
        else ProcessingConfig.__dataclass_fields__["residual_gate"].default
    )
    residual_obs = (
        _boolean_value(values, "residual_obs")
        if "residual_obs" in values
        else ProcessingConfig.__dataclass_fields__["residual_obs"].default
    )
    residual_obs_rank = _optional_positive_integer(
        values,
        "residual_obs_rank",
        default=ProcessingConfig.__dataclass_fields__["residual_obs_rank"].default,
    )
    adaptive_window = (
        _boolean_value(values, "adaptive_window")
        if "adaptive_window" in values
        else ProcessingConfig.__dataclass_fields__["adaptive_window"].default
    )
    local_neighbor_count = (
        _integer_value(values, "local_neighbor_count", minimum=2)
        if "local_neighbor_count" in values
        else ProcessingConfig.__dataclass_fields__["local_neighbor_count"].default
    )
    if local_neighbor_count % 2:
        raise ConfigurationError("processing.local_neighbor_count must be even")

    residual_gate_mad_multiplier = _optional_finite_number(
        values,
        "residual_gate_mad_multiplier",
        default=8.0,
        minimum=0.0,
    )
    residual_gate_ratio = _optional_finite_number(
        values,
        "residual_gate_ratio",
        default=8.0,
        minimum=0.0,
    )
    residual_gate_max_fraction = _optional_finite_number(
        values,
        "residual_gate_max_fraction",
        default=0.02,
        minimum=0.0,
        maximum=1.0,
    )
    adaptive_improvement_ratio = _optional_finite_number(
        values,
        "adaptive_improvement_ratio",
        default=0.85,
        minimum=0.0,
        maximum=1.0,
    )
    output_sampling_rate_hz = _finite_number(
        values,
        "output_sampling_rate_hz",
        minimum=0.0,
    )
    line_noise_frequencies_hz = _line_noise_frequencies(
        values,
        nyquist_hz=0.5 * output_sampling_rate_hz,
    )

    return ProcessingConfig(
        method=method,
        residual_threshold_uv=residual_threshold_uv,
        residual_gate=residual_gate,
        residual_obs=residual_obs,
        residual_obs_rank=residual_obs_rank,
        adaptive_window=adaptive_window,
        local_neighbor_count=local_neighbor_count,
        residual_gate_mad_multiplier=residual_gate_mad_multiplier,
        residual_gate_ratio=residual_gate_ratio,
        residual_gate_max_fraction=residual_gate_max_fraction,
        adaptive_improvement_ratio=adaptive_improvement_ratio,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        template_high_pass_hz=template_high_pass_hz,
        search_radius_samples=_integer_value(
            values,
            "search_radius_samples",
            minimum=0,
        ),
        lowpass_hz=_finite_number(values, "lowpass_hz", minimum=0.0),
        output_sampling_rate_hz=output_sampling_rate_hz,
        channel_batch_size=_integer_value(
            values,
            "channel_batch_size",
            minimum=1,
        ),
        reference_channel=_reference_channel(values),
        line_noise_frequencies_hz=line_noise_frequencies_hz,
        non_eeg_channels=_non_eeg_channels(values),
    )


def _non_eeg_channels(values: Mapping[str, object]) -> tuple[str, ...]:
    """Names of channels the correction must not fit a scalar or a basis to."""
    default = ProcessingConfig.__dataclass_fields__["non_eeg_channels"].default
    if "non_eeg_channels" not in values:
        return default
    names = values["non_eeg_channels"]
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ConfigurationError(
            "processing.non_eeg_channels must be a list of nonempty channel names"
        )
    return tuple(names)


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


def _line_noise_frequencies(
    values: Mapping[str, object],
    *,
    nyquist_hz: float,
) -> tuple[float, ...]:
    value = values.get("line_noise_frequencies_hz")
    if not isinstance(value, list):
        raise ConfigurationError(
            "processing.line_noise_frequencies_hz must be a list"
        )
    frequencies: list[float] = []
    for frequency in value:
        if isinstance(frequency, bool) or not isinstance(frequency, Real):
            raise ConfigurationError(
                "processing.line_noise_frequencies_hz must contain numbers"
            )
        numeric = float(frequency)
        if not math.isfinite(numeric) or numeric <= 0.0 or numeric >= nyquist_hz:
            raise ConfigurationError(
                "processing.line_noise_frequencies_hz must contain finite, "
                "positive frequencies below the output Nyquist frequency"
            )
        frequencies.append(numeric)
    if len(frequencies) != len(set(frequencies)):
        raise ConfigurationError(
            "processing.line_noise_frequencies_hz must not contain duplicates"
        )
    return tuple(frequencies)


def _section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
    *,
    optional_keys: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    values = _require_mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    for field_name in sorted(expected_keys - optional_keys):
        if field_name not in values:
            raise ConfigurationError(
                f"missing required field: {name}.{field_name}"
            )
    return values


def _optional_section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, object]:
    if name not in root:
        return {}
    values = _require_mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    return values


def _optional_finite_number(
    values: Mapping[str, object],
    name: str,
    *,
    default: float,
    minimum: float,
    inclusive: bool = False,
    maximum: float | None = None,
) -> float:
    if name not in values:
        return default
    return _finite_number(
        values,
        name,
        minimum=minimum,
        inclusive=inclusive,
        maximum=maximum,
    )


def _optional_positive_integer(
    values: Mapping[str, object],
    name: str,
    *,
    default: int,
) -> int:
    if name not in values or values[name] is None:
        return default
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _optional_integer_or_none(
    values: Mapping[str, object],
    name: str,
) -> int | None:
    if name not in values or values[name] is None:
        return None
    return _integer_value(values, name, minimum=1)


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


def _boolean_value(values: Mapping[str, object], name: str) -> bool:
    value = _required_value(values, name)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


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
    inclusive: bool = False,
    maximum: float | None = None,
) -> float:
    value = _required_value(values, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    number = float(value)
    too_small = number < minimum if inclusive else number <= minimum
    too_large = maximum is not None and number > maximum
    if not math.isfinite(number) or too_small or too_large:
        if too_large:
            raise ConfigurationError(f"{name} must be less than or equal to {maximum}")
        bound = "greater than or equal to" if inclusive else "greater than"
        raise ConfigurationError(f"{name} must be {bound} {minimum}")
    return number
