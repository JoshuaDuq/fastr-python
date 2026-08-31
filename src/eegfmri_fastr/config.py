"""Validated configuration for the public correction pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import yaml

from .fastr_timing import FmriAcquisitionTiming
from .fastr_types import FastrInputError
from .residual_qc import residual_qc_defaults


__all__ = [
    "ConfigurationError",
    "CorrectionConfig",
    "DiagnosticsConfig",
    "InputConfig",
    "OutputConfig",
    "ProcessingConfig",
    "QualityControlConfig",
    "TimingConfig",
    "TrimConfig",
    "load_config",
]


class ConfigurationError(ValueError):
    """Raised when a YAML configuration does not describe a valid run."""


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Input files required for one correction run.

    ``fmri_metadata`` is absent when the acquisition timing is declared inline
    or measured from acquisition-group markers.
    """

    raw_vhdr: Path
    fmri_metadata: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output file requested for one correction run."""

    vhdr: Path


@dataclass(frozen=True, slots=True)
class TimingConfig:
    """Which markers locate the acquisition, and what they each stand for.

    ``marker_kind`` is ``volume`` when one marker begins a volume and the group
    positions inside it come from declared slice timing, or ``slice`` when every
    acquisition group is marked and its position is read from the recording.
    ``groups_per_volume`` belongs to the second case only, because a marked
    excitation does not say which volume it starts, and
    ``expected_repetition_time_seconds`` is the optional check on that count.
    """

    marker_type: str
    marker_description: str
    marker_kind: str = "volume"
    groups_per_volume: int | None = None
    expected_repetition_time_seconds: float | None = None
    missing_volume_markers: str = "error"
    expected_volume_count: int | None = None
    volume_marker_start_index: int | None = None
    volume_marker_count: int | None = None


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
    residual_obs_rank: int | str = 4
    residual_obs_section_seconds: float | None = None
    pre_trigger_fraction: float = 0.03
    adaptive_noise_cancellation: bool = False
    adaptive_window: bool = False
    channel_adaptive_window: bool = False
    local_neighbor_count: int = 20
    local_window_channels: tuple[str, ...] = ()
    residual_gate_mad_multiplier: float = 8.0
    residual_gate_ratio: float = 8.0
    residual_gate_max_fraction: float = 0.02
    adaptive_improvement_ratio: float = 0.85
    channel_failure_policy: str = "report"


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
    # Highest volume harmonic reported, capped at the output Nyquist frequency.
    volume_spectrum_max_hz: float = 110.0
    report_channel_outliers: bool = True
    # Absolute floor a channel-block residual must clear before the automatic
    # policy may call it a failure. Unlike residual_threshold_uv this one is
    # never relative: a spatial comparison alone cannot separate a broken
    # electrode from a quiet recording. See residual_qc.
    bad_channel_residual_uv: float = 5.0


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Settings for generated PSD diagnostics."""

    psd_max_frequency_hz: float = 100.0
    psd_n_fft: int | None = None


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Complete immutable configuration for a correction run.

    ``acquisition`` holds slice timing declared inline instead of read from a
    BIDS sidecar. It is absent when ``input.fmri_metadata`` supplies the same
    timing, and when acquisition-group markers make it unnecessary.
    """

    input: InputConfig
    output: OutputConfig
    timing: TimingConfig
    processing: ProcessingConfig
    trim: TrimConfig
    acquisition: FmriAcquisitionTiming | None = None
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
        "acquisition",
        "trim",
        "quality_control",
        "diagnostics",
    }
)
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing"})
_TRIM_MODES = frozenset({"none", "first_to_last_volume"})
_TRIM_KEYS = frozenset({"mode"})
_INPUT_KEYS = frozenset({"raw_vhdr", "fmri_metadata"})
_OPTIONAL_INPUT_KEYS = frozenset({"fmri_metadata"})
_OUTPUT_KEYS = frozenset({"vhdr"})
_ACQUISITION_KEYS = frozenset(
    {
        "repetition_time_seconds",
        "slice_timing_seconds",
        "multiband_acceleration_factor",
    }
)
MARKER_KINDS = frozenset({"volume", "slice"})
_TIMING_KEYS = frozenset(
    {
        "marker_type",
        "marker_description",
        "marker_kind",
        "groups_per_volume",
        "expected_repetition_time_seconds",
        "missing_volume_markers",
        "expected_volume_count",
        "volume_marker_start_index",
        "volume_marker_count",
    }
)
_OPTIONAL_TIMING_KEYS = frozenset(
    {
        "marker_kind",
        "groups_per_volume",
        "expected_repetition_time_seconds",
        "missing_volume_markers",
        "expected_volume_count",
        "volume_marker_start_index",
        "volume_marker_count",
    }
)
_MISSING_VOLUME_MARKER_POLICIES = frozenset({"error", "repair"})
_QUALITY_CONTROL_KEYS = frozenset(
    {
        "block_seconds",
        "mains_frequency_hz",
        "mains_exclusion_hz",
        "residual_mad_multiplier",
        "residual_minimum_channels",
        "volume_spectrum_max_hz",
        "report_channel_outliers",
        "bad_channel_residual_uv",
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
        "residual_obs_section_seconds",
        "pre_trigger_fraction",
        "adaptive_noise_cancellation",
        "adaptive_window",
        "channel_adaptive_window",
        "local_neighbor_count",
        "local_window_channels",
        "residual_gate_mad_multiplier",
        "residual_gate_ratio",
        "residual_gate_max_fraction",
        "adaptive_improvement_ratio",
        "channel_failure_policy",
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
        "residual_obs_section_seconds",
        "pre_trigger_fraction",
        "adaptive_noise_cancellation",
        "adaptive_window",
        "channel_adaptive_window",
        "local_neighbor_count",
        "local_window_channels",
        "residual_gate_mad_multiplier",
        "residual_gate_ratio",
        "residual_gate_max_fraction",
        "adaptive_improvement_ratio",
        "channel_failure_policy",
    }
)
_SUPPORTED_METHODS = frozenset({"acquisition_group_fastr"})
_CHANNEL_FAILURE_POLICIES = frozenset(
    {"report", "retry_local_and_recommend_bad"}
)


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

    input_values = _section(
        root,
        "input",
        _INPUT_KEYS,
        optional_keys=_OPTIONAL_INPUT_KEYS,
    )
    output_values = _section(root, "output", _OUTPUT_KEYS)
    timing_values = _section(
        root,
        "timing",
        _TIMING_KEYS,
        optional_keys=_OPTIONAL_TIMING_KEYS,
    )
    processing_values = _section(
        root,
        "processing",
        _PROCESSING_KEYS,
        optional_keys=_OPTIONAL_PROCESSING_KEYS,
    )

    base_directory = config_path.parent
    timing = _timing_config(timing_values)
    fmri_metadata = (
        _path_value(input_values, "fmri_metadata", base_directory)
        if "fmri_metadata" in input_values
        else None
    )
    acquisition = _acquisition_config(root)
    _validate_timing_sources(
        timing,
        fmri_metadata=fmri_metadata,
        acquisition=acquisition,
    )
    trim = _trim_config(root)
    _validate_marker_selection_trim(timing, trim)
    processing = _processing_config(processing_values)
    quality_control = _quality_control_config(root)
    _validate_channel_failure_policy(processing, quality_control)
    return CorrectionConfig(
        input=InputConfig(
            raw_vhdr=_path_value(input_values, "raw_vhdr", base_directory),
            fmri_metadata=fmri_metadata,
        ),
        output=OutputConfig(
            vhdr=_path_value(output_values, "vhdr", base_directory),
        ),
        timing=timing,
        processing=processing,
        trim=trim,
        acquisition=acquisition,
        quality_control=quality_control,
        diagnostics=_diagnostics_config(root),
    )


def _acquisition_config(
    root: Mapping[str, object],
) -> FmriAcquisitionTiming | None:
    """Read slice timing declared inline instead of in a BIDS sidecar.

    The three fields are the BIDS ones, transcribed, so a recording whose
    sidecar omits ``SliceTiming`` or ``MultibandAccelerationFactor`` can be
    corrected without hand-editing a JSON file. They go through the same
    validation as the sidecar, so an inline declaration is not a weaker one.
    """
    if "acquisition" not in root:
        return None
    values = _require_mapping(root["acquisition"], "acquisition")
    _reject_unknown_keys(values, _ACQUISITION_KEYS, "acquisition")
    for name in sorted(_ACQUISITION_KEYS):
        if name not in values:
            raise ConfigurationError(f"missing required field: acquisition.{name}")
    slice_timing = values["slice_timing_seconds"]
    if not isinstance(slice_timing, list) or not slice_timing:
        raise ConfigurationError(
            "acquisition.slice_timing_seconds must be a nonempty list of "
            "offsets in seconds, one per slice"
        )
    try:
        return FmriAcquisitionTiming(
            repetition_time_seconds=values["repetition_time_seconds"],
            slice_timing_seconds=tuple(slice_timing),
            multiband_acceleration_factor=values["multiband_acceleration_factor"],
        )
    except FastrInputError as error:
        raise ConfigurationError(f"invalid acquisition section: {error}") from error


def _validate_timing_sources(
    timing: TimingConfig,
    *,
    fmri_metadata: Path | None,
    acquisition: FmriAcquisitionTiming | None,
) -> None:
    """Require exactly one source of truth for where acquisition groups fire.

    Volume markers locate only volumes, so the offsets inside them have to be
    declared. Acquisition-group markers record those offsets already, and a
    second declaration could only contradict what the recording says.
    """
    if fmri_metadata is not None and acquisition is not None:
        raise ConfigurationError(
            "declare the acquisition timing once: set either "
            "input.fmri_metadata or the acquisition section, not both"
        )
    declared = fmri_metadata is not None or acquisition is not None
    if timing.marker_kind == "volume" and not declared:
        raise ConfigurationError(
            "volume markers need declared slice timing: set "
            "input.fmri_metadata to a BIDS sidecar, or fill in the acquisition "
            "section"
        )
    if timing.marker_kind == "slice" and declared:
        raise ConfigurationError(
            "acquisition-group markers record their own timing: remove "
            "input.fmri_metadata and the acquisition section, or set "
            "timing.marker_kind to 'volume'"
        )


def _validate_channel_failure_policy(
    processing: ProcessingConfig,
    quality_control: QualityControlConfig,
) -> None:
    """Require the retry policy to own the local window it decides to install.

    The policy earns its answer by comparing one wide correction against one
    local retry of the same channel. Another mode that already moved some
    channels to the local window would make that comparison meaningless, and a
    local window no narrower than the wide one would make it empty.
    """
    active = processing.channel_failure_policy == "retry_local_and_recommend_bad"
    if not active:
        return
    conflicting = (
        processing.adaptive_window
        or processing.channel_adaptive_window
        or bool(processing.local_window_channels)
    )
    if conflicting:
        raise ConfigurationError(
            "processing.channel_failure_policy cannot be combined with "
            "adaptive or explicit local-window modes"
        )
    if not quality_control.report_channel_outliers:
        raise ConfigurationError(
            "retry_local_and_recommend_bad requires "
            "quality_control.report_channel_outliers"
        )
    if processing.local_neighbor_count >= processing.neighbor_count:
        raise ConfigurationError(
            "processing.local_neighbor_count must be smaller than "
            "processing.neighbor_count when retrying failed channels"
        )


def _validate_marker_selection_trim(
    timing: TimingConfig,
    trim: TrimConfig,
) -> None:
    if (
        timing.volume_marker_start_index is not None
        and trim.mode != "first_to_last_volume"
    ):
        raise ConfigurationError(
            "explicit volume marker selection requires "
            "trim.mode 'first_to_last_volume' so unmatched acquisitions are "
            "excluded from the output"
        )


def _timing_config(values: Mapping[str, object]) -> TimingConfig:
    marker_kind = (
        _string_value(values, "marker_kind")
        if "marker_kind" in values
        else "volume"
    )
    if marker_kind not in MARKER_KINDS:
        raise ConfigurationError(
            f"timing.marker_kind must be one of {sorted(MARKER_KINDS)}"
        )

    groups_per_volume = _optional_integer_or_none(values, "groups_per_volume")
    if marker_kind == "slice" and groups_per_volume is None:
        raise ConfigurationError(
            "timing.groups_per_volume is required when timing.marker_kind is "
            "'slice': the markers say when each acquisition group fired, not "
            "how many of them make a volume"
        )
    if marker_kind == "volume" and groups_per_volume is not None:
        raise ConfigurationError(
            "timing.groups_per_volume is only valid when timing.marker_kind is "
            "'slice'; with volume markers it comes from the declared slice "
            "timing"
        )

    expected_repetition_time = _optional_positive_number_or_none(
        values,
        "expected_repetition_time_seconds",
    )
    if marker_kind == "volume" and expected_repetition_time is not None:
        raise ConfigurationError(
            "timing.expected_repetition_time_seconds is only valid when "
            "timing.marker_kind is 'slice'; with volume markers the repetition "
            "time is declared, not measured"
        )

    policy = (
        _string_value(values, "missing_volume_markers")
        if "missing_volume_markers" in values
        else "error"
    )
    if policy not in _MISSING_VOLUME_MARKER_POLICIES:
        raise ConfigurationError(
            "timing.missing_volume_markers must be one of "
            f"{sorted(_MISSING_VOLUME_MARKER_POLICIES)}"
        )
    if marker_kind == "slice" and policy != "error":
        raise ConfigurationError(
            "timing.missing_volume_markers must be 'error' when "
            "timing.marker_kind is 'slice': a missing acquisition-group marker "
            "shifts every later volume boundary and cannot be located from the "
            "marker series"
        )

    expected_count = _optional_integer_or_none(
        values,
        "expected_volume_count",
    )
    if policy == "repair" and expected_count is None:
        raise ConfigurationError(
            "timing.expected_volume_count is required when missing volume "
            "markers are repaired"
        )
    if policy == "error" and expected_count is not None:
        raise ConfigurationError(
            "timing.expected_volume_count is only valid when "
            "missing_volume_markers is 'repair'"
        )

    marker_start = (
        _integer_value(values, "volume_marker_start_index", minimum=0)
        if "volume_marker_start_index" in values
        else None
    )
    marker_count = _optional_integer_or_none(values, "volume_marker_count")
    if (marker_start is None) != (marker_count is None):
        raise ConfigurationError(
            "timing.volume_marker_start_index and timing.volume_marker_count "
            "must be configured together"
        )
    if marker_kind == "slice" and marker_start is not None:
        raise ConfigurationError(
            "timing.volume_marker_start_index and timing.volume_marker_count "
            "are only valid for volume markers"
        )
    if marker_start is not None and policy == "repair":
        raise ConfigurationError(
            "explicit volume marker selection cannot be combined with "
            "missing volume marker repair"
        )

    return TimingConfig(
        marker_type=_string_value(values, "marker_type"),
        marker_description=_string_value(values, "marker_description"),
        marker_kind=marker_kind,
        groups_per_volume=groups_per_volume,
        expected_repetition_time_seconds=expected_repetition_time,
        missing_volume_markers=policy,
        expected_volume_count=expected_count,
        volume_marker_start_index=marker_start,
        volume_marker_count=marker_count,
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
        volume_spectrum_max_hz=_optional_finite_number(
            values,
            "volume_spectrum_max_hz",
            default=QualityControlConfig.__dataclass_fields__[
                "volume_spectrum_max_hz"
            ].default,
            minimum=0.0,
        ),
        report_channel_outliers=(
            _boolean_value(values, "report_channel_outliers")
            if "report_channel_outliers" in values
            else QualityControlConfig.__dataclass_fields__[
                "report_channel_outliers"
            ].default
        ),
        bad_channel_residual_uv=_optional_finite_number(
            values,
            "bad_channel_residual_uv",
            default=QualityControlConfig.__dataclass_fields__[
                "bad_channel_residual_uv"
            ].default,
            minimum=0.0,
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
    residual_obs_rank = _optional_obs_rank(
        values,
        "residual_obs_rank",
        default=ProcessingConfig.__dataclass_fields__["residual_obs_rank"].default,
    )
    residual_obs_section_seconds = _optional_positive_number_or_none(
        values,
        "residual_obs_section_seconds",
    )
    pre_trigger_fraction = _optional_finite_number(
        values,
        "pre_trigger_fraction",
        default=0.03,
        minimum=0.0,
        inclusive=True,
        maximum=1.0,
    )
    lowpass_hz = _finite_number(values, "lowpass_hz", minimum=0.0, inclusive=True)
    adaptive_noise_cancellation = (
        _boolean_value(values, "adaptive_noise_cancellation")
        if "adaptive_noise_cancellation" in values
        else False
    )
    if adaptive_noise_cancellation and lowpass_hz == 0.0:
        # fmrib_fastr.m forces a 70 Hz cutoff in this case. Silently replacing a
        # cutoff the configuration states is worse than refusing the pair: the
        # canceller's reference has to be band-limited to mean anything.
        raise ConfigurationError(
            "processing.adaptive_noise_cancellation requires a nonzero "
            "processing.lowpass_hz, because the canceller's reference must be "
            "band-limited"
        )

    adaptive_window = (
        _boolean_value(values, "adaptive_window")
        if "adaptive_window" in values
        else ProcessingConfig.__dataclass_fields__["adaptive_window"].default
    )
    channel_adaptive_window = (
        _boolean_value(values, "channel_adaptive_window")
        if "channel_adaptive_window" in values
        else ProcessingConfig.__dataclass_fields__[
            "channel_adaptive_window"
        ].default
    )
    if adaptive_window and channel_adaptive_window:
        raise ConfigurationError(
            "processing.adaptive_window and "
            "processing.channel_adaptive_window cannot both be enabled"
        )
    local_window_channels = _channel_names(
        values,
        "local_window_channels",
        default=(),
    )
    if local_window_channels and (adaptive_window or channel_adaptive_window):
        raise ConfigurationError(
            "processing.local_window_channels cannot be combined with "
            "adaptive window modes"
        )
    non_eeg_channels = _non_eeg_channels(values)
    overlap = set(local_window_channels) & set(non_eeg_channels)
    if overlap:
        channels = ", ".join(sorted(overlap))
        raise ConfigurationError(
            "processing.local_window_channels cannot contain non-EEG "
            f"channels: {channels}"
        )
    local_neighbor_count = (
        _integer_value(values, "local_neighbor_count", minimum=2)
        if "local_neighbor_count" in values
        else ProcessingConfig.__dataclass_fields__["local_neighbor_count"].default
    )
    if local_neighbor_count % 2:
        raise ConfigurationError("processing.local_neighbor_count must be even")
    if (
        adaptive_window or channel_adaptive_window or local_window_channels
    ) and local_neighbor_count >= neighbor_count:
        raise ConfigurationError(
            "processing.local_neighbor_count must be smaller than "
            "processing.neighbor_count when a local-window mode is enabled"
        )

    channel_failure_policy = (
        _string_value(values, "channel_failure_policy")
        if "channel_failure_policy" in values
        else ProcessingConfig.__dataclass_fields__[
            "channel_failure_policy"
        ].default
    )
    if channel_failure_policy not in _CHANNEL_FAILURE_POLICIES:
        supported = ", ".join(sorted(_CHANNEL_FAILURE_POLICIES))
        raise ConfigurationError(
            f"processing.channel_failure_policy must be one of {supported}; "
            f"got {channel_failure_policy!r}"
        )

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
        residual_obs_section_seconds=residual_obs_section_seconds,
        pre_trigger_fraction=pre_trigger_fraction,
        adaptive_noise_cancellation=adaptive_noise_cancellation,
        adaptive_window=adaptive_window,
        channel_adaptive_window=channel_adaptive_window,
        local_neighbor_count=local_neighbor_count,
        local_window_channels=local_window_channels,
        residual_gate_mad_multiplier=residual_gate_mad_multiplier,
        residual_gate_ratio=residual_gate_ratio,
        residual_gate_max_fraction=residual_gate_max_fraction,
        adaptive_improvement_ratio=adaptive_improvement_ratio,
        channel_failure_policy=channel_failure_policy,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        template_high_pass_hz=template_high_pass_hz,
        search_radius_samples=_integer_value(
            values,
            "search_radius_samples",
            minimum=0,
        ),
        lowpass_hz=lowpass_hz,
        output_sampling_rate_hz=output_sampling_rate_hz,
        channel_batch_size=_integer_value(
            values,
            "channel_batch_size",
            minimum=1,
        ),
        reference_channel=_reference_channel(values),
        line_noise_frequencies_hz=line_noise_frequencies_hz,
        non_eeg_channels=non_eeg_channels,
    )


def _non_eeg_channels(values: Mapping[str, object]) -> tuple[str, ...]:
    """Names of channels the correction must not fit a scalar or a basis to."""
    default = ProcessingConfig.__dataclass_fields__["non_eeg_channels"].default
    return _channel_names(values, "non_eeg_channels", default=default)


def _channel_names(
    values: Mapping[str, object],
    name: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if name not in values:
        return default
    names = values[name]
    if not isinstance(names, list) or any(
        not isinstance(channel_name, str) or not channel_name
        for channel_name in names
    ):
        raise ConfigurationError(
            f"processing.{name} must be a list of nonempty channel names"
        )
    if len(names) != len(set(names)):
        raise ConfigurationError(
            f"processing.{name} must not contain duplicate channel names"
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


def _optional_obs_rank(
    values: Mapping[str, object],
    name: str,
    *,
    default: int | str,
) -> int | str:
    if name not in values:
        return default
    value = values[name]
    if value == "auto":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(
            f"{name} must be a positive integer or 'auto'"
        )
    return value


def _optional_positive_number_or_none(
    values: Mapping[str, object],
    name: str,
) -> float | None:
    if name not in values or values[name] is None:
        return None
    return _finite_number(values, name, minimum=0.0)


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
