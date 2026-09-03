"""Decode validated configuration sections."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from ..residual_qc import ResidualQcDefaults
from .models import (
    ConfigurationError,
    DiagnosticsConfig,
    ProcessingConfig,
    QualityControlConfig,
    TimingConfig,
    TrimConfig,
)
from .schema import (
    _boolean_value,
    _finite_number,
    _integer_value,
    _optional_finite_number,
    _optional_integer_or_none,
    _optional_obs_rank,
    _optional_positive_integer,
    _optional_positive_number_or_none,
    _optional_section,
    _reject_unknown_keys,
    _require_mapping,
    _string_value,
)

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

def _timing_marker_settings(
    values: Mapping[str, object],
) -> tuple[str, int | None, float | None]:
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
    return marker_kind, groups_per_volume, expected_repetition_time


def _timing_missing_marker_settings(
    values: Mapping[str, object],
    marker_kind: str,
) -> tuple[str, int | None]:
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
    return policy, expected_count


def _timing_volume_selection(
    values: Mapping[str, object],
    *,
    marker_kind: str,
    policy: str,
) -> tuple[int | None, int | None]:
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
    return marker_start, marker_count


def _timing_config(values: Mapping[str, object]) -> TimingConfig:
    marker_kind, groups_per_volume, expected_repetition_time = (
        _timing_marker_settings(values)
    )
    policy, expected_count = _timing_missing_marker_settings(
        values,
        marker_kind,
    )
    marker_start, marker_count = _timing_volume_selection(
        values,
        marker_kind=marker_kind,
        policy=policy,
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
            default=ResidualQcDefaults.MAD_MULTIPLIER,
            minimum=0.0,
            inclusive=True,
        ),
        residual_minimum_channels=_optional_positive_integer(
            values,
            "residual_minimum_channels",
            default=ResidualQcDefaults.MINIMUM_CHANNELS,
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
