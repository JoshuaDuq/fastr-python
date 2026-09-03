"""Immutable correction configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..correction.timing import FmriAcquisitionTiming
from ..quality.residuals import ResidualQcDefaults

_TRIM_MODES = frozenset({"none", "first_to_last_volume"})
DEFAULT_NON_EEG_CHANNELS = ("ECG",)
DEFAULT_TEMPLATE_HIGH_PASS_HZ = 1.0
DEFAULT_RESIDUAL_THRESHOLD_UV = 1.0
DEFAULT_RESIDUAL_OBS_RANK = 4
DEFAULT_LOCAL_NEIGHBOR_COUNT = 20
DEFAULT_CHANNEL_FAILURE_POLICY: Literal["report"] = "report"
DEFAULT_VOLUME_SPECTRUM_MAX_HZ = 110.0
DEFAULT_REPORT_CHANNEL_OUTLIERS = True
DEFAULT_BAD_CHANNEL_RESIDUAL_UV = 5.0


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
    non_eeg_channels: tuple[str, ...] = DEFAULT_NON_EEG_CHANNELS
    template_high_pass_hz: float = DEFAULT_TEMPLATE_HIGH_PASS_HZ
    residual_threshold_uv: float = DEFAULT_RESIDUAL_THRESHOLD_UV
    residual_gate: bool = False
    residual_obs: bool = False
    residual_obs_rank: int | Literal["auto"] = DEFAULT_RESIDUAL_OBS_RANK
    residual_obs_section_seconds: float | None = None
    pre_trigger_fraction: float = 0.03
    adaptive_noise_cancellation: bool = False
    adaptive_window: bool = False
    channel_adaptive_window: bool = False
    local_neighbor_count: int = DEFAULT_LOCAL_NEIGHBOR_COUNT
    local_window_channels: tuple[str, ...] = ()
    residual_gate_mad_multiplier: float = 8.0
    residual_gate_ratio: float = 8.0
    residual_gate_max_fraction: float = 0.02
    adaptive_improvement_ratio: float = 0.85
    channel_failure_policy: Literal["report", "retry_local_and_recommend_bad"] = (
        DEFAULT_CHANNEL_FAILURE_POLICY
    )


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """How the pipeline restricts its output to the scanning period."""

    mode: str = "none"

    def __post_init__(self) -> None:
        if self.mode not in _TRIM_MODES:
            raise ConfigurationError(f"trim.mode must be one of {sorted(_TRIM_MODES)}")


@dataclass(frozen=True, slots=True)
class QualityControlConfig:
    """Settings for residual-gradient measurements and annotations."""

    block_seconds: float = 30.0
    mains_frequency_hz: float = 60.0
    mains_exclusion_hz: float = 1.0
    # How far above a channel's own median residual a block must sit, and on
    # how many channels at once, before it is annotated. See residual_qc.
    residual_mad_multiplier: float = ResidualQcDefaults.MAD_MULTIPLIER
    residual_minimum_channels: int = ResidualQcDefaults.MINIMUM_CHANNELS
    # Highest volume harmonic reported, capped at the output Nyquist frequency.
    volume_spectrum_max_hz: float = DEFAULT_VOLUME_SPECTRUM_MAX_HZ
    report_channel_outliers: bool = DEFAULT_REPORT_CHANNEL_OUTLIERS
    # Absolute floor a channel-block residual must clear before the automatic
    # policy may call it a failure. Unlike residual_threshold_uv this one is
    # never relative: a spatial comparison alone cannot separate a broken
    # electrode from a quiet recording. See residual_qc.
    bad_channel_residual_uv: float = DEFAULT_BAD_CHANNEL_RESIDUAL_UV


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
