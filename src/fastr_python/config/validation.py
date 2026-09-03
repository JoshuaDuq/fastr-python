"""Cross-field configuration invariants."""

from __future__ import annotations

from pathlib import Path

from ..correction.timing import FmriAcquisitionTiming
from .models import (
    ConfigurationError,
    ProcessingConfig,
    QualityControlConfig,
    TimingConfig,
    TrimConfig,
)


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
