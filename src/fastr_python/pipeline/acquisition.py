"""Resolve configured acquisition timing and marker geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import CorrectionConfig
from ..fastr import (
    AcquisitionGeometry,
    FmriAcquisitionTiming,
    load_bids_fmri_timing,
    repair_volume_starts,
    slice_marker_geometry,
    volume_marker_geometry,
)


@dataclass(frozen=True, slots=True)
class _ResolvedAcquisition:
    """Acquisition geometry, plus what the run was told and what it found.

    ``declared_timing`` is absent when acquisition-group markers supplied the
    geometry, and ``detected_volume_count`` is the count before any volume
    marker repair, so the sidecar can report both numbers.
    """

    geometry: AcquisitionGeometry
    declared_timing: FmriAcquisitionTiming | None
    detected_volume_count: int


def _resolve_acquisition(
    config: CorrectionConfig,
    marker_samples: np.ndarray,
    *,
    sampling_rate: float,
) -> _ResolvedAcquisition:
    """Resolve where every acquisition group fires from the configured markers.

    Volume markers are expanded with declared slice timing, optionally after
    repairing interior gaps; acquisition-group markers are measured where they
    were recorded. Both paths end at the same geometry, so only this function
    has to know which convention the recording used.
    """
    if config.timing.marker_kind == "slice":
        geometry = slice_marker_geometry(
            marker_samples,
            sampling_rate=sampling_rate,
            groups_per_volume=config.timing.groups_per_volume,
            expected_repetition_time_seconds=(
                config.timing.expected_repetition_time_seconds
            ),
        )
        return _ResolvedAcquisition(
            geometry=geometry,
            declared_timing=None,
            detected_volume_count=geometry.volume_count,
        )

    timing = config.acquisition or load_bids_fmri_timing(config.input.fmri_metadata)
    volume_starts = marker_samples
    detected_volume_count = int(volume_starts.size)
    if config.timing.missing_volume_markers == "repair":
        volume_starts = repair_volume_starts(
            volume_starts,
            samples_per_volume=math.floor(
                timing.repetition_time_seconds * sampling_rate + 0.5
            ),
            expected_volume_count=config.timing.expected_volume_count,
        )
    return _ResolvedAcquisition(
        geometry=volume_marker_geometry(
            volume_starts,
            sampling_rate=sampling_rate,
            timing=timing,
        ),
        declared_timing=timing,
        detected_volume_count=detected_volume_count,
    )
