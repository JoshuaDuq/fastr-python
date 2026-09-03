"""Marker transformations and annotations for pipeline outputs."""

from __future__ import annotations

import math

import numpy as np

from ..fastr import FastrGeometry
from ..io.brainvision import BrainVisionMarker
from ..window import OutputWindow
from .models import PipelineInputError


def validate_marker_output_positions(
    markers: tuple[BrainVisionMarker, ...],
    output_sample_count: int,
) -> None:
    """Validate that resampled markers fit inside the output recording."""
    if any(marker.position > output_sample_count for marker in markers):
        raise PipelineInputError(
            "resampled marker positions extend beyond the output recording"
        )


def residual_qc_markers(
    residual_qc: dict[str, object],
    *,
    output_rate: float,
    output_sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    """Annotate blocks carrying an unusually high residual, advisorily.

    These are deliberately not "Bad Interval" markers. MNE rejects any
    annotation whose "type/description" begins with "bad", so labelling a
    soft quality signal that way silently drops the affected epochs from
    every downstream analysis. A block flagged here is worth looking at; it
    is not established as unusable. Uncorrected spans, which genuinely do
    still carry raw artifact, keep the Bad Interval marker below.
    """
    flagged = np.asarray(residual_qc["flagged_blocks"], dtype=bool)
    if flagged.size == 0:
        return ()
    block_samples = round(float(residual_qc["block_seconds"]) * output_rate)
    markers = []
    for block in np.flatnonzero(flagged):
        position = int(block) * block_samples + 1
        if position > output_sample_count:
            continue
        size = min(block_samples, output_sample_count - position + 1)
        markers.append(
            BrainVisionMarker(
                marker_type="Comment",
                description="QC_ResidualHigh",
                position=position,
                size=int(max(size, 1)),
                channel=0,
            )
        )
    return tuple(markers)


def skipped_group_spans(
    group_triggers: np.ndarray,
    geometry: FastrGeometry,
) -> tuple[tuple[int, int], ...]:
    """Group the uncorrected acquisition groups into contiguous input-sample spans."""
    skipped = np.asarray(geometry.skipped_group_indices, dtype=np.int64)
    if skipped.size == 0:
        return ()
    factor = geometry.interpolation_factor
    before = math.ceil(geometry.epoch.samples_before / factor)
    after = math.ceil(geometry.epoch.samples_after / factor)
    last_index = group_triggers.size - 1

    breaks = np.flatnonzero(np.diff(skipped) != 1)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks, [skipped.size - 1]))

    spans = []
    for start, stop in zip(starts, stops, strict=True):
        first_group = int(skipped[start])
        last_group = int(skipped[stop])
        first_sample = int(group_triggers[first_group]) - before
        if last_group < last_index:
            last_sample = int(group_triggers[last_group + 1]) - 1
        else:
            last_sample = int(group_triggers[last_group]) + after
        spans.append((max(first_sample, 0), last_sample))
    return tuple(spans)


def bad_gradient_markers(
    spans: tuple[tuple[int, int], ...],
    *,
    window: OutputWindow,
    decimation: int,
    output_sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    """Annotate every emitted span the correction left untouched.

    Without these a corrected file gives no sign that part of it still carries
    the raw gradient artifact.
    """
    markers = []
    for first_sample, last_sample in spans:
        first = max(first_sample, window.start)
        last = min(last_sample, window.stop - 1)
        if last < first:
            continue
        start = (first - window.start) // decimation + 1
        stop = (last - window.start) // decimation + 1
        if start > output_sample_count:
            continue
        stop = min(stop, output_sample_count)
        markers.append(
            BrainVisionMarker(
                marker_type="Bad Interval",
                description="Bad_Gradient",
                position=int(start),
                size=int(max(stop - start + 1, 1)),
                channel=0,
            )
        )
    return tuple(markers)
