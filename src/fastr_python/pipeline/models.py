"""Data models for the configuration-driven correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..quality.residuals import LocalRetryEvaluation


class PipelineInputError(ValueError):
    """Raised when a configured correction run cannot be performed."""


@dataclass(frozen=True, slots=True)
class CorrectionSummary:
    """Stable summary of one completed correction run."""

    output_vhdr: Path
    output_eeg: Path
    output_vmrk: Path
    provenance_json: Path
    psd_before: Path
    psd_after: Path
    method: str
    input_sampling_rate_hz: float
    output_sampling_rate_hz: float
    channel_count: int
    input_sample_count: int
    output_sample_count: int
    marker_count: int
    processed_group_count: int
    skipped_group_count: int


@dataclass(frozen=True, slots=True)
class ChannelFailurePolicyResult:
    """Every automatic channel decision one run made, keyed by channel index.

    The pipeline decides and the provenance module serializes, so this record
    lives between them rather than inside either. It is a record and not an
    instruction: nothing downstream of it removes, replaces or interpolates a
    channel, and ``recommended_bad_channels`` reaches the JSON sidecar and
    stops there.
    """

    candidate_blocks_by_channel: dict[int, np.ndarray]
    retry_evaluations: dict[int, LocalRetryEvaluation]
    accepted_channels: frozenset[int]
    final_failed_blocks_by_channel: dict[int, np.ndarray]
    recommended_bad_channels: frozenset[int]

    @classmethod
    def inactive(cls) -> ChannelFailurePolicyResult:
        """Build the result for a run that never nominated a channel."""
        return cls(
            candidate_blocks_by_channel={},
            retry_evaluations={},
            accepted_channels=frozenset(),
            final_failed_blocks_by_channel={},
            recommended_bad_channels=frozenset(),
        )
