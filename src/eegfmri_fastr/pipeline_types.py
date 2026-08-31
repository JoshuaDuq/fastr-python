"""Shared types for the configuration-driven correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .residual_qc import LocalRetryEvaluation


class PipelineInputError(ValueError):
    """Raised when a configured correction run cannot be performed."""


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
        """The result of a run that never nominated a channel."""
        return cls(
            candidate_blocks_by_channel={},
            retry_evaluations={},
            accepted_channels=frozenset(),
            final_failed_blocks_by_channel={},
            recommended_bad_channels=frozenset(),
        )
