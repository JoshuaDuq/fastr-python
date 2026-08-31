"""Immutable FASTR data types and domain errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FastrInputError(ValueError):
    """Raised when FASTR acquisition metadata or triggers are invalid."""


@dataclass(frozen=True, slots=True)
class _TemplateWindow:
    """Which epochs form each target's template, and how to sum them cheaply.

    The chosen epochs always form one run of a residue class of the group index,
    so a running total over that class gives every template in one pass.
    """

    indices: np.ndarray
    stride: int
    run_starts: np.ndarray
    run_length: int
    contains_target: bool
    summed_contiguous: bool = True

    def __post_init__(self) -> None:
        for field_name in ("indices", "run_starts"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True)
class _ArtifactEpoch:
    """Extent of one artifact epoch around its trigger, in interpolated samples."""

    samples_before: int
    samples_after: int
    slack: int

    @property
    def length(self) -> int:
        return self.samples_before + self.samples_after + 1

    @property
    def residual_samples_after(self) -> int:
        """FASTR fits residual components over the longest possible artifact."""
        return self.samples_after + self.slack


@dataclass(frozen=True, slots=True, eq=False)
class FastrProvenance:
    """Complete record of how each acquisition group was corrected."""

    interpolation_factor: int
    samples_before_trigger: int
    samples_after_trigger: int
    search_radius: int
    neighbor_indices: np.ndarray
    shifts: np.ndarray
    correlations: np.ndarray
    amplitudes: np.ndarray
    skipped_group_indices: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "neighbor_indices",
            "shifts",
            "correlations",
            "amplitudes",
            "skipped_group_indices",
        ):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class FastrCorrection:
    """A corrected recording on the original sample grid and its provenance."""

    data: np.ndarray
    provenance: FastrProvenance


@dataclass(frozen=True, slots=True, eq=False)
class ChannelAdaptiveFastrCorrection:
    """Per-channel adaptive correction and its channel-specific decisions."""

    data: np.ndarray
    amplitudes: np.ndarray
    adapted_group_indices: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        for field_name in ("data", "amplitudes"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)
        indices = tuple(
            np.array(channel_indices, copy=True)
            for channel_indices in self.adapted_group_indices
        )
        for channel_indices in indices:
            channel_indices.setflags(write=False)
        object.__setattr__(self, "adapted_group_indices", indices)


@dataclass(frozen=True, slots=True, eq=False)
class ResidualObsCorrection:
    """Residual-OBS output and the rank fitted in each channel section."""

    data: np.ndarray
    selected_ranks: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("data", "selected_ranks"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class AncCorrection:
    """Adaptive-noise-cancellation output and per-channel diagnostics."""

    data: np.ndarray
    reference_scales: np.ndarray
    step_sizes: np.ndarray
    filter_order: int

    def __post_init__(self) -> None:
        for field_name in ("data", "reference_scales", "step_sizes"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class FastrGeometry:
    """Validated acquisition geometry shared by all channel batches."""

    triggers: np.ndarray
    fine_triggers: np.ndarray
    epoch: _ArtifactEpoch
    pre_trigger_fraction: float
    window: _TemplateWindow
    interpolation_factor: int
    interpolation_taps: np.ndarray
    search_radius: int
    group_indices: np.ndarray
    skipped_group_indices: np.ndarray
    sample_count: int
    excluded_group_indices: np.ndarray
    adapted_group_indices: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "triggers",
            "fine_triggers",
            "interpolation_taps",
            "group_indices",
            "skipped_group_indices",
            "excluded_group_indices",
            "adapted_group_indices",
        ):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True, slots=True, eq=False)
class FastrAlignment:
    """Group alignment fitted once and reusable across channel batches."""

    shifts: np.ndarray
    correlations: np.ndarray
    fitted_triggers: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("shifts", "correlations", "fitted_triggers"):
            values = np.array(getattr(self, field_name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)
