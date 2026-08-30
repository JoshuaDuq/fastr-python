"""Public FASTR correction API."""

from __future__ import annotations

import numpy.typing as npt

from .fastr_geometry import (
    adapt_fastr_geometry,
    gate_fastr_geometry,
    prepare_fastr_geometry,
)
from .fastr_processing import (
    _run_fastr,
    _run_fastr_with_edges,
    apply_fastr_batch,
    fit_fastr_alignment,
    obs_trigger_subset,
    residual_obs,
)
from .fastr_timing import (
    FmriAcquisitionTiming,
    load_bids_fmri_timing,
    make_group_trigger_samples,
    repair_volume_starts,
)
from .fastr_types import (
    FastrAlignment,
    FastrCorrection,
    FastrGeometry,
    FastrInputError,
    FastrProvenance,
)

__all__ = [
    "FastrAlignment",
    "FastrCorrection",
    "FastrGeometry",
    "FastrInputError",
    "FastrProvenance",
    "FmriAcquisitionTiming",
    "acquisition_group_fastr",
    "acquisition_group_fastr_with_edges",
    "adapt_fastr_geometry",
    "apply_fastr_batch",
    "fit_fastr_alignment",
    "gate_fastr_geometry",
    "load_bids_fmri_timing",
    "make_group_trigger_samples",
    "obs_trigger_subset",
    "prepare_fastr_geometry",
    "repair_volume_starts",
    "residual_obs",
    "slice_fastr",
    "slice_fastr_with_edges",
]


def slice_fastr(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
    pre_trigger_fraction: float = 0.03,
) -> FastrCorrection:
    """Subtract target-excluding alternating FASTR templates.

    This is the classical alternating slice-trigger variant. For multiband data,
    use :func:`acquisition_group_fastr`, which matches repeated acquisition-time
    slots instead of treating adjacent groups as interchangeable.
    """
    return _run_fastr(
        data,
        group_triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=None,
        pre_trigger_fraction=pre_trigger_fraction,
    )


def acquisition_group_fastr(
    data: npt.ArrayLike,
    volume_starts: npt.ArrayLike,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
    interpolation_factor: int = 10,
    neighbor_count: int = 20,
    search_radius_samples: int = 3,
    pre_trigger_fraction: float = 0.03,
) -> FastrCorrection:
    """Correct repeated multiband acquisition-time slots with FASTR fitting.

    ``volume_starts`` are zero-based sample positions in ``data``. The group
    triggers are derived from the validated BIDS timing, so the slot-matching
    geometry cannot be accidentally paired with a different acquisition layout.
    """
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    return _run_fastr(
        data,
        triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=timing.groups_per_volume,
        pre_trigger_fraction=pre_trigger_fraction,
    )


def slice_fastr_with_edges(
    data: npt.ArrayLike,
    group_triggers: npt.ArrayLike,
    *,
    interpolation_factor: int = 10,
    neighbor_count: int = 30,
    search_radius_samples: int = 3,
    pre_trigger_fraction: float = 0.03,
) -> FastrCorrection:
    """Correct estimable groups and report boundary groups left untouched.

    The strict :func:`slice_fastr` core rejects incomplete epochs. This explicit
    wrapper leaves groups whose search windows exceed the recording untouched and
    records their original indices in provenance.
    """
    return _run_fastr_with_edges(
        data,
        group_triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=None,
        pre_trigger_fraction=pre_trigger_fraction,
    )


def acquisition_group_fastr_with_edges(
    data: npt.ArrayLike,
    volume_starts: npt.ArrayLike,
    *,
    sampling_rate: float,
    timing: FmriAcquisitionTiming,
    interpolation_factor: int = 10,
    neighbor_count: int = 20,
    search_radius_samples: int = 3,
    pre_trigger_fraction: float = 0.03,
) -> FastrCorrection:
    """Correct estimable complete volumes and report skipped boundary volumes."""
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=sampling_rate,
        timing=timing,
    )
    return _run_fastr_with_edges(
        data,
        triggers,
        interpolation_factor=interpolation_factor,
        neighbor_count=neighbor_count,
        search_radius_samples=search_radius_samples,
        groups_per_volume=timing.groups_per_volume,
        pre_trigger_fraction=pre_trigger_fraction,
    )
