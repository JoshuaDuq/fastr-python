from mri_correction import fastr
from mri_correction.fastr_geometry import (
    adapt_fastr_geometry,
    gate_fastr_geometry,
    prepare_fastr_geometry,
)
from mri_correction.fastr_processing import (
    apply_fastr_batch,
    fit_fastr_alignment,
    residual_obs,
)
from mri_correction.fastr_timing import (
    FmriAcquisitionTiming,
    load_bids_fmri_timing,
    make_group_trigger_samples,
)
from mri_correction.fastr_types import (
    FastrAlignment,
    FastrCorrection,
    FastrGeometry,
    FastrInputError,
    FastrProvenance,
)


def test_fastr_facade_reexports_extracted_implementations() -> None:
    assert fastr.prepare_fastr_geometry is prepare_fastr_geometry
    assert fastr.gate_fastr_geometry is gate_fastr_geometry
    assert fastr.adapt_fastr_geometry is adapt_fastr_geometry
    assert fastr.apply_fastr_batch is apply_fastr_batch
    assert fastr.fit_fastr_alignment is fit_fastr_alignment
    assert fastr.residual_obs is residual_obs
    assert fastr.FmriAcquisitionTiming is FmriAcquisitionTiming
    assert fastr.load_bids_fmri_timing is load_bids_fmri_timing
    assert fastr.make_group_trigger_samples is make_group_trigger_samples


def test_fastr_facade_reexports_stable_value_types() -> None:
    assert fastr.FastrAlignment is FastrAlignment
    assert fastr.FastrCorrection is FastrCorrection
    assert fastr.FastrGeometry is FastrGeometry
    assert fastr.FastrInputError is FastrInputError
    assert fastr.FastrProvenance is FastrProvenance


def test_fastr_facade_declares_only_public_names() -> None:
    assert fastr.__all__ == [
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
        "prepare_fastr_geometry",
        "residual_obs",
        "slice_fastr",
        "slice_fastr_with_edges",
    ]
