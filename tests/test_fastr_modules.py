from eegfmri_fastr import fastr
from eegfmri_fastr.fastr_geometry import (
    adapt_fastr_geometry,
    gate_fastr_geometry,
    prepare_fastr_geometry,
)
from eegfmri_fastr.fastr_processing import (
    apply_fastr_batch,
    fit_fastr_alignment,
    residual_obs,
)
from eegfmri_fastr.fastr_timing import (
    FmriAcquisitionTiming,
    load_bids_fmri_timing,
    make_group_trigger_samples,
)
from eegfmri_fastr.fastr_types import (
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
        "AcquisitionGeometry",
        "AncCorrection",
        "FastrAlignment",
        "FastrCorrection",
        "FastrGeometry",
        "FastrInputError",
        "FastrProvenance",
        "FmriAcquisitionTiming",
        "ResidualObsCorrection",
        "acquisition_group_fastr",
        "acquisition_group_fastr_with_edges",
        "adapt_fastr_geometry",
        "adaptive_noise_cancel",
        "apply_fastr_batch",
        "fit_fastr_alignment",
        "fit_residual_obs",
        "fmrib_lms",
        "gate_fastr_geometry",
        "load_bids_fmri_timing",
        "make_group_trigger_samples",
        "obs_trigger_subset",
        "prepare_fastr_geometry",
        "repair_volume_starts",
        "residual_obs",
        "select_obs_rank",
        "slice_fastr",
        "slice_fastr_with_edges",
        "slice_marker_geometry",
        "volume_marker_geometry",
    ]
