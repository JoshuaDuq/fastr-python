# FASTR Project Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the FASTR-only research package modular and explicitly configurable without changing its numerical behavior, public imports, outputs, or failure policy.

**Architecture:** Keep mri_correction.fastr and mri_correction.pipeline as public facades while moving their implementations into focused timing, geometry, template, processing, I/O, marker, and provenance modules. Extend the existing strict YAML model with optional quality-control and diagnostic settings, and pass every new setting through to the existing algorithms with today’s values as defaults.

**Tech Stack:** Python 3.12, NumPy, SciPy, MNE-Python 1.12.1, PyYAML, pybv, pytest, Ruff, uv.

---

## Scope and invariants

- Work from local main, currently aligned with GitHub’s merge commit f3aec7d.
- Preserve every existing public symbol imported from mri_correction.fastr, mri_correction.pipeline, and the existing utility modules.
- Preserve the current YAML keys and defaults. New quality_control and diagnostics sections and new optional robustness fields are additive.
- Keep the current numerical operations and operation order. Do not change filters, interpolation taps, marker serialization, output naming, or floating-point arithmetic as part of this cleanup.
- Keep explicit errors. Do not add repair, inference, or silent fallback behavior.
- Keep BCG/ECG implementation out of FASTR-Python; it remains in the separate BCG-Correction project.

## File map

### New modules

- src/mri_correction/fastr_types.py — FASTR error, immutable result models, and private geometry value objects.
- src/mri_correction/fastr_timing.py — BIDS timing loading and volume-to-group trigger conversion.
- src/mri_correction/fastr_geometry.py — geometry validation/building, boundary policy, template-window selection, residual gate, and adaptive-window selection.
- src/mri_correction/fastr_templates.py — interpolation, epoch operations, template means, high-pass estimation, alignment correlation, and amplitude fitting.
- src/mri_correction/fastr_processing.py — shared alignment fitting, batch correction, residual OBS, and processing validation.
- src/mri_correction/pipeline_io.py — run input/output validation, rate/reference resolution, and filtering/decimation.
- src/mri_correction/pipeline_markers.py — output marker transforms and bad-gradient annotations.
- src/mri_correction/pipeline_provenance.py — provenance assembly, JSON conversion, and file hashing.

### Modified modules

- src/mri_correction/fastr.py — thin public facade and convenience wrappers.
- src/mri_correction/pipeline.py — orchestration and compatibility wrappers only.
- src/mri_correction/config.py — typed QC/diagnostic settings and explicit defaults.
- src/mri_correction/psd.py — use the caller’s optional FFT size without changing the omitted default.
- src/mri_correction/residual_qc.py — keep existing defaults while accepting explicit pipeline values.
- examples/configuration.yml, README.md, docs/algorithm.md, docs/validation.md — document the configuration surface and module boundary.

### Tests

- Modify tests/test_config.py, tests/test_fastr.py, tests/test_fastr_batch.py, tests/test_pipeline.py, and tests/test_residual_qc.py for new settings and facade behavior.
- Add tests/test_fastr_modules.py for implementation-module import boundaries and public re-exports.
- Add tests/test_pipeline_modules.py for pipeline helper boundaries and compatibility seams.
- Keep the existing synthetic, marker, boundary, QC, and end-to-end tests unchanged unless they need an explicit assertion for a new setting.

---

## Task 1: Add typed optional configuration for QC and diagnostics

**Files:**

- Modify: src/mri_correction/config.py
- Modify: tests/test_config.py
- Modify: examples/configuration.yml

### Step 1: Add failing tests for the new configuration surface

Extend tests/test_config.py with tests that load the existing four-section document and assert today’s defaults:

```
def test_optional_quality_control_defaults_are_explicit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.block_seconds == 30.0
    assert config.quality_control.mains_frequency_hz == 60.0
    assert config.quality_control.mains_exclusion_hz == 1.0


def test_optional_diagnostics_defaults_preserve_current_output(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.diagnostics.psd_max_frequency_hz == 100.0
    assert config.diagnostics.psd_n_fft is None


def test_custom_quality_control_and_diagnostic_values_are_loaded(
    tmp_path: Path,
) -> None:
    document = valid_document() + """
quality_control:
  block_seconds: 15.0
  mains_frequency_hz: 50.0
  mains_exclusion_hz: 0.5
diagnostics:
  psd_max_frequency_hz: 80.0
  psd_n_fft: 4096
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.block_seconds == 15.0
    assert config.quality_control.mains_frequency_hz == 50.0
    assert config.quality_control.mains_exclusion_hz == 0.5
    assert config.diagnostics.psd_max_frequency_hz == 80.0
    assert config.diagnostics.psd_n_fft == 4096
```

Add parameterized invalid-value coverage for non-positive block length, non-positive
mains frequency, negative mains exclusion, non-positive PSD limit, non-positive FFT
size, and non-integer FFT size. Add unknown-key tests for both new sections.

Run:

```
uv run pytest tests/test_config.py -k 'quality_control or diagnostics' -v
```

Expected: FAIL because CorrectionConfig has no new sections.

### Step 2: Implement the typed settings

In src/mri_correction/config.py, add these immutable models before CorrectionConfig:

```
@dataclass(frozen=True, slots=True)
class QualityControlConfig:
    """Settings for residual-gradient measurements and annotations."""

    block_seconds: float = 30.0
    mains_frequency_hz: float = 60.0
    mains_exclusion_hz: float = 1.0


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Settings for generated PSD diagnostics."""

    psd_max_frequency_hz: float = 100.0
    psd_n_fft: int | None = None
```

Append defaulted fields to CorrectionConfig so existing positional construction
remains valid:

```
    quality_control: QualityControlConfig = field(
        default_factory=QualityControlConfig,
    )
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
```

Import field from dataclasses. Add optional section key sets and parsers. Keep
the existing strict required-section parser separate from the optional parser so
missing keys use documented defaults. The parser must perform these exact actions:

- return QualityControlConfig() when the section is absent;
- reject keys outside block_seconds, mains_frequency_hz, and mains_exclusion_hz;
- parse the three values with the existing finite-number validator;
- return DiagnosticsConfig() when diagnostics is absent;
- reject keys outside psd_max_frequency_hz and psd_n_fft;
- parse psd_max_frequency_hz as a strictly positive finite number;
- parse psd_n_fft as either null/None or a positive integer, rejecting booleans.

Add quality_control and diagnostics to load_config and serialize them through the
existing asdict-based provenance path. Use explicit defaults in the parser rather
than ProcessingConfig.__dataclass_fields__.

Update examples/configuration.yml with the two optional sections and comments that
the values are protocol/reporting settings, not automatic corrections.

### Step 3: Verify and commit

Run:

```
uv run pytest tests/test_config.py -v
uv run ruff check src/mri_correction/config.py tests/test_config.py
git diff --check
```

Expected: all configuration tests pass and Ruff reports no errors.

Commit:

```
git add src/mri_correction/config.py tests/test_config.py examples/configuration.yml
git commit -m "feat: expose QC and diagnostic configuration"
```

---

## Task 2: Expose optional robustness parameters without changing defaults

**Files:**

- Modify: src/mri_correction/config.py
- Modify: src/mri_correction/fastr.py during the facade transition in Task 4
- Modify: src/mri_correction/fastr_geometry.py during extraction in Task 4
- Modify: tests/test_config.py
- Modify: tests/test_residual_gate.py

Add four optional processing settings with today’s values:

```
residual_gate_mad_multiplier: float = 8.0
residual_gate_ratio: float = 8.0
residual_gate_max_fraction: float = 0.02
adaptive_improvement_ratio: float = 0.85
```

Add them to the optional processing key set and parse them using finite-number
validation. Enforce:

- both multiplier and ratio are strictly positive;
- maximum excluded fraction is in (0, 1];
- adaptive improvement ratio is in (0, 1].

Add tests for defaults, custom values, unknown fields, booleans, non-finite values,
and out-of-range values. The tests must continue to assert that residual_gate and
adaptive_window default to False.

During the FASTR extraction, add these keyword-only parameters to gate_fastr_geometry:
residual_gate_mad_multiplier=8.0, residual_gate_ratio=8.0,
residual_gate_max_fraction=0.02, mains_frequency_hz=60.0, and
mains_exclusion_hz=1.0. Add adaptive_improvement_ratio=0.85 as a keyword-only
parameter to adapt_fastr_geometry. Keep the existing parameters and ordering
unchanged.

Replace only the uses of the residual-gate and adaptive-improvement constants inside
those call paths. Validate the new direct-call values at the function boundary.
Leave fixed implementation details such as the minimum usable-neighbour count and
protected edge volumes as code constants.

Add a gate test with non-default thresholds and an adaptive-window test with a
non-default improvement ratio, then verify the existing default behavior remains
unchanged.

Run:

```
uv run pytest tests/test_config.py tests/test_residual_gate.py tests/test_adaptive_window.py -v
uv run ruff check src tests
```

Commit:

```
git add src/mri_correction/config.py tests/test_config.py tests/test_residual_gate.py tests/test_adaptive_window.py
git commit -m "feat: configure FASTR robustness thresholds"
```

---

## Task 3: Extract pipeline support modules with compatibility wrappers

**Files:**

- Create: src/mri_correction/pipeline_io.py
- Create: src/mri_correction/pipeline_markers.py
- Create: src/mri_correction/pipeline_provenance.py
- Modify: src/mri_correction/pipeline.py
- Modify: tests/test_pipeline.py
- Create: tests/test_pipeline_modules.py

### Step 1: Add module-boundary tests

Create tests/test_pipeline_modules.py with import and delegation checks:

```
from mri_correction import pipeline, pipeline_io, pipeline_markers


def test_pipeline_keeps_existing_private_test_seams() -> None:
    assert callable(pipeline._lowpass_and_decimate)
    assert callable(pipeline._save_psd_plot)
    assert callable(pipeline._measure_residual_qc)


def test_pipeline_helpers_are_available_in_focused_modules() -> None:
    assert callable(pipeline_io.lowpass_and_decimate)
    assert callable(pipeline_markers.bad_gradient_markers)
```

Move the existing pipeline tests’ direct helper calls to focused modules only where
they test the new module’s behavior; keep the original calls that verify compatibility
wrappers.

### Step 2: Move I/O and filtering helpers unchanged

Create pipeline_io.py with six functions, retaining the existing signatures and
validation bodies under public non-underscore names: validate_input_files,
output_paths, validate_output_paths, validate_rates, resolve_reference_channel,
and lowpass_and_decimate.

Use the current scipy.signal.butter and filtfilt call and preserve the
window-start decimation phase exactly. In pipeline.py, retain wrappers with the
existing underscore names, for example:

```
def _lowpass_and_decimate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    window: OutputWindow,
) -> np.ndarray:
    return pipeline_io.lowpass_and_decimate(
        data,
        sampling_rate=sampling_rate,
        output_sampling_rate=output_sampling_rate,
        lowpass_hz=lowpass_hz,
        window=window,
    )
```

The orchestration function must continue calling the wrapper so existing
monkeypatch.setattr(pipeline_module, "_lowpass_and_decimate", capture_window)
tests remain effective.

### Step 3: Move marker and provenance helpers

Create pipeline_markers.py with validate_marker_output_positions,
skipped_group_spans, bad_gradient_markers, and residual_qc_markers. Pass all
required values explicitly; do not read pipeline globals.

Create pipeline_provenance.py with trim_provenance, make_provenance,
jsonable_config, stringify_paths, and sha256. Keep the current sidecar keys and
value types. make_provenance must include new quality_control and diagnostics
values through serialized configuration and add actual PSD/QC settings to their
respective report blocks.

Keep underscore wrappers in pipeline.py for existing private helper names, and make
the wrappers pure delegates. Avoid adding a second implementation of any helper.

### Step 4: Verify extraction

Run:

```
uv run pytest tests/test_pipeline.py tests/test_pipeline_modules.py -v
uv run ruff check src tests
git diff --check
```

Expected: all pipeline behavior tests pass, including PSD monkeypatch tests,
low-pass phase tests, marker annotations, and provenance assertions.

Commit:

```
git add src/mri_correction/pipeline.py src/mri_correction/pipeline_io.py src/mri_correction/pipeline_markers.py src/mri_correction/pipeline_provenance.py tests/test_pipeline.py tests/test_pipeline_modules.py
git commit -m "refactor: split pipeline support modules"
```

---

## Task 4: Extract FASTR timing, geometry, templates, and processing

**Files:**

- Create: src/mri_correction/fastr_types.py
- Create: src/mri_correction/fastr_timing.py
- Create: src/mri_correction/fastr_geometry.py
- Create: src/mri_correction/fastr_templates.py
- Create: src/mri_correction/fastr_processing.py
- Modify: src/mri_correction/fastr.py
- Modify: tests/test_fastr_modules.py and the existing FASTR tests

### Step 1: Establish the dependency direction

Use this import graph and do not import the facade from an implementation module:

```
fastr_types
    ^
fastr_timing       fastr_templates
    ^                    ^
    |              fastr_geometry
    |                    ^
    +------------ fastr_processing
                         ^
                      fastr.py
```

fastr_types.py owns FastrInputError, FastrProvenance, FastrCorrection,
FastrGeometry, FastrAlignment, _TemplateWindow, and _ArtifactEpoch. Keep the
current frozen/slot settings, array-copy behavior, read-only flags, field order,
and docstrings.

fastr_timing.py owns FmriAcquisitionTiming, load_bids_fmri_timing,
make_group_trigger_samples, and timing/file validators.

fastr_geometry.py owns prepare_fastr_geometry, gate_fastr_geometry,
adapt_fastr_geometry, _build_fastr_geometry, _run_fastr_with_edges,
template-window selection, epoch-bound checks, residual-gate scoring,
adaptive-window selection, and corresponding validators.

fastr_templates.py owns _make_template_high_pass, _template_estimate_signal,
_make_interpolation_filter, _interpolate, _extract_epochs, _place_epochs,
_make_templates, _mean_selected_epochs, _template_residual, _correlate,
_fit_group_shifts, and _fit_channel_noise.

fastr_processing.py owns fit_fastr_alignment, apply_fastr_batch, residual_obs,
and recording/reference/channel/basis validation. It imports the types, geometry
objects, and template helpers directly.

### Step 2: Move code without semantic edits

Move each function body verbatim first. Only change imports and qualified references.
Do not improve numerical expressions during extraction. Add __all__ in the facade only
for existing public names:

```
__all__ = [
    "FastrAlignment",
    "FastrCorrection",
    "FastrGeometry",
    "FastrInputError",
    "FastrProvenance",
    "FmriAcquisitionTiming",
    "acquisition_group_fastr",
    "acquisition_group_fastr_with_edges",
    "apply_fastr_batch",
    "adapt_fastr_geometry",
    "fit_fastr_alignment",
    "gate_fastr_geometry",
    "load_bids_fmri_timing",
    "make_group_trigger_samples",
    "prepare_fastr_geometry",
    "residual_obs",
    "slice_fastr",
    "slice_fastr_with_edges",
]
```

The facade’s four convenience wrappers must call extracted geometry and processing
functions with the same defaults as today. Direct calls that omit new robustness
parameters must produce the same results as before.

### Step 3: Thread explicit robustness values

In fastr_geometry.py, add small validators for positive ratios and fractions, then
pass values through _residual_outlier_scores, _slice_harmonic_energy,
_outlier_groups, _robust_outliers, _cap_outliers, and
_volumes_helped_by_local_window. Defaults must be exactly 8.0, 8.0, 0.02,
60.0, 1.0, and 0.85 at the public boundary.

In fastr_processing.py, retain residual_obs defaults rank=4 and
interpolation_factor=10; leave its fixed 70 Hz residual high-pass implementation
unchanged because it is a method implementation constant, not a pipeline setting.

### Step 4: Add facade and numerical regression tests

Add tests that verify the facade objects are the exact extracted objects:

```
from mri_correction import fastr
from mri_correction.fastr_geometry import prepare_fastr_geometry
from mri_correction.fastr_processing import apply_fastr_batch


def test_fastr_facade_reexports_extracted_implementations() -> None:
    assert fastr.prepare_fastr_geometry is prepare_fastr_geometry
    assert fastr.apply_fastr_batch is apply_fastr_batch
```

Retain and run all existing FASTR geometry, alignment, batch invariance, boundary,
residual OBS, and synthetic tests. Add one explicit default-vs-explicit comparison:
call gate/adaptive functions once with omitted optional values and once with the
documented defaults, then assert equal arrays and equal provenance.

### Step 5: Verify extraction and commit

Run:

```
uv run pytest tests/test_fastr.py tests/test_fastr_batch.py tests/test_slice_fastr.py tests/test_adaptive_window.py tests/test_residual_gate.py tests/test_fastr_modules.py -v
uv run ruff check src tests
git diff --check
```

Expected: all existing FASTR results and provenance arrays remain unchanged.

Commit:

```
git add src/mri_correction/fastr.py src/mri_correction/fastr_types.py src/mri_correction/fastr_timing.py src/mri_correction/fastr_geometry.py src/mri_correction/fastr_templates.py src/mri_correction/fastr_processing.py tests/test_fastr_modules.py tests/test_fastr.py tests/test_fastr_batch.py tests/test_slice_fastr.py tests/test_adaptive_window.py tests/test_residual_gate.py
git commit -m "refactor: split FASTR implementation modules"
```

---

## Task 5: Wire configurable QC and PSD settings through the pipeline

**Files:**

- Modify: src/mri_correction/pipeline.py
- Modify: src/mri_correction/pipeline_io.py
- Modify: src/mri_correction/psd.py
- Modify: src/mri_correction/residual_qc.py
- Modify: src/mri_correction/pipeline_provenance.py
- Modify: tests/test_pipeline.py, tests/test_residual_qc.py

### Step 1: Add forwarding tests

Extend pipeline fixture tests so a configuration containing custom values proves:

- quality_control.block_seconds controls residual-QC block count and duration;
- quality_control.mains_frequency_hz and mains_exclusion_hz are passed to harmonic selection;
- diagnostics.psd_max_frequency_hz becomes PSD fmax, capped only by output Nyquist;
- diagnostics.psd_n_fft is passed to MNE when non-null and omitted when null.

Use the existing PSD capture seam in tests/test_pipeline.py rather than inspecting MNE
internals. Assert the sidecar contains effective settings.

### Step 2: Wire values without changing omitted behavior

Change _measure_residual_qc to accept explicit block_seconds,
mains_frequency_hz, and mains_exclusion_hz; call:

```
harmonics = slice_harmonics(
    groups_per_volume=timing.groups_per_volume,
    repetition_time_seconds=timing.repetition_time_seconds,
    nyquist_hz=output_rate / 2.0,
    mains_hz=mains_frequency_hz,
    exclusion_hz=mains_exclusion_hz,
)
residuals = block_residual_uv(
    np.asarray(corrected) * 1e6,
    sampling_rate=output_rate,
    harmonics=harmonics,
    block_seconds=block_seconds,
)
```

Call it from _run_correction with config.quality_control values. Replace the pipeline
_RESIDUAL_BLOCK_SECONDS constant with the typed config value.

Call gate_fastr_geometry with the four processing robustness values and the two
quality-control mains values. Call adapt_fastr_geometry with
adaptive_improvement_ratio.

Call _save_psd_plot with
fmax=min(config.diagnostics.psd_max_frequency_hz, output_rate / 2.0) and
n_fft=config.diagnostics.psd_n_fft. Keep the wrapper’s n_fft=None behavior unchanged.

Update pipeline_provenance.py to add effective values under residual_qc and an
output.psd_settings object while retaining all existing keys.

### Step 3: Verify and commit

Run:

```
uv run pytest tests/test_pipeline.py tests/test_residual_qc.py -v
uv run ruff check src tests
git diff --check
```

Expected: existing fixture output remains valid; custom settings are visible in the
sidecar and diagnostic captures.

Commit:

```
git add src/mri_correction/pipeline.py src/mri_correction/pipeline_io.py src/mri_correction/psd.py src/mri_correction/residual_qc.py src/mri_correction/pipeline_provenance.py tests/test_pipeline.py tests/test_residual_qc.py
git commit -m "feat: wire configurable pipeline diagnostics"
```

---

## Task 6: Clean documentation and enforce the repository quality gate

**Files:**

- Modify: README.md
- Modify: docs/algorithm.md
- Modify: docs/validation.md
- Modify: examples/configuration.yml
- Do not modify: pyproject.toml; its existing quality settings are the required target
- Modify: affected tests for line length and duplicate assertions

Document:

- focused module responsibilities at a short package-maintainer level;
- all user-tunable processing, quality-control, diagnostic, robustness, and trim values
  in the YAML example;
- which values are protocol/scientific choices and which remain fixed implementation
  details;
- that BCG/ECG code lives in BCG-Correction and is not part of this package; and
- that provenance records resolved settings.

Remove redundant assertions and fix the seven existing Ruff line-length violations as
ordinary cleanup. Do not weaken Ruff configuration or add exclusions.

Run the complete quality gate:

```
uv run pytest
uv run ruff check src tests
git diff --check
```

Expected: 308 or more tests pass with zero failures, Ruff exits 0, and
git diff --check is silent. If the test count changes, explain the added regression
coverage in the commit summary rather than deleting existing tests.

Commit:

```
git add README.md docs/algorithm.md docs/validation.md examples/configuration.yml pyproject.toml tests src
git commit -m "docs: organize FASTR configuration and maintenance guidance"
```

---

## Final review checklist

- [ ] git status --short --branch shows the intended commits and no untracked files.
- [ ] git diff origin/main..HEAD --stat contains only the approved design, modularization, configuration, tests, and documentation changes.
- [ ] mri_correction.fastr imports all existing public names from one stable facade.
- [ ] mri_correction.pipeline private test seams still delegate to exactly one implementation.
- [ ] Existing YAML without new sections produces the same CorrectionConfig behavior and output defaults.
- [ ] New configurable values are validated at both YAML and direct function boundaries.
- [ ] Provenance contains resolved configuration and effective QC/PSD settings.
- [ ] No BCG/ECG code or dependency was added to FASTR-Python.
- [ ] Full pytest, Ruff, and diff checks have fresh passing evidence.
