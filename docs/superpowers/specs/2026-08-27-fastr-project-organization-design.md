# FASTR project organization and configuration design

**Date:** 2026-08-27
**Status:** Approved for implementation

## Objective

Make the FASTR-Python repository easier to read, test, and extend as research
software while preserving the current numerical algorithm, public module imports,
configuration behavior, output files, marker semantics, and failure policy.

The separate BCG-Correction project remains out of scope. This repository contains
scanner-gradient correction only.

## Current context

The merged `main` branch is the current FASTR-only implementation. The baseline
contains 308 passing tests and a clean Git worktree. The main organization problems
are concentrated in three modules:

- `fastr.py` combines BIDS timing, input validation, geometry construction, template
  estimation, alignment, residual gating, adaptive windows, and residual OBS in one
  1,776-line module.
- `pipeline.py` combines run orchestration, file/rate validation, filtering, marker
  annotation, PSD setup, residual QC, and provenance in one 745-line module.
- `config.py` parses the run configuration but leaves several QC and diagnostic
  choices as module-level constants, and uses dataclass internals to recover
  defaults.

The existing `psd.py`, `residual_qc.py`, and `window.py` modules are already useful
boundaries and will remain small focused modules.

## Invariants

The implementation must preserve these externally observable properties:

1. `mri_correction.fastr` remains the public import location for its existing public
   classes and functions. `mri_correction.pipeline` remains the public pipeline
   import location. Any moved implementation is re-exported through these facades.
2. Existing YAML files, including the current `examples/configuration.yml`, remain
   valid and retain their current defaults. New configuration sections and fields are
   optional; omitted values resolve to today’s behavior.
3. Existing direct Python calls retain their signatures and numerical defaults.
   New tunable values may be additive keyword-only arguments with the same defaults.
4. BrainVision positions stay one-based; NumPy sample indices stay zero-based.
5. Timing gaps, invalid data, output collisions, invalid rates, and invalid
   configuration continue to fail explicitly. No repair or silent fallback is added.
6. Output files, marker types/descriptions, marker positions, sidecar structure, and
   correction sample values remain unchanged when new options are omitted.
7. All stochastic simulation behavior remains deterministic for the same seed.

The cleanup will not replace `filtfilt` with another filter implementation, alter
floating-point operation order intentionally, rename public configuration keys, or
add a BCG/ECG pipeline.

## Architecture

### FASTR implementation

The current public module becomes a thin compatibility façade. Its implementation is
split by responsibility into these modules:

| Module | Responsibility |
| --- | --- |
| `fastr_types.py` | `FastrInputError`, immutable result/provenance models, and private geometry value objects. |
| `fastr_timing.py` | `FmriAcquisitionTiming`, BIDS JSON loading, volume-marker validation, and fractional group-trigger construction. |
| `fastr_geometry.py` | Geometry validation/building, boundary handling, acquisition-slot and alternating neighbour windows, residual gating, and adaptive-window selection. |
| `fastr_templates.py` | Interpolation, epoch extraction/placement, template means, high-pass template signals, amplitude fitting, and correlation helpers. |
| `fastr_processing.py` | Shared alignment fitting, batch application, residual OBS, and processing-specific input validation. |
| `fastr.py` | Public convenience wrappers (`slice_fastr`, acquisition-group variants, and re-exports). |

The dependency direction is one-way: timing and types are foundational; geometry
depends on types and template helpers; processing depends on geometry, types, and
template helpers; the façade depends on all public implementation modules. No moved
module imports the façade, so circular imports cannot hide behavior.

The existing public `FastrGeometry`, `FastrAlignment`, `FastrCorrection`,
`FastrProvenance`, `FmriAcquisitionTiming`, and `FastrInputError` names are imported
back into `fastr.py`. Private implementation names are not treated as API, but the
existing pipeline test seams remain available as wrappers where tests and diagnostics
currently rely on them.

### Pipeline implementation

The run flow stays in `pipeline.py`, but side-effecting and reporting concerns move
behind focused helpers:

| Module | Responsibility |
| --- | --- |
| `pipeline_io.py` | Input/output path validation, rate validation, reference-channel resolution, and full-recording filtering/decimation. |
| `pipeline_markers.py` | Mapping output markers, skipped-group spans, residual-QC annotations, and output-position validation. |
| `pipeline_provenance.py` | JSON-safe configuration conversion, hashes, trim/QC/FASTR provenance assembly. |
| `pipeline.py` | Load inputs, construct geometry, execute channel batches, invoke reporting helpers, and return `CorrectionSummary`. |

The pipeline continues to create outputs only after all input and output collision
checks pass. Temporary corrected data remains a local implementation detail of the
orchestrator.

### Configuration surface

`ProcessingConfig` continues to expose the current flat processing fields. Its
optional values use explicit loader defaults instead of reading
`__dataclass_fields__`. The existing processing controls remain the correction
controls: interpolation factor, template neighbour count, alignment search radius,
template high-pass, output filter/rate, batch size, reference channel, residual gate,
and adaptive window.

The following optional configuration groups expose parameters that currently affect
research interpretation but are hidden in implementation constants:

```yaml
quality_control:
  block_seconds: 30.0
  mains_frequency_hz: 60.0
  mains_exclusion_hz: 1.0

diagnostics:
  psd_max_frequency_hz: 100.0
  psd_n_fft: null
```

The residual-QC threshold remains `processing.residual_threshold_uv` so current YAML
files and sidecars retain their established field name. Optional robustness controls
for the enabled residual gate and adaptive window are added under `processing` with
the current values as defaults: MAD multiplier, ratio threshold, maximum excluded
fraction, and adaptive improvement ratio. These are the smallest set of hidden
algorithm decisions that can materially change which data are corrected. Fixed
implementation details such as interpolation filter tap span, chunk size, and marker
serialization format remain code constants.

Every resolved value is included in the existing `configuration` provenance object;
the residual-QC report also records its effective block and mains settings, and the
PSD report records its effective frequency limit and FFT setting.

All new values are validated at configuration load time and again at the lower-level
function boundary where a direct Python caller can provide them. Invalid types,
non-finite values, non-positive durations/frequencies, and invalid integer ranges
raise the existing domain-specific errors.

## Data flow

```text
YAML -> typed CorrectionConfig
                    |
BrainVision + BIDS -> validated markers/timing -> FASTR geometry
                    |                              |
                    |                              -> shared alignment
                    |                              -> optional gate/window selection
                    v
              channel batches -> correction -> full-recording filter -> output window/decimation
                    |                                                               |
                    +---------------- provenance + residual QC + PSD ---------------+
```

The numerical order remains exactly: validate, load, derive triggers, build geometry,
fit alignment, optionally adjust template membership, correct batches, filter the
whole recording, window and decimate, write markers/data, then write diagnostics and
provenance.

## Testing and documentation

Tests will be extended before implementation changes where practical and will cover:

- imports through the existing public façade after module extraction;
- configuration defaults, custom QC/diagnostic values, robustness validation, and
  provenance serialization;
- direct lower-level calls with omitted and explicit new keyword arguments;
- pipeline forwarding of QC and PSD settings;
- unchanged synthetic correction outputs, marker mapping, boundary annotations,
  batch-size invariance, and output collision behavior; and
- the full suite, Ruff, and `git diff --check`.

`README.md`, `docs/algorithm.md`, `docs/validation.md`, and
`examples/configuration.yml` will describe the new optional sections, distinguish
algorithm parameters from implementation constants, and keep the BCG-Correction
project boundary explicit.

## Completion criteria

The cleanup is complete when the repository has focused modules with no duplicated
orphaned implementation, all currently hidden research/QC choices intended for user
tuning are represented in YAML and provenance, existing public behavior is covered
by tests, Ruff is clean, and the complete test suite passes from a fresh command.
