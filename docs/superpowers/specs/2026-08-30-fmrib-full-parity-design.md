# FMRIB FASTR Full-Parity Design

## Goal

Extend the existing BIDS/BrainVision Python pipeline so it exposes every
scientifically relevant processing capability of the FMRIB `fmrib_fastr.m`
pipeline without adopting EEGLAB structures, MATLAB GUI behavior, or silent
legacy heuristics.

The authoritative reference is `sccn/fMRIb` commit
`2aa522bc5ec4215f42b3ba8efdb2b84d2a312935`. The installed EEGLAB FMRIB 2.1
copy has the same SHA-256 digest for `fmrib_fastr.m`:
`0c193406735266e94000eb16aeeaf13d62e4e3f9b975f55e19f84e30c12dd4de`.

## Scope

The Python pipeline will provide the MATLAB algorithm's processing choices
through its existing YAML-driven BIDS/BrainVision interface:

- sub-sample trigger alignment on a shared reference channel;
- moving-average artifact templates and per-channel least-squares scaling;
- unscaled template subtraction and residual-stage exclusion for non-EEG
  channels;
- configurable trigger position within the artifact epoch;
- explicit missing-volume-marker repair;
- residual optimal-basis subtraction with a fixed or automatically selected
  rank and optional section-wise basis estimation;
- optional adaptive noise cancellation (ANC);
- optional low-pass filtering, including no output low-pass when no output
  decimation is requested;
- internal interpolation and return to the requested output sample rate.

EEGLAB GUI callbacks, EEGLAB history strings, direct MATLAB positional
arguments, MEX compilation, QRS detection, and pulse-artifact subtraction are
outside the scope. The last two are separate tools in the FMRIB plug-in and are
not stages of `fmrib_fastr.m`.

## Reference audit

| FMRIB capability | Current Python state | Design decision |
| --- | --- | --- |
| Slice-trigger alignment | Low-level `slice_fastr` | Retain and test; production remains BIDS acquisition-group based |
| Volume-trigger processing | BIDS acquisition slots derived from volume markers | Retain the more explicit multiband geometry |
| Interpolation factor | Supported | Retain strict positive-integer validation |
| Moving averaging window | Supported | Retain strict even-count validation instead of MATLAB's silent adjustment |
| Three-native-sample alignment search | Supported and configurable | Retain |
| Trigger position fraction | Hard-coded to `0.03` | Add one validated configuration value |
| Missing-trigger correction | Missing markers are rejected | Add explicit repair mode with an expected volume count; never repair implicitly |
| Non-EEG channel handling | Supported | Retain and include flat channels in ANC exclusion |
| OBS disabled or fixed rank | Supported | Retain |
| Automatic OBS rank | Not implemented | Add the three-criterion FMRIB rank selector |
| Section-wise OBS | Low-level support only | Expose an optional section duration in YAML |
| 70 Hz residual high-pass | Supported | Retain |
| Adaptive noise cancellation | Not implemented in production | Add a faithful, opt-in LMS stage |
| Low-pass disabled | Not supported by the production rate validator | Allow only when output rate equals input rate |
| FMRIB `firls`/`filtfilt` output filter | Deliberately replaced by a flatter MNE FIR | Keep the current filter; validate MATLAB comparisons before the differing output filter |
| First/last incomplete artifacts | Preserved and annotated | Retain the explicit Python behavior |
| 60-second memory sections | Not needed for template subtraction because Python batches channels | Do not duplicate memory-only segmentation |

Silent MATLAB behavior is not copied. Odd or oversized windows, ambiguous
marker gaps, impossible filters, insufficient OBS epochs, and unstable ANC
references raise domain errors. This preserves the project's fail-fast policy.

## Architecture

### Timing and marker repair

`fastr_timing.py` will own marker repair. A new function accepts validated volume
starts, the integer samples per volume, and an expected count. It identifies
missing positions only when every observed interval is an integer multiple of
the repetition interval within the existing one-sample clock tolerance and the
result has exactly the declared count. Any ambiguity raises `FastrInputError`.

The pipeline configuration will expose `timing.missing_volume_markers` with
values `error` or `repair`. Repair requires `timing.expected_volume_count`.
The default remains `error`.

### Artifact geometry

`prepare_fastr_geometry` will accept `pre_trigger_fraction`. The value must be
finite and lie in `[0, 1]`. `_measure_artifact_epoch` will use it instead of a
module constant. The default remains `0.03`, so existing outputs do not change.
The selected value will be written to provenance.

### Residual optimal basis set

OBS responsibilities will be separated:

- `select_obs_rank` chooses the rank from ordered explained variance using the
  FMRIB slope, cumulative-variance, and per-component-variance criteria;
- section construction defines which consecutive epochs share a basis;
- residual fitting subtracts a fixed rank or a rank selected independently for
  each channel and section.

`processing.residual_obs_rank` will accept a positive integer or `auto`.
`processing.residual_obs_section_seconds` will accept a positive number or
`null`. Existing defaults remain rank 4 and one basis for the recording.

Automatic selection must return at least one component and no more components
than can be estimated from the section. A spectrum that does not satisfy the
FMRIB knee rules raises a clear error instead of indexing past the array as the
MATLAB implementation can.

### Adaptive noise cancellation

A focused `fastr_anc.py` module will implement the LMS stage from `fastranc.c`:

1. high-pass the template-corrected channel;
2. use the estimated gradient artifact as the reference;
3. scale the reference by least squares;
4. derive the filter order from the longest artifact epoch;
5. derive the LMS step from reference variance;
6. subtract the adaptive filter output from the template-corrected signal.

`apply_fastr_batch` will return the artifact estimate alongside corrected data
so ANC consumes the actual fitted template and OBS estimate rather than
reconstructing a reference. ANC will run after optional OBS and before output
rate conversion, matching the MATLAB stage order. Excluded and flat channels
will bypass ANC. Zero-variance references, non-finite states, or divergent
outputs raise `FastrInputError`; they are not silently skipped.

The YAML option is `processing.adaptive_noise_cancellation`, default `false`.
The provenance sidecar records whether it ran, its filter order, and the
per-channel reference scale and step size. The documentation will retain the
measured warning that ANC can remove genuine narrowband activity near scanner
harmonics.

### Output filtering

`processing.lowpass_hz` will accept zero to mean no output low-pass. Zero is
valid only when `output_sampling_rate_hz` equals the input rate. Decimation
without an anti-alias filter remains an error. PSD limits will use output
Nyquist when the low-pass is disabled.

The current MNE-designed, delay-compensated FIR remains the production filter.
The FMRIB `firls` filter is an implementation detail rather than a user-facing
capability, and the repository already documents why its twice-applied response
is scientifically undesirable.

## Data flow

1. Read BrainVision data and exact markers.
2. Load BIDS fMRI timing.
3. Validate volume markers or explicitly repair them.
4. Derive acquisition-group triggers.
5. Build geometry with the configured trigger fraction.
6. Fit one alignment from the configured reference channel.
7. Apply optional residual gating or adaptive window selection.
8. For each channel batch, fit and subtract moving templates while retaining
   the fitted artifact estimate.
9. Apply optional OBS and add its fitted residual to the artifact estimate.
10. Apply optional ANC to EEG channels.
11. Apply optional output low-pass, window, and decimation.
12. Apply explicitly configured line regression, write outputs, diagnostics,
    annotations, and provenance.

## Validation strategy

### Automated tests

Every production change follows a failing-test-first cycle. Tests will cover:

- unique and ambiguous marker repair, count mismatches, and clock tolerance;
- trigger fractions at 0, 0.03, and 1;
- FMRIB automatic-rank criteria and degenerate spectra;
- fixed, automatic, global, and section-wise OBS;
- LMS sample-by-sample output against a frozen MATLAB/C-derived fixture;
- ANC exclusion, flat reference rejection, divergence detection, and batch
  invariance;
- ANC pipeline stage ordering and provenance;
- disabled low-pass validation and behavior;
- unchanged output under the existing default configuration.

The full Pytest and Ruff suites must remain clean.

### MATLAB oracle

MATLAB R2026a, EEGLAB, and the installed FMRIB 2.1 plug-in will generate oracle
results. Small deterministic fixtures will isolate alignment/template fitting,
automatic OBS rank, OBS fitting, and ANC. Identical stage contracts require
sample-level agreement within a documented floating-point tolerance.

For representative recordings under
`/Volumes/KINGSTON/EEG_fMRI_data/source_data/*/eeg/original_untrimmed_5khz`, the
validation harness will compare the same channels, marker span, and processing
stages. Because MATLAB's volume mode treats a volume as one artifact while the
Python pipeline uses BIDS-derived multiband acquisition slots, whole-pipeline
acceptance is based on both residual suppression and injected-signal retention,
not exact sample equality. Results will report scanner-harmonic residuals,
broadband transfer, ECG preservation, and boundary coverage.

Private recordings and generated outputs remain outside Git. Reusable oracle
scripts and aggregate comparison results may be tracked; subject data and
absolute private paths may not.

## Documentation and provenance

`README.md`, `docs/algorithm.md`, `docs/validation.md`, and the example YAML will
describe every parity option, its default, its risks, and the MATLAB mapping.
Provenance will include the FMRIB reference commit, marker repair details,
trigger fraction, OBS rank mode and selected ranks, OBS section duration, and
ANC diagnostics.

## Completion criteria

The work is complete when:

- each in-scope row in the reference audit is implemented or explicitly
  documented as an intentional Python interface difference;
- existing default configurations preserve their prior results;
- all new functions have regression and boundary tests;
- isolated shared-contract stages agree with MATLAB fixtures;
- real-recording comparison reports both artifact suppression and signal
  preservation;
- the full test, lint, and whitespace checks pass;
- an independent code review finds no unresolved critical or important issues.
