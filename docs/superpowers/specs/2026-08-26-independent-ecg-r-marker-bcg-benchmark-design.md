# Independent ECG R-marker detection and BCG benchmark

**Date:** 2026-08-26
**Status:** Design approved; implementation pending

## Objective

Determine whether the FASTR-Python pipeline can produce a better BCG-corrected EEG
recording than the BrainVision Analyzer reference for this EEG-fMRI cohort.

The first required capability is an independent cardiac-event detector. It must recover a
complete and reliable train of `Pulse Artifact,R` markers from the ECG carried by the
FASTR gradient-corrected recording. The detector must not depend on BrainVision Analyzer
markers, recovered Analyzer markers, or the Analyzer-corrected EEG.

The second capability is a BCG correction stage that consumes those independent marker
times and can be compared fairly with the existing Analyzer output.

## Scope

In scope:

- deterministic ECG-only R-peak detection at the recording's native sampling rate;
- BrainVision-compatible `Pulse Artifact,R` marker output and provenance;
- MNE PCA-OBS and a local-average/AAS correction arm using the independent markers;
- paired comparison with the existing Analyzer BCG-corrected recordings;
- unit, integration, and cohort-level validation with held-out and null-shift metrics.

Out of scope:

- using Analyzer markers to initialize, fit, tune, or repair the production detector;
- using EEG channels, scanner-volume markers, or pain-task events to create cardiac markers;
- reproducing undocumented Analyzer internals exactly;
- silently falling back to a different input stage or detector when validation fails;
- changing the existing FASTR gradient-correction algorithm in this work.

## Data provenance and benchmark arms

The detector input must be the FASTR-only gradient-corrected BrainVision recording. The
supplied directory
`/Volumes/KINGSTON/EEG_fMRI_data/source_data/step3_bcg_corrected` is an Analyzer
BCG-corrected reference stage, not the fair input for our correction arm. It may be used
for comparison and for verifying that ECG samples survive Analyzer's correction, but it
must not be used as the EEG input to our correction.

The implementation will require an explicit FASTR input root and an explicit Analyzer
reference root. It will fail if either root is missing, ambiguously paired, or contains a
different recording geometry. Existing `step2_pulse_markers_recovered_v2` annotations
are legacy evidence only; they are not production detector input.

The headline comparison will contain at least these arms:

1. FASTR gradient-corrected EEG with independent ECG-derived R markers and our BCG
   correction.
2. Analyzer's existing BCG-corrected recording as the empirical reference.

Diagnostic ablations may add FASTR EEG corrected with Analyzer markers and the same local
correction implementation. That arm isolates marker quality from correction-method
quality, but it is not an independent production result and will not be presented as one.

Every paired recording must be checked for matching sample count, sampling frequency,
channel order, and ECG identity. The ECG channel must be excluded from EEG BCG fitting and
must remain sample-identical in any written corrected output.

## Independent detector design

The production detector API will accept only an ECG vector, sampling frequency, and the
detector configuration. It will not accept a Raw object or an annotation collection. This
interface makes Analyzer-marker leakage structurally difficult and makes the core
algorithm testable with synthetic signals.

### Signal conditioning

The detector will work on a copy of the ECG and preserve the original samples for final
timing localization. Conditioning will be deterministic and explicitly configured:

- finite one-dimensional ECG samples are required;
- a zero-phase 7--40 Hz QRS-enhancing representation and a short smoothing operation are
  used for candidate generation, following the single-channel MRI detector described by
  Niazy et al.;
- a k-Teager energy representation is used as the polarity-invariant complex lead, with
  the emphasis parameter derived from the sampling rate and a documented target
  frequency;
- edge handling is explicit, and candidates whose full fitting window is unavailable are
  rejected rather than silently clipped.

The initial protocol configuration will use a documented QRS band and a fixed template
window appropriate for the 1 kHz recordings. Configuration belongs in the project YAML,
not in scattered module constants. Any change to these values requires a new benchmark
provenance record.

### Candidate generation and train selection

Candidate generation will be intentionally permissive. It will identify local QRS-like
events from the conditioned signal and retain a score for each candidate rather than
pretending that a single fixed amplitude threshold is reliable in the MRI environment.
The primary candidate generator will follow the published FMRIB sequence: a combined
adaptive MFR threshold over the nonnegative k-Teager complex lead, followed by local
candidate consolidation.

The accepted train will then be selected using only ECG-derived evidence:

1. Estimate a robust cardiac-period range from the candidate population itself.
2. Build an initial QRS template from candidates with mutually consistent morphology and
   spacing. No external event times may enter this template.
3. Match the template back to the ECG while enforcing an explicit physiological
   refractory interval.
4. Score each proposed beat using morphology agreement, QRS evidence, and compatibility
   with neighbouring RR intervals.
5. Refine the template and beat train deterministically for a fixed number of iterations.
6. Reject low-confidence candidates, double marks, and candidates that violate the
   run-level interval model. Preserve rejection counts in QC output.

The detector must choose the R timing from the original ECG within the fitted QRS window,
not from an arbitrary filtered-signal sample. It must support both ECG polarities and
must explicitly guard against T-wave and magnetohydrodynamic double detections.

The prior `pain_study` implementation supplies reusable ideas for template construction,
double-mark handling, refractory checks, and lock-ratio measurement. Its
Analyzer-seeded gap search will not be reused as the production detector.

The FMRIB/EEGLAB implementation is a method reference only. Its public MATLAB source is
GPL-licensed; the Python implementation will be an independent implementation of the
published Christov/Niazy method and will not copy MATLAB source or depend on EEGLAB.

### Marker output and provenance

Accepted events will be written as BrainVision `Pulse Artifact,R` markers at the exact
sample positions selected by the detector. Existing non-cardiac markers will be preserved
according to the current BrainVision IO rules. The output sidecar will record:

- input recording and file hashes;
- detector configuration and software version;
- ECG channel name/index and sampling frequency;
- accepted and rejected candidate counts;
- RR summary, implied rate, refractory violations, and morphology-lock statistics;
- detector status and any explicit validation failure.

Analyzer annotations may be loaded only by a separate audit step after detection. The audit
will perform one-to-one matching within a documented tolerance and report agreement,
Analyzer-marker support, timing lag, and lag spread. These are comparisons, not labels of
ground truth.

## BCG correction design

The correction stage will consume the FASTR EEG and independent R-peak times. It will
never read Analyzer annotations internally.

The first correction implementations will be:

- MNE PCA-OBS, using `mne.preprocessing.apply_pca_obs` with explicit channel picks and
  independent QRS times;
- a local-average/AAS correction with a leave-one-out or otherwise held-out template
  where required by the metric, using the established prior-study implementation as a
  conceptual reference.

Both methods must state their correction window, channel picks, rank/components, unit
conversion, boundary policy, and whether the ECG channel is copied through unchanged.
No undocumented claim will be made that either method is identical to Analyzer's
implementation.

## Validation and scientific comparison

### Detector validation

The detector test suite will include synthetic ECG-like signals with known events,
variable amplitude and polarity, RR variability, noise, T-wave-like deflections,
double-mark opportunities, long missed-event stretches, and boundary events. Required
properties include deterministic output, no duplicate events within the refractory rule,
correct sample-coordinate handling, and independence from arbitrary annotation changes.

On real recordings, QC will report:

- total markers and coverage;
- RR median, robust spread, minimum, maximum, and implied rate;
- QRS-template lock/correlation statistics;
- rejected candidates and rejection reasons;
- one-to-one agreement with Analyzer markers as an audit only;
- agreement with an optional second detector as an audit only, never as an acceptance rule.

MNE's documented ECG-event detector will be implemented as a baseline/audit where useful,
not silently substituted for the independent detector. See the official MNE ECG artifact
workflow and PCA-OBS documentation:

- https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
- https://mne.tools/stable/generated/mne.preprocessing.find_ecg_events.html
- https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html

### Correction validation

The primary BCG-removal metric will be held-out heartbeat-locked residual energy or RMS,
computed separately per run and summarized with paired differences. The event-locked
template must be fit on one subset of beats and scored on held-out beats to avoid rewarding
the correction for averaging away noise.

Required controls and preservation metrics:

- circularly shifted beat-train nulls to test true cardiac locking;
- raw-to-corrected residual reduction in predefined BCG-sensitive windows/bands;
- broadband and task-relevant signal retention;
- evoked/task-event preservation where a valid task event train exists;
- synthetic artifact-injection recovery to quantify attenuation and phase distortion;
- visual inspection of representative clean, difficult, and low-marker runs.

Spectral notch reduction alone is not a success criterion. A method that lowers power by
attenuating broad EEG activity or by accepting false markers must fail the preservation or
null-control checks.

The cohort report will include run-level results, paired effect summaries, uncertainty
intervals, detector QC, and explicit failure counts. A claim that FASTR-Python outperforms
Analyzer will require a predeclared comparison rule that combines lower primary residual
with no unacceptable loss of signal preservation; the implementation will not encode an
unsupported post-hoc threshold.

## Reproducibility and failure policy

- All detector and correction settings are explicit YAML configuration.
- All processing is deterministic and records configuration/provenance.
- Input pairing is by validated recording identity, not by directory order.
- Existing outputs are never overwritten implicitly.
- Missing ECG, malformed BrainVision sidecars, invalid sample geometry, non-finite data,
  or insufficient detector evidence are explicit errors or recorded run failures.
- There are no fallback input stages, fallback marker sources, or silent detector swaps.

## Implementation boundary

The implementation should add focused modules for independent ECG detection, marker audit,
BCG correction, and benchmark reporting rather than expanding the existing gradient
correction function. Public interfaces should use small typed data structures for detector
results, marker QC, correction results, and paired metrics.

The first implementation checkpoint is a synthetic/unit-tested detector and one real-run
diagnostic. Only after those pass should the cohort benchmark be run. The final cohort
report must identify the exact FASTR input root used; if the currently supplied
`step3_bcg_corrected` path is the only available corrected recording, the workflow must
stop and request or generate the FASTR-only input rather than silently benchmarking the
wrong stage.
