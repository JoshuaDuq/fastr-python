# Recording-Level ECG Polarity and BCG Quality Gate Design

## Objective

Repair the independent ECG detector so one recording cannot mix positive and
negative cardiac deflections into a single R-event train, and prevent BCG
correction from writing output when the resulting train violates its declared
physiological interval bounds.

This is the first repair stage. It deliberately leaves AAS, PCA-OBS, correction
windows, and the existing cohort outputs unchanged. Correction behavior cannot
be evaluated rigorously until its cardiac anchors are valid.

## Evidence and failure mechanism

The detector currently assigns polarity independently to every event. Template
construction, alignment, and correlation therefore transform both positive and
negative extrema into the same positive morphology. On the recent cohort run,
15 recordings reported median-derived rates above 100 beats per minute, and
77--93% of successive detections in those recordings alternated conditioned-ECG
polarity. Several recordings also contained multi-second gaps, including a
13.746-second maximum RR interval.

The output pipeline records the detector status but applies correction and
writes BrainVision output even when that status is `degraded`. Consequently,
invalid anchors can drive correction across most of a recording while the batch
entry still reports execution status `ok`.

## Selected approach

Use one polarity for the complete recording. Candidate generation remains
polarity-independent because Teager energy is useful for finding possible QRS
locations in MRI-corrupted ECG. Before estimating the cardiac period, split the
candidates by the sign of their locally aligned conditioned-ECG extremum and
evaluate the positive and negative candidate trains independently.

For each polarity arm:

1. retain candidates whose strongest local conditioned-ECG extremum has that
   polarity;
2. estimate a physiological period from only those candidates;
3. consolidate close candidates using the existing prominence scores; and
4. measure robust arm support from the consolidated train's Teager prominence
   and physiological interval coherence.

Select the arm with the strongest deterministic support. If neither arm can
construct a physiological seed train, raise `CardiacInputError`. A global
inversion of the ECG swaps the arms but must preserve the returned event
samples. All subsequent template construction, alignment, correlation, and
missing-event recovery use the selected constant polarity. An opposite-polarity
event is rejected rather than individually inverted.

This retains the existing MRI-specific detector while fixing the confirmed
failure at its source. Replacing the detector with MNE `find_ecg_events` was
rejected because it also produced implausible event trains on several affected
recordings. Combining multiple detectors was rejected because it adds an
unvalidated reconciliation layer before the simpler polarity defect is fixed.

## Detector interfaces and quality data

Keep `detect_r_peaks()` as the public entry point. Add small private data
structures for a polarity-specific seed result so polarity selection remains
separate from morphology refinement.

Extend `CardiacDetectionQuality` with:

- `selected_polarity`, encoded as `1` or `-1`;
- positive and negative candidate counts, so the selection is auditable; and
- `degradation_reasons`, an immutable tuple of stable reason strings.

`status` remains `ok` or `degraded`, derived only from whether
`degradation_reasons` is empty. Initially, the reasons cover intervals below
`minimum_rr_seconds` and intervals above `maximum_rr_seconds`. Detector
conditions that prevent a usable train continue to raise immediately rather
than becoming a fallback status.

Do not add tunable polarity flags or thresholds to YAML. Polarity is an
observable property of the supplied ECG, not a user-selected correction mode.

## Pipeline quality gate

After detection and before `correct_bcg()`, `run_bcg_correction()` must require
`detection.quality.status == "ok"`. A degraded train raises
`CardiacInputError` containing the degradation reasons. No `.vhdr`, `.vmrk`,
`.eeg`, or `.bcg.json` file may be created.

This gate intentionally surfaces failures. It does not silently use Analyzer
markers, relax interval bounds, retry with another detector, or write an output
labelled as lower quality.

The successful provenance sidecar records all new polarity and degradation
fields through the existing quality serialization.

## Data flow

The repaired flow is:

1. validate and condition ECG;
2. generate polarity-independent Teager candidates;
3. assign each candidate one local signed extremum;
4. construct and score positive and negative seed trains;
5. select one recording-level polarity;
6. refine and recover events using only that polarity;
7. summarize RR quality and explicit degradation reasons;
8. stop on degraded quality; otherwise apply the configured BCG method; and
9. write output and provenance.

No Analyzer marker enters this flow. Analyzer remains an optional agreement
audit and is not ground truth.

## Testing strategy

Follow strict red-green-refactor cycles.

Unit tests must demonstrate:

- a signal containing a sharp QRS and a sizeable opposite-polarity secondary
  deflection yields one event per cardiac cycle;
- global ECG inversion returns the same samples and the opposite selected
  polarity;
- one isolated opposite-polarity disturbance does not redefine polarity or add
  an event;
- a train with an unrecoverable long RR gap reports an explicit maximum-RR
  degradation reason; and
- quality status is derived from the reason tuple.

Pipeline tests must demonstrate that degraded detection raises before any output
file exists. Successful correction and provenance behavior remain covered by the
existing tests.

After automated tests pass, run read-only validation on stratified existing
recordings:

- high alternating-polarity cases from sub-0005 and sub-0017;
- the long-gap sub-0011 run;
- a currently coherent sub-0003 run; and
- at least one ordinary-rate recording from another participant.

Report event count, selected polarity, median/minimum/maximum RR, degradation
reasons, and adjacent polarity alternation. The repair is acceptable only if it
removes alternating-deflection trains without damaging the coherent sub-0003
case. These recordings are validation evidence, not tracked test fixtures.

## Non-goals

This repair does not:

- modify or delete the existing BCG output cohort;
- change AAS templates, PCA-OBS behavior, correction windows, or the fixed
  ECG-to-BCG delay;
- claim Analyzer sensitivity or precision;
- add automatic fallback detection; or
- declare BCG correction scientifically validated.

After this repair produces trustworthy anchors, AAS boundary continuity,
correction coverage, method-specific immutable outputs, and held-out signal
transfer should be addressed as separate reviewed changes.
