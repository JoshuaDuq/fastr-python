# BCG PSD Diagnostics Design

## Goal

Make the independent BCG correction pipeline write before/after PSD figures with
the same diagnostic contract as the FASTR pipeline.

## Behavior

`run_bcg_correction()` will write two PNG files beside the requested BrainVision
header:

- `<output stem>_psd_before.png`
- `<output stem>_psd_after.png`

Both figures will use MNE's raw PSD renderer, a 0--100 Hz frequency range,
standard EEG spatial colors when channel positions can be resolved, and the same
time samples. The common sample set is exactly the samples changed by the BCG
splice. Samples between separate corrected BCG windows are excluded from the PSD
calculation with temporary `bad_psd_gap` annotations, so untouched recording
segments cannot dominate the comparison.

The `BcgCorrectionSummary` will expose both paths. The `.bcg.json` sidecar will
record both paths and the common PSD interval in seconds. The PNG paths will be
included in the output-existence check, so a partially existing diagnostic run
fails before writing any output.

## Architecture

The existing FASTR PSD renderer and channel-preparation logic will move to a
small shared `mri_correction.psd` module. The shared functions will retain the
current FASTR call behavior and titles. FASTR will import them without changing
its interval calculation or output naming.

BCG will build a pre-correction MNE `Raw` copy before closing the input reader and
will read the written corrected recording for the after figure. A BCG-specific
helper will validate `corrected_samples`, find its first and last sample, and add
temporary bad annotations for all gaps within that bounding interval. The shared
renderer will receive both prepared copies with the same `tmin` and `tmax`.

## Error handling

The BCG PSD interval must contain at least two valid samples and must be strictly
ordered. Invalid or empty corrected coverage will raise the existing BCG input
error rather than silently plotting the full recording. Unexpected MNE or file
errors will remain visible to the caller.

## Testing

Add an end-to-end BCG pipeline test that verifies the two PNGs, summary paths,
sidecar paths, and a valid common PSD interval. Add a focused renderer-call test
that captures both calls, checks identical time bounds, and verifies that the
before/after PSD copies carry the temporary bad-gap annotations. Preserve the
existing FASTR PSD tests and run the complete test and lint suites.
