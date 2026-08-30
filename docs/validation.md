# Validation checklist

The EEG-fMRI FASTR implementation is intended to be validated at two levels:
deterministic software
checks and protocol-specific signal checks.

## Software checks

Run the complete automated suite and static checks from the repository root:

```text
uv run pytest
uv run ruff check src tests validation
git diff --check
```

The tests cover:

- strict YAML structure and scalar validation;
- exact BrainVision marker selection and lossless marker round trips;
- BIDS timing, TR-sample, marker-gap, boundary, and explicit repair validation;
- one source of acquisition timing, whichever route declares it;
- measuring geometry from acquisition-group markers, and the equivalence of the
  two marker conventions over one recording;
- shared FASTR alignment and channel-batch invariance;
- fixed and automatic sectioned OBS, FMRIB LMS ANC, and stage ordering;
- output-rate, filter, and output-collision checks;
- safe disabled-low-pass behavior without decimation;
- pairing two folders under a declared naming convention;
- correcting the generated demo dataset end to end, checking that the artifact
  falls and the off-comb probe tone survives;
- reopening the generated BrainVision recording with MNE; and
- generation of before/after MNE PSD figures.

`eegfmri-fastr demo --output-dir DIR` reproduces that end-to-end check by hand,
on a dataset that needs no recording of your own.

The BCG-Correction package (AAS/PCA-OBS) additionally verifies that the FASTR input contains no
pre-existing `Pulse Artifact,R` markers, detects R samples from ECG only, preserves the
source marker collection while appending detector markers, preserves ECG and all samples
outside bounded correction windows, and compares methods with held-out cardiac residuals
and circular-shift nulls. Analyzer marker agreement is an audit measure, not detector
ground truth. Production correction also rejects degraded ECG trains and any
heartbeat-locked after/before RMS ratio above `maximum_residual_ratio` before
writing output.

## Run-level checks

Before interpreting a corrected run:

1. Confirm the provenance sidecar hashes and resolved configuration identify the
   intended input files.
2. Confirm that the declared volume-marker series has no gaps or unexplained timing
   changes.
3. Inspect raw, corrected, and independently corrected reference data in both time
   and frequency domains.
4. Quantify scanner-locked residuals at `1 / RepetitionTime` and its relevant
   harmonics using absolute amplitude or power as well as relative band ratios.
5. Quantify signal transfer using synthetic injected tones, known physiological
   features, or another reference that is not used to construct the correction
   template.
6. Review skipped boundary groups and fitted alignment correlations in the sidecar.

Do not treat a lower residual line as sufficient evidence of a better correction:
an algorithm can lower a line by removing neural signal at the same frequency. The
most informative comparison reports both residual artifact suppression and signal
preservation.

## Comparison to another correction implementation

An independent implementation may be used as a benchmark oracle, but it should not
become a runtime dependency or a hidden default. Compare identical input channels,
sample ranges, marker definitions, filters, output rates, and quality metrics. Keep
private recordings and generated outputs outside the tracked public package. The
reusable oracle runners and aggregate evidence are documented in
[`fmrib-parity-validation.md`](fmrib-parity-validation.md).
For the cardiac/BCG comparison, use the FASTR gradient-corrected recording derived from
the raw unmarked stage as the own-method input. Do not use an Analyzer pulse-marked or
BrainVision Analyzer-corrected file as that input. Supply Analyzer's pre-BCG input and
post-BCG output separately so each correction arm is scored against its own pre-correction
baseline. Verify the paired recordings have the same channel order, compatible sampling
rate and sample geometry, and high interior ECG correlation before interpreting method
differences. Analyzer marker agreement is not ground truth when Analyzer is known to miss
beats.

BCG/ECG detection and correction are outside this package. The EEG-fMRI FASTR
implementation retains cardiac validation metrics and deterministic simulation
helpers so those downstream checks can be run without making BCG-Correction a
runtime dependency.
