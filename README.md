# FASTR-Python

FASTR-Python is a configuration-driven Python tool for scanner-gradient artifact
correction in simultaneous EEG-fMRI recordings. It accepts a BrainVision recording
and BIDS fMRI timing metadata, validates the acquisition markers, applies an
acquisition-slot FASTR correction, and writes a BrainVision recording with preserved
markers plus a provenance sidecar.

This is research software. Inspect the provenance and validate the correction for
each acquisition protocol before using the output for inference.

## Installation

The project supports Python 3.12. Install the runtime and development dependencies
with `uv`:

```text
uv sync
```

Or install the package into an existing environment:

```text
uv pip install .
```

## Run a correction

Copy [`examples/configuration.yml`](examples/configuration.yml), edit every path and
processing value for the recording, then run:

```text
mri-correct run --config /path/to/configuration.yml
```

The configuration is deliberately strict. Marker type and description are exact
matches; input and output paths are explicit; and existing output files, timing
gaps, non-integer rate conversions, and invalid filter settings fail rather than
being repaired implicitly.

The output consists of the requested `.vhdr`, its `.eeg` and `.vmrk` companions,
`*_psd_before.png`, `*_psd_after.png`, and a `.json` provenance sidecar. The PSD
figures are generated with MNE's `mne.viz.plot_raw_psd`, use the same complete-epoch
interval for both conditions, and show 0--100 Hz with spatial channel colors when
standard EEG positions can be identified. Boundary groups without complete FASTR
epochs are excluded from both PSD figures and recorded in the sidecar. The sidecar
also records resolved settings, source hashes, timing validation, alignment shifts,
and fitted amplitudes.

## Validate timing only

For a BrainVision recording, validate the configured volume markers against the BIDS
timing metadata before running correction:

```text
mri-correct validate-timing \
  --metadata /path/to/bold.json \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Volume \
  --marker-description volume-start \
  --output /path/to/timing-validation.json
```

This command fails on missing markers, duplicate positions, marker gaps, timing
jitter beyond the declared tolerance, or an inconsistent TR-to-sample conversion.
It does not infer missing markers from the EEG waveform.

## Method

The public pipeline uses the acquisition-group variant of FASTR for multiband data:
the validated BIDS slice timing determines repeated acquisition-time slots, and
templates are formed from neighboring volumes in the same slot while excluding the
target group. Alignment and the least-squares template are estimated from a 1 Hz
high-passed copy of each channel (Niazy et al. 2005 stage 2); the fitted artifact is
subtracted from the unfiltered recording. See [`docs/algorithm.md`](docs/algorithm.md)
for the processing model, the stages that are not run (residual OBS and ANC), and
the 1/TR limitation.

The default example uses 60 neighboring volumes for the acquisition-slot template.
This is a validated starting point, not a protocol-independent guarantee; tune and
report it only through the YAML configuration and compare results with appropriate
signal-preservation controls. Set `trim.mode` to `first_to_last_volume` on untrimmed
recordings so boundary volumes keep the margin FASTR needs; uncorrected spans and
blocks over `residual_threshold_uv` are annotated `Bad Interval, Bad_Gradient`.

## Development

Run the complete test and lint checks:

```text
uv run pytest
uv run ruff check src tests
git diff --check
```

The public test suite contains synthetic BrainVision round trips, strict
configuration and marker validation, FASTR geometry/alignment tests, batch-size
invariance checks, and end-to-end pipeline tests. See
[`docs/validation.md`](docs/validation.md) for a validation checklist.

Analyzer comparison code and private benchmark data are intentionally kept outside
the tracked public package in `.local/analyzer_comparison/`.

## Related pipelines

Cardiac detection and AAS/PCA-OBS BCG correction now live in **BCG-Python**
(`bcg-correct`). The deep-learning BCGNet path lives in **BCGNet-Python**
(`bcgnet`). This package only removes scanner-gradient artifact.
