# EEG-fMRI FASTR

`eegfmri-fastr` is a Python implementation of the FASTR scanner-gradient artifact
correction method for simultaneous EEG-fMRI recordings. It accepts a BrainVision
recording and BIDS fMRI timing metadata, validates the acquisition markers, applies
acquisition-slot FASTR correction, and writes a corrected BrainVision recording
with preserved markers and provenance.

This is research software. Inspect the provenance and validate the correction for
each acquisition protocol before using the output for inference.

It is not the FMRIB EEGLAB plug-in and is not affiliated with, sponsored by, or
endorsed by the FMRIB Centre or the University of Oxford. The project is released
under GPL-2.0-only; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

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

## Correct a recording

Copy [`examples/configuration.yml`](examples/configuration.yml), set the paths and
processing values for the recording, then run:

```text
mri-correct run --config /path/to/configuration.yml
```

All important run settings are defined in YAML rather than hardcoded. The strict
configuration rejects ambiguous markers, invalid timing, unsuitable filters, and
existing output files. The run produces BrainVision files, before/after PSD figures,
and a JSON provenance sidecar. `line_noise_frequencies_hz` is required: use
`[60.0]` for explicit 60 Hz sinusoidal regression or `[]` to retain every line.

## Compare uncorrected vs FASTR folders

Pair an uncorrected scanner-artifact folder with FASTR-corrected recordings
and write PSD/epoch overlays plus a CSV of band-power ratios:

```text
mri-correct compare --config examples/compare.yaml
```

The example compares
`step1_scanner_artifact_pulse_marked` with `fastr_python`. Matching uses the
shared recording stem (`…_fastr` vs `…_first_to_last_volume_scanner_artifact_with_pulse_markers`).

## Validate timing only

Validate configured volume markers against BIDS timing metadata before correction:

```text
mri-correct validate-timing \
  --metadata /path/to/bold.json \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Volume \
  --marker-description volume-start \
  --output /path/to/timing-validation.json
```

The command fails on missing or duplicate markers, marker gaps, excessive timing
jitter, or an inconsistent TR-to-sample conversion. It does not infer markers from
the EEG waveform.

## Method

The pipeline implements the acquisition-group variant of FASTR for multiband data,
following the published method and using the
[FMRIB fMRIb FASTR implementation](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m)
as a reference. It is a separate Python implementation, not a drop-in port of the
FMRIB plug-in. See [`docs/algorithm.md`](docs/algorithm.md) for the processing model,
limitations, and validation details. Residual OBS is available separately and
adaptive noise cancellation is not implemented in the pipeline.

## Development

Run the test and lint checks:

```text
uv run pytest
uv run ruff check src tests
git diff --check
```

See [`docs/validation.md`](docs/validation.md) for the validation checklist.

## Related pipelines

Cardiac detection and AAS/PCA-OBS BCG correction live in **BCG-Correction**
(`bcg-correct`). The deep-learning BCGNet path lives in **BCGNet-Python**
(`bcgnet`). This package only removes scanner-gradient artifacts.
