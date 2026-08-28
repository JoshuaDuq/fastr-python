# FASTR-Python

FASTR-Python corrects scanner-gradient artifacts in simultaneous EEG-fMRI
recordings. It accepts a BrainVision recording and BIDS fMRI timing metadata,
validates the acquisition markers, applies acquisition-slot FASTR correction, and
writes a corrected BrainVision recording with preserved markers and provenance.

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

## Correct a recording

Copy [`examples/configuration.yml`](examples/configuration.yml), set the paths and
processing values for the recording, then run:

```text
mri-correct run --config /path/to/configuration.yml
```

All important run settings are defined in YAML rather than hardcoded. The strict
configuration rejects ambiguous markers, invalid timing, unsuitable filters, and
existing output files. The run produces BrainVision files, before/after PSD figures,
and a JSON provenance sidecar.

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
following the [FMRIB fMRIb FASTR implementation](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m)
and [Niazy et al. (2005)](https://pubmed.ncbi.nlm.nih.gov/16150610/). See
[`docs/algorithm.md`](docs/algorithm.md) for the processing model, limitations, and
validation details. Residual OBS and adaptive noise cancellation are not part of
the pipeline.

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
(`bcgnet`). FASTR-Python only removes scanner-gradient artifacts.
