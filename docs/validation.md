# Validation checklist

Validation has two parts: deterministic software checks and protocol-specific
signal evidence. Passing the test suite establishes software contracts, not
correction quality for every scanner, acquisition, montage, or analysis.

## Software checks

Run from the repository root:

```text
uv run pytest
uv run ruff check src tests validation
git diff --check
uv build
```

The suite covers configuration, BrainVision I/O, BIDS and acquisition-group
timing, geometry, channel-batch invariance, template/OBS/ANC stages, filtering,
provenance, examples, folder comparison, and deterministic synthetic data.
Validation and comparison helpers are not runtime dependencies of correction.

## Protocol-specific signal checks

Before a study run, use a representative recording from the target scanner,
marker convention, sampling rate, montage, and task. Confirm:

- marker type and description select the intended events;
- timing matches the acquisition protocol;
- volume spacing and within-volume offsets are stable;
- the output rate and filter preserve the required frequency range;
- non-EEG channels are identified; and
- any marker repair or explicit block selection has a documented reason.

Run `validate-timing` when timing has not already been audited. It does not
infer timing from the EEG waveform.

## Run-level checks

For each corrected run:

1. Check input paths and hashes in the provenance sidecar.
2. Check timing, marker counts, output window, and skipped boundary groups.
3. Inspect raw and corrected signals in time and frequency domains.
4. Measure residual amplitude or power at `1 / RepetitionTime` and relevant
   harmonics, treating mains collisions explicitly.
5. Measure signal transfer with injected tones, known physiological features,
   or an independent reference not used to build the template.
6. Review alignment, residual-QC blocks, and channel recommendations before
   downstream rejection or interpolation.

## Comparison to another correction implementation

Use the [FMRIB parity audit](fmrib-parity-validation.md) and its runners when a
reference is available. Compare identical channels, sample ranges, markers,
timing, filters, output rates, and metrics. Keep private recordings and
generated outputs outside the repository.

The runners are `run_python_reference.py`,
`run_python_bids_reference.py`, and `compare_fmrib_reference.py`. They require
explicit paths and refuse to overwrite outputs.

## Interpreting residual suppression and signal transfer

Lower scanner-harmonic power is not sufficient evidence of a better correction:
template subtraction, OBS, and ANC can remove neural signal at the same
frequency. Report residual suppression with signal transfer, and inspect
off-comb and on-comb probes separately. Channel-failure output is advisory; it
does not delete or interpolate channels.

## Reproducibility record

Keep these with each result:

- `fastr-python --version` output;
- exact YAML configuration;
- provenance JSON and input hashes;
- scanner, sequence, sampling rate, montage, and marker details;
- timing-validation output; and
- residual and signal-transfer measures used for interpretation.

The [FMRIB parity validation](fmrib-parity-validation.md) labels
project-generated evidence with its dataset scope. It is not a general
performance guarantee.
