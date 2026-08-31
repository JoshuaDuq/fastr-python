# Validation checklist

FASTR validation has two parts: deterministic software checks and
protocol-specific signal evidence. Passing the test suite establishes software
contracts; it does not establish that a correction is appropriate for every
scanner, acquisition, montage, or downstream analysis.

## Software checks

Run these commands from the repository root:

```text
uv run pytest
uv run ruff check src tests validation
git diff --check
uv build
```

The suite covers configuration and cross-field validation, strict BrainVision
marker I/O, BIDS and acquisition-group timing, geometry, channel-batch
invariance, template/OBS/ANC stages, output filtering, provenance, examples,
folder comparison, and deterministic synthetic data. The comparison and
validation helpers are not runtime dependencies of the production correction
path.

## Protocol-specific signal checks

Before a study run, use a representative recording from the intended scanner,
marker convention, sampling rate, montage, and task. Confirm:

- the configured marker type and description select the intended events;
- the timing source agrees with the acquisition protocol;
- volume spacing and within-volume offsets are stable;
- the output rate and filter preserve the frequency range needed downstream;
- non-EEG channels are explicitly identified; and
- any marker repair or explicit marker block selection has a documented reason.

Run `validate-timing` before correction when timing has not already been
audited. The command does not infer timing from the EEG waveform.

## Run-level checks

For every corrected run:

1. Confirm the provenance sidecar identifies the intended input paths and
   hashes.
2. Confirm resolved timing, marker counts, output window, and skipped boundary
   groups match the protocol.
3. Inspect raw and corrected signals in time and frequency domains.
4. Measure scanner-locked residual amplitude or power at `1 / RepetitionTime`
   and relevant harmonics, with mains collisions treated explicitly.
5. Measure signal transfer using injected tones, known physiological features,
   or an independent reference not used to construct the template.
6. Review alignment values, residual-QC blocks, and advisory channel
   recommendations before downstream rejection or interpolation.

## Comparison to another correction implementation

Use the [FMRIB parity audit](fmrib-parity-validation.md) and its runners when a
reference implementation is available. Compare the same channels, sample
range, markers, timing, filter, output rate, and metrics. Keep private
recordings and generated outputs outside the tracked repository.

The top-level runners retain their current names:
`run_python_reference.py`, `run_python_bids_reference.py`, and
`compare_fmrib_reference.py`. They require explicit paths and refuse to
overwrite outputs.

## Interpreting residual suppression and signal transfer

A lower scanner line is not sufficient evidence of a better correction. Neural
signal at the same frequency can be removed by template subtraction, OBS, or
ANC. Report residual suppression together with signal-transfer measurements;
inspect off-comb and on-comb probes separately. Treat the automatic channel
failure policy as an advisory recommendation, not as an automatic channel
deletion or interpolation decision.

## Reproducibility record

Keep the following with each result:

- installed `eegfmri-fastr --version` output;
- the exact YAML configuration;
- the provenance JSON sidecar and input hashes;
- scanner, sequence, sampling-rate, montage, and marker details;
- timing-validation output; and
- the residual and signal-transfer measures used for interpretation.

The detailed [FMRIB parity validation](fmrib-parity-validation.md) labels
project-generated evidence with its dataset scope. It is not a general
performance guarantee.
