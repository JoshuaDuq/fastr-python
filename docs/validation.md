# Validation checklist

FASTR-Python is intended to be validated at two levels: deterministic software
checks and protocol-specific signal checks.

## Software checks

Run the complete automated suite and static checks from the repository root:

```text
uv run pytest
uv run ruff check src tests
git diff --check
```

The tests cover:

- strict YAML structure and scalar validation;
- exact BrainVision marker selection and lossless marker round trips;
- BIDS timing, TR-sample, marker-gap, and boundary validation;
- shared FASTR alignment and channel-batch invariance;
- output-rate, filter, and output-collision checks; and
- reopening the generated BrainVision recording with MNE; and
- generation of before/after MNE PSD figures.

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
the benchmark harness and private recordings outside the tracked public package.
