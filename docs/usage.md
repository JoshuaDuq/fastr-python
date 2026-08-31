# Usage

## Installation

Use Python 3.12. For the reproducible development environment:

```text
uv sync
```

For an existing compatible environment:

```text
uv pip install .
```

Record the installed package version with:

```text
fastr-python --version
```

## Synthetic demo

Generate a BrainVision recording, BIDS timing sidecar, and configuration:

```text
fastr-python demo --output-dir /path/to/demo
```

Correct that generated recording:

```text
fastr-python run --config /path/to/demo/demo.yml
```

The synthetic data exercises the software path only; it does not establish
performance on a scanner acquisition.

## Correct a recording

Copy [`examples/configuration.yml`](../examples/configuration.yml) for a
volume-marker recording or
[`examples/configuration-slice.yml`](../examples/configuration-slice.yml) for
acquisition-group markers. Set the paths and timing fields, then run:

```text
fastr-python run --config /path/to/configuration.yml
```

Paths in the YAML are resolved relative to the YAML file. Existing output files
are rejected. Invalid marker selection, timing, geometry, rates, or signal
inputs fail before an incomplete output is written.

## Validate timing before correction

For one marker per volume, provide BIDS metadata:

```text
fastr-python validate-timing \
  --metadata /path/to/bold.json \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Volume \
  --marker-description volume-start \
  --output /path/to/timing-validation.json
```

For one marker per acquisition group, provide the group count and optional TR
check instead:

```text
fastr-python validate-timing \
  --marker-kind slice \
  --groups-per-volume 18 \
  --expected-repetition-time-seconds 0.9 \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Slice \
  --marker-description slice-start \
  --output /path/to/timing-validation.json
```

The command writes resolved repetition time, groups per volume, group offsets,
and volume starts. It fails on missing or duplicate markers, non-increasing
markers, timing gaps, inconsistent periodicity, and invalid TR-to-sample
conversion. It does not infer scanner events from EEG waveforms.

## Compare folders

Declare both export naming conventions in `examples/compare.yaml`, then run:

```text
fastr-python compare --config examples/compare.yaml
```

The comparison pairs recordings only through the configured suffixes and
subject/run rules. It writes PSD overlays and CSV/JSON summaries. A load or
alignment error is reported for the affected pair; unexpected programming
errors remain visible.

## Python API

Use the high-level façade for a configuration-driven run:

```python
from fastr_python.api import load_config, run_correction

summary = run_correction(load_config("configuration.yml"))
print(summary.output_vhdr)
```

The low-level functions in `fastr_python.fastr` accept validated arrays and
geometry when a lab needs to integrate FASTR into a larger analysis system.

## Output files

For an output stem such as `corrected.vhdr`, the pipeline writes:

- `corrected.vhdr`, `corrected.eeg`, and `corrected.vmrk`;
- `corrected.json`, the provenance sidecar;
- `corrected_psd_before.png`; and
- `corrected_psd_after.png`.

The sidecar records the resolved configuration, input SHA-256 hashes, timing
source and geometry, output window, alignment, diagnostics, residual QC,
channel-policy decisions, and runtime. Read it alongside the corrected data.

## Failure behavior

Configuration, input, marker, timing, geometry, and output-collision errors are
reported with a nonzero CLI status. The loader does not silently choose between
conflicting timing sources or repair an ambiguous marker stream. Programming
errors are not converted into a successful-looking output.

## Before interpreting a corrected run

1. Confirm the sidecar names the intended input files and hashes.
2. Confirm timing and marker selection match the acquisition protocol.
3. Inspect raw and corrected data in time and frequency domains.
4. Measure scanner-locked residuals at `1 / RepetitionTime` and relevant
   harmonics.
5. Measure signal transfer independently of the correction template.
6. Review boundary groups, skipped spans, alignment values, and advisory
   channel recommendations.

A lower scanner line alone is not evidence of a better correction: neural signal
at the same frequency can be removed as well.
