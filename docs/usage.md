# Usage

## Installation

FASTR-Python supports Python 3.12. For the locked development environment:

```text
uv sync
```

For an existing compatible environment:

```text
uv pip install .
```

Check the installed version:

```text
fastr-python --version
```

## Synthetic demo

Generate a BrainVision recording, BIDS timing sidecar, and configuration, then
run the correction:

```text
fastr-python demo --output-dir /path/to/demo
fastr-python run --config /path/to/demo/demo.yml
```

The synthetic data tests the software path; it is not scanner validation.

## Correct a recording

Use [`configuration.yml`](../examples/configuration.yml) for volume markers or
[`configuration-slice.yml`](../examples/configuration-slice.yml) for
acquisition-group markers. Set the paths and timing fields, then run:

```text
fastr-python run --config /path/to/configuration.yml
```

Paths resolve relative to the YAML file. Existing outputs, invalid marker
selection, timing, geometry, rates, or signal inputs cause an error before an
incomplete output is written.

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
check:

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

The command writes resolved repetition time, group count, offsets, and volume
starts. It fails on missing or duplicate markers, non-increasing markers,
timing gaps, inconsistent periodicity, and invalid TR-to-sample conversion. It
does not infer scanner events from EEG waveforms.

## Compare folders

Set both export naming conventions in `examples/compare.yaml`:

```text
fastr-python compare --config examples/compare.yaml
```

Pairing uses only the configured suffixes and subject/run rules. The command
writes PSD overlays and CSV/JSON summaries. Pair-specific load or alignment
errors are reported; unexpected programming errors remain visible.

## Python API

Use the high-level API for a configuration-driven run:

```python
from fastr_python.api import load_config, run_correction

summary = run_correction(load_config("configuration.yml"))
print(summary.output_vhdr)
```

The low-level functions in `fastr_python.fastr` accept validated arrays and
geometry for integration into a larger analysis system.

## Output files

For `corrected.vhdr`, outputs are:

- `corrected.vhdr`, `corrected.eeg`, and `corrected.vmrk`;
- `corrected.json`, the provenance sidecar; and
- `corrected_psd_before.png` and `corrected_psd_after.png`.

The sidecar records configuration, input hashes, timing and geometry, output
window, alignment, diagnostics, residual QC, channel decisions, and runtime.
Read it with the corrected data.

## Failure behavior

Configuration, input, marker, timing, geometry, and output-collision errors
return a nonzero CLI status. Conflicting timing sources and ambiguous marker
streams are not silently resolved. Programming errors remain visible.

## Before interpreting a corrected run

1. Check input paths and hashes in the sidecar.
2. Check timing, marker selection, output window, and skipped groups.
3. Inspect raw and corrected signals in time and frequency domains.
4. Measure residuals at `1 / RepetitionTime` and relevant harmonics.
5. Measure signal transfer independently of the correction template.
6. Review alignment, skipped spans, and channel recommendations.

Lower scanner-harmonic power alone is not evidence of a better correction;
neural signal at the same frequencies may also be removed.
