# Usage Guide

This guide covers installation, command-line operations, Python API workflows, pre-flight timing validation, output file inspection, and pre-interpretation scientific checks.

---

## 1. Environment & Installation

FASTR-Python requires **Python 3.12**.

### Recommended: Locked Virtual Environment via `uv`

For scientific reproducibility, use [`uv`](https://docs.astral.sh/uv/) to synchronize the exact dependency tree pinned in `uv.lock`:

```bash
uv sync --locked
```

### Existing Virtual Environment

To install into an existing active Python 3.12 environment:

```bash
uv pip install .
# or standard pip:
pip install .
```

Verify your installation and runtime environment:

```bash
fastr-python --version
```

---

## 2. Quickstart: Synthetic Pipeline Verification

FASTR-Python includes a deterministic synthetic generator to verify your software environment and complete execution path without needing scanner data:

```bash
# 1. Generate synthetic BrainVision recording, BIDS sidecar, and YAML config
fastr-python demo --output-dir /tmp/fastr_demo

# 2. Execute gradient artifact correction
fastr-python run --config /tmp/fastr_demo/demo.yml
```

The demo simulates an 18-channel EEG recording containing scanner gradient switching, an injected 10.5 Hz neural signal, and BrainVision markers.

> [!NOTE]
> The demo validates the end-to-end numerical and I/O pipeline. It is an automated software verification step, not scanner- or protocol-specific validation.

---

## 3. Correcting a Real BrainVision Recording

Production correction runs are driven by YAML configurations. Template configurations are provided in `examples/`:

- [`examples/configuration.yml`](../examples/configuration.yml): For volume-triggered recordings (one scanner trigger per whole volume).
- [`examples/configuration-slice.yml`](../examples/configuration-slice.yml): For slice- or group-triggered recordings (one scanner trigger per acquisition slice/multiband group).

Run the correction:

```bash
fastr-python run --config /path/to/configuration.yml
```

### Key Execution Behaviors

1. **Relative Paths**: All relative paths inside the configuration YAML resolve relative to the directory containing that YAML file.
2. **Fail-Fast Safety**: If destination output files already exist, or if marker selection, timing, sampling rates, or channel labels are invalid, FASTR-Python raises an explicit error before any file is created or partially modified.
3. **No Hidden State**: Each run is completely self-contained and reproducible from the configuration and raw inputs alone.

---

## 4. Validate Timing Before Correction

Scanner triggers are prone to missing pulses, jitter, or timing-source ambiguities. Run the pre-flight `validate-timing` CLI command to verify timing integrity before launching a correction:

### Volume Markers (BIDS Timing Source)

```bash
fastr-python validate-timing \
  --metadata /path/to/bold.json \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Volume \
  --marker-description volume-start \
  --output /path/to/timing-validation.json
```

### Slice / Acquisition-Group Markers

```bash
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

### What `validate-timing` Checks

- Marker ordering (strictly monotonically increasing sample indices).
- Duplicate triggers or missing interior pulses.
- Periodic interval stability and alignment with declared $T_R$.
- Integer conversion of repetition time and group offsets to input sampling samples.
- Complete artifact epochs across the entire recording.

> [!IMPORTANT]
> `validate-timing` verifies the logged event markers against declared sequence parameters. It does **not** attempt to guess scanner events from EEG waveforms.

---

## 5. Cohort Comparison Tool

To evaluate correction performance across multiple runs or compare two processing pipelines (e.g. raw vs corrected, or MATLAB FMRIB vs Python FASTR):

```bash
fastr-python compare --config examples/compare.yaml
```

Configure `examples/compare.yaml` with your directory pairs, filename suffixes, and channels of interest. The command outputs:
- Multi-channel PSD overlay plots;
- Summary CSV tables of scanner-harmonic RMS residual suppression; and
- JSON metrics recording broadband signal transfer and ECG preservation.

---

## 6. Python API

FASTR-Python provides two stable API entrypoints: a high-level configuration-driven API and a low-level array API.

### High-Level API: `fastr_python.api`

Recommended for analysis scripts, workflow managers (e.g., Snakemake, Nextflow), and BIDS pipelines:

```python
from pathlib import Path
from fastr_python.api import load_config, run_correction

# 1. Load and strictly validate YAML configuration into a frozen dataclass
config = load_config(Path("study_config.yml"))

# 2. Execute end-to-end correction
summary = run_correction(config)

# 3. Access summary metadata
print("Correction Status:", summary.status)
print("Emitted VHDR:", summary.output_vhdr)
print("Provenance JSON:", summary.provenance_json)
print("Volumes processed:", summary.processed_volumes)
```

### Low-Level Array API: `fastr_python.fastr`

For researchers integrating FASTR into custom pipelines operating directly on in-memory NumPy arrays:

```python
import numpy as np
from fastr_python.fastr import apply_fastr_batch, slice_fastr

# Apply low-level FASTR template subtraction on validated numeric arrays
# (refer to docstrings in fastr_python.fastr for exact argument schemas)
```

---

## 7. Output Artifacts & Provenance Sidecar

Given an output target `output.vhdr = "sub-01_corrected.vhdr"`, each run produces the following companion files in the target folder:

| File Pattern | Description & Format |
| --- | --- |
| `sub-01_corrected.vhdr` | BrainVision Core Data Format header file. |
| `sub-01_corrected.eeg` | Binary multiplexed float32 / int16 signal data. |
| `sub-01_corrected.vmrk` | BrainVision marker file containing resampled scanner markers and `Bad_Gradient` annotations. |
| `sub-01_corrected.json` | Comprehensive JSON provenance sidecar. |
| `sub-01_corrected_psd_before.png` | Welch power spectral density estimate before correction. |
| `sub-01_corrected_psd_after.png` | Welch power spectral density estimate after correction. |

### Anatomy of the Provenance Sidecar (`.json`)

The JSON sidecar contains exhaustive cryptographic and scientific metadata:
1. **Input Hashes**: SHA-256 hashes of input `.vhdr`, `.eeg`, `.vmrk`, and BIDS JSON files.
2. **Software Environment**: Exact version numbers for `fastr-python`, Python, NumPy, SciPy, MNE-Python, and pybv.
3. **Timing & Geometry**: Resolved repetition time, group offsets, volume start indices, and sub-sample interpolation factors.
4. **Processing Configuration**: Complete snapshot of all parameters used.
5. **Quality Control Summary**:
   - Temporal block residual excess in $\mu\mathrm{V}$;
   - Volume harmonic RMS across multiples of $1/T_R$;
   - Channel outlier recommendations; and
   - Flagged boundary groups or uncorrected regions.

---

## 8. Failure Behavior & Error Handling

FASTR-Python enforces a strict **fail-fast, no hidden fallback** design philosophy:

- **Invalid Configuration**: Unknown keys, missing fields, or invalid types raise `ConfigurationError` immediately.
- **Ambiguous Timing**: Specifying conflicting timing sources (e.g. both BIDS metadata and slice marker count) raises an error.
- **Output Collision**: Existing files matching target stems are never overwritten silently.
- **Marker Inconsistencies**: Non-monotonic markers, unexpected gaps, or boundary truncations without declared repair policies halt execution.
- **Divergence Guard**: If adaptive LMS filters encounter non-finite weights or numerical divergence, execution aborts with a diagnostic trace.

---

## 9. Pre-Interpretation Checklist

Before using corrected EEG data in downstream scientific analyses (ERP, frequency bands, connectivity, or BIDS export):

1. **Verify Provenance**: Open `corrected.json` and confirm SHA-256 input hashes match your source files.
2. **Inspect PSD Plots**: Compare `corrected_psd_before.png` and `corrected_psd_after.png` to confirm gradient comb peaks are suppressed without wiping out non-harmonic physiological bands.
3. **Examine Comb Harmonics**: Check `residual_qc.volume_harmonic_qc` in the JSON sidecar. Ensure no persistent residual bursts remain during task periods.
4. **Evaluate Broadband Signal Transfer**: Verify that frequencies between gradient harmonics maintain near-unity transfer.
5. **Review Boundary Markers**: Identify any samples annotated with `Bad_Gradient` and mask or reject them in downstream epoching.
6. **Check ECG Channel**: If recorded, verify that the ECG waveform preserved QRS morphology without distortion.

