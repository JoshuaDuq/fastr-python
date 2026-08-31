# Architecture

## Principles

Validate configuration before I/O, resolve geometry before numerical processing,
and write outputs only after required checks pass. Modules expose small data
structures and functions; names include units where needed. Unexpected errors
surface.

## Public API

Use [`fastr_python.api`](../src/fastr_python/api.py) for the stable,
configuration-driven interface:

```python
from fastr_python.api import load_config, run_correction

summary = run_correction(load_config("configuration.yml"))
```

The low-level array interface is [`fastr_python.fastr`](../src/fastr_python/fastr.py).
The package root exports only `__version__` and does not eagerly load MNE or the
pipeline.

## Correction data flow

```text
YAML -> config -> timing/markers -> geometry -> channel batches ->
optional OBS/ANC -> output filter/decimation -> markers/QC/PSD/provenance
```

FASTR reads BrainVision headers and markers, resolves one timing source,
constructs acquisition-group geometry, corrects channels in batches, and writes
BrainVision output with a JSON provenance sidecar.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package version and root export boundary. |
| `api.py` | Stable high-level configuration-driven re-exports. |
| `brainvision.py` | Strict BrainVision marker parsing and writing. |
| `brainvision_io.py` | BrainVision file discovery, marker selection, and recording output. |
| `cli.py` | Command-line parsing and operator-facing commands. |
| `compare/__init__.py` | Comparison package boundary. |
| `compare/config.py` | YAML configuration for folder comparisons. |
| `compare/pairs.py` | Declared naming-based recording pairing. |
| `compare/pipeline.py` | Comparison orchestration and summary writing. |
| `compare/plots.py` | MNE loading, alignment checks, PSD overlays, and comparison metrics. |
| `config.py` | YAML schema, defaults, paths, and cross-field validation. |
| `demo.py` | Reproducible synthetic BrainVision dataset generation. |
| `diagnostics.py` | Signal diagnostics and acquisition-period candidates. |
| `fastr.py` | Low-level FASTR façade and public array API. |
| `fastr_anc.py` | Normalized LMS adaptive noise cancellation. |
| `fastr_geometry.py` | Epoch geometry, alignment, and adaptive-window decisions. |
| `fastr_processing.py` | Array-level template, OBS, and channel-batch processing. |
| `fastr_templates.py` | Acquisition-slot template construction. |
| `fastr_timing.py` | BIDS timing, marker geometry, and timing validation. |
| `fastr_types.py` | Low-level immutable correction and geometry data structures. |
| `fastr_validation.py` | Shared shape, range, and parameter validation. |
| `markers.py` | BrainVision marker/sample coordinate transformations. |
| `matlab_comparison.py` | Array and MATLAB-reference comparison helpers. |
| `metrics.py` | Signal-transfer, residual, and comparison metrics. |
| `pipeline.py` | Configuration-driven correction orchestration and summary. |
| `pipeline_io.py` | Pipeline input/output paths, rates, filters, and channels. |
| `pipeline_markers.py` | Output annotations for residual and skipped regions. |
| `pipeline_provenance.py` | JSON-safe provenance assembly and file hashes. |
| `pipeline_types.py` | Pipeline-facing input and channel-policy data structures. |
| `psd.py` | Before/after PSD preparation and plotting. |
| `residual_qc.py` | Residual harmonic measurements and advisory channel decisions. |
| `simulation.py` | Deterministic synthetic EEG-fMRI signal generation. |
| `window.py` | Output-span resolution and trim validation. |

## Production versus validation code

Production correction uses `config.py`, `pipeline.py`, the BrainVision I/O
modules, timing/geometry/processing modules, and pipeline support modules. It
does not import simulation or comparison helpers. The validation-only modules
(`diagnostics.py`, `metrics.py`, `matlab_comparison.py`, and `simulation.py`)
serve tests, demos, and audit runners.

The validation area contains:

- `validation/run_python_reference.py`: shared classical volume-stage contract;
- `validation/run_python_bids_reference.py`: production BIDS geometry path; and
- `validation/compare_fmrib_reference.py`: aggregate comparison metrics.

The original MATLAB reference is `validation/fmrib_reference.m`.

## Naming and units

Names state domain and units where needed: BIDS timing uses seconds, signal
arrays use volts, residual reports use microvolts, internal samples are
zero-based, and BrainVision marker positions are one-based. `*_hz`,
`*_seconds`, and `*_uv` identify frequencies, durations, and microvolt values.
