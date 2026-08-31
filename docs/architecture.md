# Architecture

## Design principles

The production path has explicit boundaries: configuration is validated before
I/O, acquisition geometry is resolved before numerical processing, and outputs
are written only after all required checks pass. Modules expose small data
structures and functions with units visible in their names. Unexpected errors
are allowed to surface.

## Public API

Use [`fastr_python.api`](../src/fastr_python/api.py) for the stable,
configuration-driven interface:

```python
from fastr_python.api import load_config, run_correction

summary = run_correction(load_config("configuration.yml"))
```

The low-level array interface is [`fastr_python.fastr`](../src/fastr_python/fastr.py).
The package root exports only `__version__`; importing it does not eagerly load
MNE or the correction pipeline.

## Correction data flow

```text
YAML -> config -> timing/markers -> geometry -> channel batches ->
optional OBS/ANC -> output filter/decimation -> markers/QC/PSD/provenance
```

The pipeline accepts BrainVision headers and marker files, resolves one timing
source, constructs acquisition-group geometry, corrects channels in batches,
and writes a complete BrainVision output with a JSON provenance sidecar.

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

Production correction is the path through `config.py`, `pipeline.py`, the
BrainVision I/O modules, timing/geometry/processing modules, and pipeline
support modules. It imports neither simulation helpers nor comparison metrics.
The validation-only modules (`diagnostics.py`, `metrics.py`,
`matlab_comparison.py`, and `simulation.py`) support tests, demos, and audit
runners without becoming runtime dependencies of correction.

The top-level validation area retains these descriptive entry points:

- `validation/run_python_reference.py`: shared classical volume-stage contract;
- `validation/run_python_bids_reference.py`: production BIDS geometry path; and
- `validation/compare_fmrib_reference.py`: aggregate comparison metrics.

The original MATLAB reference is `validation/fmrib_reference.m`.

## Naming and units

Names state their domain and units where ambiguity is likely: BIDS timing is in
seconds, input and internal signal arrays use volts, residual reports use
microvolts, marker positions and geometry samples are zero-based internally,
and BrainVision marker positions are one-based on disk. `*_hz` names frequency
values; `*_seconds` names durations; `*_uv` names reported microvolt values.
