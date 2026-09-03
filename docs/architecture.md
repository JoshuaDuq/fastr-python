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

## Package responsibilities

```text
fastr_python/
├── api.py                 stable high-level API
├── fastr.py               stable low-level array API
├── cli.py                 command-line boundary
├── config/                YAML decoding, models, schema, and validation
├── correction/            numerical FASTR algorithms and timing geometry
├── io/                    BrainVision markers and recordings
├── pipeline/              correction orchestration and provenance
├── quality/               PSD and residual-quality measurements
├── validation/            simulation, metrics, and reference comparisons
└── compare/               folder-level corrected/uncorrected comparison
```

The domain packages are dependency boundaries, not alternative public APIs.
External callers should import from `fastr_python.api` or
`fastr_python.fastr`; internal modules may change as responsibilities become
clearer.

The pipeline package separates acquisition resolution, channel processing,
recording I/O, marker handling, quality measurements, provenance, and the run
coordinator. The correction package owns numerical algorithms and has no
configuration or file-writing responsibilities.

## Production versus validation code

Production correction uses `config`, `correction`, `io`, `pipeline`, and
`quality`. It does not import the simulation or reference-comparison helpers in
`fastr_python.validation`; those serve tests, demos, and audit runners.

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
