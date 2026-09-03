# FASTR-Python

FASTR-Python is the Python version of [FMRIB FASTR](https://github.com/sccn/fMRIb)
for correcting scanner-gradient artifacts in simultaneous EEG-fMRI recordings.

## What it does

`fastr-python` corrects scanner-gradient artifact in BrainVision EEG-fMRI
recordings. It validates acquisition timing, applies acquisition-slot FASTR,
preserves markers, and writes provenance. BCG, bad-electrode, motion, and other
physiological artifacts are outside its scope.

The package is `fastr_python`; the command is `fastr-python`; and the stable
API is [`fastr_python.api`](src/fastr_python/api.py). This is separate Python
software from the FMRIB EEGLAB plug-in and is not affiliated with or endorsed
by the FMRIB Centre or the University of Oxford. It is released under
GPL-2.0-only; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Installation

Python 3.12 is the tested interpreter. For development, install the locked
environment with:

```text
uv sync
```

For an existing compatible environment:

```text
uv pip install .
```

## Quickstart

Generate a self-contained synthetic dataset and correct it:

```text
fastr-python demo --output-dir /path/to/demo
fastr-python run --config /path/to/demo/demo.yml
```

The demo checks the software path only; it is not protocol-specific validation.

## Correct a real BrainVision recording

Copy [`examples/configuration.yml`](examples/configuration.yml), set the paths,
confirm the timing source, and run:

```text
fastr-python run --config /path/to/configuration.yml
```

Use [`configuration-slice.yml`](examples/configuration-slice.yml) for recordings
marked once per acquisition group. See the [usage guide](docs/usage.md) and
[configuration reference](docs/configuration.md) for details.

## Outputs and provenance

Each run writes a corrected `.vhdr` recording with `.eeg`/`.vmrk` companions,
before/after PSD figures, and a JSON sidecar. The sidecar records configuration,
input hashes, timing, geometry, alignment, quality-control measurements, and
skipped or advisory regions. Keep it with the corrected data.

## Acquisition timing

Choose one timing source:

| Marker stream | `timing.marker_kind` | Source of group timing |
| --- | --- | --- |
| One marker per volume | `volume` | BIDS JSON or inline `acquisition` fields |
| One marker per acquisition group | `slice` | Marker positions plus declared `groups_per_volume` |

Volume markers require `RepetitionTime`, `SliceTiming`, and
`MultibandAccelerationFactor`. Slice markers use measured group positions and
declared `groups_per_volume`. See the [configuration reference](docs/configuration.md).

## Scientific limitations

Artifact suppression alone is not evidence of a better correction: neural
signal at the same frequencies may also be removed. Validate timing, inspect
provenance, measure residuals and signal transfer, and review boundary groups,
markers, non-EEG channels, and advisory recommendations. See the
[algorithm](docs/algorithm.md) and [validation checklist](docs/validation.md).

## Documentation and citation

Start with the [documentation index](docs/README.md). The method follows
[Niazy et al. (2005)](docs/references.md#niazy-et-al-2005); MNE-Python provides
supported I/O and diagnostics. For method and dependency citations, see
[`CITATION.cff`](CITATION.cff). For reproducibility, record the installed
FASTR-Python version with:

```text
fastr-python --version
```

## Development

```text
uv run ruff check src tests validation
uv run ruff format --check src tests validation
uv run mypy
uv run pytest
git diff --check
uv build
```

See the [development guide](docs/development.md) and
[FMRIB parity audit](docs/fmrib-parity-validation.md).
