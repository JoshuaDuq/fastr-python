# FASTR-Python

FASTR-Python is the Python version of [FMRIB FASTR](https://github.com/sccn/fMRIb)
for correcting scanner-gradient artifacts in simultaneous EEG-fMRI recordings.

## What it does

`fastr-python` is research software for correcting scanner-gradient artifact
in simultaneous EEG-fMRI BrainVision recordings. It validates acquisition
timing, applies acquisition-slot FASTR, preserves markers, and writes a
provenance record. It corrects scanner-gradient artifact only: it does not
correct ballistocardiogram (BCG) artifact, general bad-electrode conditions, or
other physiological artifacts.

The package is `fastr_python`, the command is `fastr-python`, and the stable
configuration-driven API is in [`fastr_python.api`](src/fastr_python/api.py).
This is separate Python software from the FMRIB EEGLAB plug-in and is not
affiliated with, sponsored by, or endorsed by the FMRIB Centre or the
University of Oxford. It is released under GPL-2.0-only; see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

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

## Quickstart: synthetic demo

Generate a self-contained BrainVision dataset, then correct it:

```text
fastr-python demo --output-dir /path/to/demo
fastr-python run --config /path/to/demo/demo.yml
```

The demo checks the interface and processing path on simulated data. It is not
evidence of performance on a real acquisition.

## Correct a real BrainVision recording

Copy [`examples/configuration.yml`](examples/configuration.yml), set the input
and output paths, confirm the timing source, and run:

```text
fastr-python run --config /path/to/configuration.yml
```

For recordings marked once per acquisition group, use the separate
[`configuration-slice.yml`](examples/configuration-slice.yml) example. Read
the complete [usage guide](docs/usage.md) and [configuration reference](docs/configuration.md)
before interpreting a corrected run.

## Outputs and provenance

The run writes a corrected `.vhdr` recording and its `.eeg`/`.vmrk` companions,
before/after PSD figures, and a JSON sidecar. The sidecar records resolved
timing, configuration, input hashes, output window, alignment, quality-control
measurements, and any skipped or advisory regions. Inspect it with the
corrected data; it is part of the result, not an optional log.

## Acquisition timing

The timing source is explicit and exclusive:

| Marker stream | `timing.marker_kind` | Source of group timing |
| --- | --- | --- |
| One marker per volume | `volume` | BIDS JSON or inline `acquisition` fields |
| One marker per acquisition group | `slice` | Marker positions plus declared `groups_per_volume` |

Volume markers require declared `RepetitionTime`, `SliceTiming`, and
`MultibandAccelerationFactor`; acquisition-group markers measure the repetition
time and within-volume offsets from the recording. The [configuration reference](docs/configuration.md)
explains the validation rules.

## Scientific limitations

FASTR estimates scanner-locked structure. A correction that lowers a harmonic
is not automatically better if it also removes neural signal at that
frequency. Validate timing, inspect provenance, measure residual artifact, and
measure signal transfer for each protocol. Review boundary groups, marker
quality, non-EEG channels, and any advisory channel recommendations. See the
[algorithm](docs/algorithm.md) and [validation checklist](docs/validation.md).

## Documentation and citation

Start with the [documentation index](docs/README.md). The method is based on
FASTR as described by [Niazy et al. (2005)](docs/references.md#niazy-et-al-2005),
with MNE-Python used for supported I/O and diagnostics. Cite the software and
the references in [`CITATION.cff`](CITATION.cff), and record the installed
version with:

```text
fastr-python --version
```

## Development

```text
uv run pytest
uv run ruff check src tests validation
git diff --check
uv build
```

See the [development guide](docs/development.md) for repository conventions
and the [FMRIB parity audit](docs/fmrib-parity-validation.md) for comparison
scope and evidence.

## Related pipelines

BCG correction is a separate concern. Cardiac detection and AAS/PCA-OBS BCG
correction live in [BCG-Correction](https://github.com/JoshuaDuq/BCG-Correction),
and the deep-learning BCGNet path lives in
[BCGNet-Python](https://github.com/JoshuaDuq/BCGNet-Python). FASTR removes
scanner-gradient artifact only.
