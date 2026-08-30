"""Scanner-gradient artifact correction for simultaneous EEG-fMRI.

The correction pipeline is `config` -> `pipeline`, over `fastr` and the
BrainVision reader and writer. `demo` writes a runnable synthetic dataset, and
`compare` plots two folders of recordings against each other.

`metrics`, `diagnostics`, `simulation`, and `matlab_comparison` are validation
instrumentation: they measure or simulate a correction and are imported by the
tests, the demo, and the validation runners, never by the pipeline itself.

Cardiac detection and BCG correction live in BCG-Correction, not this package.
"""

__version__ = "0.1.0"
