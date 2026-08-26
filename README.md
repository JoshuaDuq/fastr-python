# Python MRI Correction

Scanner-gradient artifact correction for simultaneous EEG-fMRI.

The project is under active implementation. BrainVision Analyzer is retained as a
benchmark; enhanced methods are promoted only after artifact-suppression and neural
signal-transfer tests pass.

The strict classical FASTR API is `mri_correction.fastr.slice_fastr`. For multiband
recordings, use `acquisition_group_fastr`, which derives acquisition-time slots from
validated BIDS `RepetitionTime`, `SliceTiming`, and
`MultibandAccelerationFactor`. `SliceTiming` describes acquisition time, not physical
slice identity.

Validate BrainVision volume markers before correction:

```text
.venv/bin/mri-correct validate-timing \
  --metadata /path/to/bold.json \
  --vhdr /path/to/raw.vhdr \
  --sampling-rate 5000 \
  --output /path/to/timing-validation.json
```

Reproduce the sub-0001 real-data benchmark, including all channels and ECG:

```text
.venv/bin/python scripts/benchmark_sub0001.py \
  --raw-vhdr /path/to/raw.vhdr \
  --analyzer-vhdr /path/to/analyzer.vhdr \
  --fmri-json /path/to/bold.json \
  --output /path/to/benchmark.json
```

Existing output files are never overwritten. Marker gaps and non-contiguous TRs fail
before correction begins.
