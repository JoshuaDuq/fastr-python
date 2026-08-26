# Initial Slice-Group Pilot

## Purpose

Test whether a short-period artifact model can preserve signal at 1/TR while suppressing
the scanner gradient artifact better than whole-volume subtraction.

## Data and fixed exploratory parameters

- Input: sub0000 run 1 trimmed 5 kHz BrainVision source recording.
- Segment: first 100 contiguous volume markers.
- Channels: first 12 EEG channels.
- TR: 4,500 samples (0.9 s).
- Diagnostic repetition candidate: 244 samples (48.8 ms), mean first-difference
  autocorrelation approximately 0.923 across 12 channels.
- Exploratory group model: 18 groups per volume at offsets `0, 244, ..., 4148`, group
  length 244 samples, 20 target-excluding neighboring groups.
- Metric filter: fourth-order zero-phase 100 Hz Butterworth used only for this comparison.

The 244-sample period and 18-group layout were inferred from EEG and are not accepted as
production metadata. The 108-sample remainder in each TR shows why explicit sequence
timing and dead time are required.

## Results

| Method | Volume-locked RMS (uV) | Group-locked RMS (uV) | Mean 1/TR-comb contrast, harmonics 1--20 (dB) |
|---|---:|---:|---:|
| Raw | 256.121 | 244.851 | +10.696 |
| Analyzer-style volume AAS | 0.430 | 0.065 | -4.926 |
| Target-excluding slice-group AAS | 92.566 | 4.864 | +17.866 |

The slice-group candidate sharply reduced group-locked artifact and avoided the negative
1/TR spectral comb, but its residual volume-locked artifact was far above the volume-AAS
reference. It therefore fails the current replacement criterion and must not be used for
production correction.

## Interpretation

The synthetic transfer test demonstrates the identifiability advantage: volume AAS
retained effectively 0% of a pure 1/TR sinusoid, whereas slice-group AAS retained more
than 99% RMS. The real-data failure shows that the gradient waveform is not adequately
described by one uniform 244-sample group template. Exact multiband group timing,
sub-sample alignment, position-dependent waveform variation, and a validated FASTR/OBS
residual model are the next scientific tests. Spectral-bin interpolation is not an
acceptable remedy because it would invent rather than recover data.
