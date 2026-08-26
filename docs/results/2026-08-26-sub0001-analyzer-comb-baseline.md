# Sub-0001 Analyzer Comb-Notch Baseline

## Purpose

Quantify the reported notches at integer multiples of 1/TR in a run without a volume
marker gap before evaluating the new FASTR candidate. This is a descriptive real-data
metric, not evidence of neural-signal loss by itself.

## Input

- Subject/run: sub-0001 run 1
- Analyzer scanner-corrected output: 64 channels, 486,900 samples, 1,000 Hz
- Repetition time: 0.9 s
- Channels summarized: first 12 non-ECG channels
- Spectrum: Welch PSD, 180-second segments, 90-second overlap
- Notch metric: median-channel PSD at each exact 1/TR harmonic relative to the median
  PSD 0.02--0.05 Hz to either side

## Result

| Harmonic | Frequency (Hz) | Center / local shoulder (dB) |
|---:|---:|---:|
| 1 | 1.1111 | -30.47 |
| 2 | 2.2222 | -33.70 |
| 3 | 3.3333 | -33.58 |
| 4 | 4.4444 | -37.75 |
| 5 | 5.5556 | -34.04 |
| 6 | 6.6667 | -33.17 |
| 7 | 7.7778 | -30.64 |
| 8 | 8.8889 | -37.91 |
| 9 | 10.0000 | -35.32 |
| 10 | 11.1111 | -32.97 |

The Analyzer output therefore has a pronounced spectral comb at the volume repetition
frequency. A replacement is not accepted merely because these notches are shallower:
synthetic signal-injection tests must show that exact-harmonic EEG is retained while the
known gradient artifact is suppressed.
