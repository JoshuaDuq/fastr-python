# FASTR Review-Fix Verification

This verification exercises the corrected acquisition-slot FASTR implementation on
real recordings and checks the marker-integrity cases from the review. The benchmark
is reproducible with `scripts/benchmark_sub0001.py`; it refuses to overwrite its JSON
output.

## Sub-0001 run 1: complete 64-channel benchmark

The raw recording contains 542 `Volume/V  1` markers at 5 kHz and every adjacent
marker is exactly 4,500 samples apart. The benchmark measured volumes 20--419, used
one preceding and one following volume as context, corrected all 64 channels
including ECG, and applied the 100 Hz zero-phase Butterworth plus factor-5 decimation.
Wall-clock runtime on this host was 26.15 seconds.

| Method | Volume-locked RMS (uV) | Group-locked RMS (uV) | Median 1/TR comb (dB) |
|---|---:|---:|---:|
| raw | 582.95 | 578.41 | +2.12 |
| Analyzer volume AAS | 22.31 | 22.31 | −21.52 |
| acquisition-slot FASTR | **0.51** | **0.50** | −20.15 |

The edge wrapper left only the first and last 18 context groups untouched. This is a
real-data suppression comparison, not a neural-signal preservation claim; that still
requires paired injection tests and the downstream Analyzer pulse-history check.

## Fail-fast marker checks

- Sub-0000 run 1 contains a 317,672-sample interval after marker 570. The timing
  validator rejects it with `FastrInputError` before group triggers or correction are
  created.
- Sub-0018 run 1 contains two one-sample timing deviations (4,501 followed by 4,499
  samples). The validator also rejects it; it does not silently reclassify the timing
  as jitter or repair it.

The CLI prints the typed validation message and exits with status 1. A scanner break,
missing marker, and small marker-timing deviation cannot be distinguished safely from
the marker series alone, so all non-contiguous TRs remain explicit refusals.

## Code-level fixes verified

- Classical alternating FASTR and acquisition-slot FASTR are separate public entry
  points; the latter derives matching geometry from validated BIDS timing.
- Edge wrappers report skipped groups and preserve untouched samples; strict core
  functions still reject incomplete epochs.
- Fractional trigger positions are preserved in trigger-locked and event-locked
  metrics.
- Channel amplitudes are fitted against shifted, aligned templates, so timing
  perturbations do not leave the amplitude stage using stale nominal epochs.
- Provenance arrays are copied and read-only, and the CLI records paths, timing,
  channels, parameters, filter, skipped edges, and metrics.

## Remaining gates

Held-out runs and subjects, paired exact/near-harmonic and broadband injection transfer,
export/reopen validation, memory profiling on the full production path, and importing
an exported result into Analyzer 2.3 for the existing pulse-marker and pulse-correction
history remain required before declaring a final replacement.
