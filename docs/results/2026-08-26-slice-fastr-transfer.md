# Slice-FASTR Exact-Harmonic Transfer and What Governs It

## Purpose

Measure whether the `slice_fastr` candidate preserves EEG at exact multiples of
1/TR, and identify what governs that transfer. Ground truth is unknowable in real
data, so transfer must be established by injection into simulation; the parameter
the answer depends on is then measured on the real recording.

## Correction to the first version of this document (2026-08-26)

The first version of this page concluded that `slice_fastr` retains only about
78% of an exact 1/TR harmonic and therefore fails the design's 95% gate. That
number is real but it describes a simulated artifact that does not resemble this
study's acquisition, and the conclusion drawn from it was wrong.

Transfer here depends almost entirely on one property: how much net offset an
artifact epoch carries, measured as the mean of the epoch over its RMS. The
original fixture had a ratio of 0.52. Sub-0001 run 1 has a **median ratio of
0.051** across nine channels. At that ratio the mechanism below is not
meaningfully active. The physical reason is that the recorded artifact is an
induced voltage, the derivative of a gradient waveform that returns to baseline
within each readout, so it integrates to nearly zero over an epoch.

| Channel | Epoch RMS (uV) | \|epoch mean\| / RMS |
|---|---:|---:|
| Fp1 | 478.9 | 0.051 |
| Fp2 | 513.6 | 0.144 |
| F3 | 1096.6 | 0.038 |
| F4 | 1166.4 | 0.072 |
| C3 | 1121.7 | 0.027 |
| C4 | 1393.4 | 0.074 |
| P3 | 1177.5 | 0.038 |
| P4 | 1440.0 | 0.056 |
| ECG | 5953.5 | 0.040 |

Sub-0001 run 1, 120 interior volumes, 252.6-sample FASTR epochs at 5 kHz.

## The mechanism, and when it matters

FASTR fits one amplitude per channel per group as
`alpha = <epoch, template> / <template, template>`. A neural signal slower than
one epoch is nearly constant across it, so its contribution to that inner product
is proportional to the epoch mean of the template. When the artifact epoch carries
a large offset, slow signal is converted into a modulation of the much larger
artifact template and subtracted with it. When the epoch is balanced, the inner
product nearly vanishes and the signal passes through.

Simulated 1/TR transfer against epoch offset, all else held fixed:

| Artifact epoch \|mean\| / RMS | Retained 1/TR amplitude |
|---:|---:|
| 0.52 | 0.783 |
| 0.29 | 0.976 |
| 0.15 | 1.041 |
| 0.00 | 1.065 |

Holding neighbour selection, alignment and epoch geometry fixed at ratio 0.52 and
changing only the amplitude fit confirms the attribution: unscaled templates
retain 1.069, and unscaled templates with the per-epoch mean removed retain 1.453.

## Interpretation

On this acquisition the amplitude fit does not cost exact-harmonic transfer,
because the artifact epochs are balanced. The mechanism is nonetheless worth
recording: it is a property of the method, not of the data, and any acquisition
whose epochs carry a real offset — a DC-coupled amplifier, a gradient that does
not return, an epoch that does not span a whole readout — will lose slow signal
to it. The ratio is cheap to measure and should be checked per dataset before
`slice_fastr` is trusted on it.

Retention above 1.0 is not a pass either. A correction that adds energy at the
injected frequency is as wrong as one that removes it, and the 1.07 and 1.45
figures above are both distortions.

The numbers here answer only the transfer question. Whether `slice_fastr`
suppresses the artifact better than the Analyzer reference is a separate question
answered against the real recording in
[the sub-0001 head-to-head](2026-08-26-fastr-vs-analyzer-sub0001.md), and the
answer there depends on a different property of the acquisition entirely.

## Limitation

These transfer numbers come from simulation. Real recordings add amplitude drift,
timing jitter and non-stationary artifact shape. This measurement is a necessary
gate, not a sufficient one, and does not replace the real-data metrics or the
manual Analyzer 2.3 downstream check.
