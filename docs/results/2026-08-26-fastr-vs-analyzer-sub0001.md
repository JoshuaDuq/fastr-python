# Can FASTR Replace the Analyzer Scanner Correction? Sub-0001 Run 1

## Question

Whether FASTR is the method to replace the BrainVision Analyzer scanner-artifact
node, judged on the real pilot the design nominates rather than on simulation.

## Data

- Input: `step0_trimmed_raw_5khz/ThermalPainEEGFMRI_run1_sub0001_…`, 5 kHz,
  64 channels, 2,434,501 samples, 542 `Volume/V  1` markers at exactly 4,500
  samples with no gap.
- Incumbent output: the matched `step1_scanner_artifact_pulse_marked/…`
  recording, 1 kHz, 486,900 samples. Volume *k* sits at raw sample 4,500*k* and
  Analyzer sample 900*k*, so the two align without resampling either.
- Group triggers from BIDS `RepetitionTime` 0.9 s, `SliceTiming` with 18 unique
  times, `MultibandAccelerationFactor` 3.
- Measured over 400 volumes from volume 20, first 12 channels. FASTR output is
  low-passed at 100 Hz and decimated by 5 so both live at 1 kHz in the same band.

## The finding that decides it

FASTR builds each group's template from neighbouring groups, assuming they share
an artifact waveform. On this acquisition they do not. Mean epoch correlation
across 8 channels and 60 volumes:

| Epoch pair | Correlation |
|---|---:|
| same acquisition-time slot, one volume later (lag 18 groups) | 1.000 |
| same acquisition-time slot, two volumes later (lag 36 groups) | 1.000 |
| adjacent group, same volume (lag 1) | −0.032 |
| FASTR's alternating-parity neighbour (lag 2) | 0.751 |

Consecutive multiband groups are different acquisition-time slots with essentially
unrelated artifacts. The waveform that does repeat is the same acquisition-time slot one
volume later, and it repeats almost perfectly.

## Results

| Method | Volume-locked RMS (uV) | Group-locked RMS (uV) | Median 1/TR comb (dB) |
|---|---:|---:|---:|
| raw | 289.54 | 265.48 | +2.36 |
| Analyzer volume AAS | 18.07 | 18.07 | −20.91 |
| `slice_fastr`, FASTR alternating parity | 132.26 | 85.89 | +19.36 |
| `acquisition_group_fastr`, acquisition-slot matched | **0.16** | **0.10** | −19.79 |
| acquisition-slot matched + residual OBS | 0.16 | 0.10 | −19.81 |

The comb column is the median-channel PSD at harmonics 1–10 of 1/TR relative to
the median PSD 0.02–0.05 Hz to either side. Welch, 90 s segments, 45 s overlap.

## Answer

**FASTR as documented is not a viable replacement here.** Its alternating local
template is built from the wrong epochs, and it leaves seven times more residual
artifact than the incumbent while adding a +19 dB spectral comb of its own.

**FASTR's machinery with acquisition-slot-matched neighbours is a decisive
improvement.** Two changes, both in neighbour selection rather than in the
subtraction itself, take the residual from 132 uV to 0.16 uV against the
incumbent's 18.07 uV, a factor of 113:

1. Build each template from the same acquisition-time slot in neighbouring volumes.
2. Let the epoch span the whole gap to the next group. Matched groups share a
   gap structure, so this is well defined, and it corrects the volume dead time
   that group-length epochs leave untouched. Without it the residual stalls at
   40 uV: the uncorrected gap dominates everything the method does elsewhere.

What survives from FASTR proper is what makes the difference against Analyzer at
the same neighbour set: sub-sample alignment per group, a per-channel amplitude
fit, and a local rather than fixed template window.

**The 1/TR comb is not avoidable, and that was the original hope.** The artifact
repeats at the volume period with correlation 1.000, so its energy genuinely lies
at multiples of 1/TR. Any method that removes it removes those lines. The matched
candidate's comb (−19.79 dB) matches the incumbent's (−20.91 dB) rather than
avoiding it. The comb measured in
[the Analyzer baseline](2026-08-26-sub0001-analyzer-comb-baseline.md) is a
property of the artifact, not evidence that Analyzer chose badly.

**Residual OBS earns nothing here.** It leaves the residual at 0.16 uV because
there is almost nothing left for a basis set to model. It should stay opt-in and
off by default.

## Consequence for the acceptance criteria

The criteria originally required a candidate to retain at least 95% of injected
exact-harmonic amplitude. The correlation table above rules that out for any
trigger-locked method on this acquisition: the artifact is exactly volume-periodic,
so its energy is at those harmonics and removing it removes them. Measured on
simulation with per-slot artifact waveforms, by paired runs so residual artifact
cannot be mistaken for signal:

| Method | Residual group-locked RMS (uV) | Exact 1/TR transfer, median / min | Broadband 1–100 Hz |
|---|---:|---:|---:|
| Analyzer volume AAS | 1.465 | 0.000 / 0.000 | 0.978 |
| acquisition-slot matched | 2.192 | 0.008 / 0.001 | 1.018 |

Neither method retains exact 1/TR harmonics, and the incumbent never did either.
The design has since been rewritten accordingly: the gate now applies to
near-harmonic and broadband transfer, and exact-harmonic transfer is a separately
reported safety measurement with an application-specific tolerance rather than a
pass/fail threshold. Measurements against the rewritten gate, including the
near-harmonic probes, are recorded in
[the review-fix verification](2026-08-26-review-fix-verification.md).

## What this does not establish

- One run, one subject, 12 of 64 channels. Sub-0000 run 1, with its known marker
  gap, is still reserved as a fail-fast case and has not been run.
- The comparison low-passes and decimates the FASTR output with a plain
  fourth-order zero-phase Butterworth, not the compensated filter the design
  specifies for production output. That affects both methods' broadband numbers
  slightly and neither method's trigger-locked residual materially.
- No export, and therefore no Analyzer 2.3 downstream pulse-correction check.
  That manual gate remains open.
- `slice_fastr` requires a volume of margin at each end of the segment it is
  given, because the first group's searched epoch reaches back before its
  trigger. Correcting a whole recording needs that handled deliberately.
