# Gradient-correction boundaries and template quality

Date: 2026-08-26
Status: approved

## Why

A 102-run validation cohort was compared against BrainVision Analyzer
(`step1_scanner_artifact_pulse_marked`, the same processing stage: gradient
correction, 5 kHz to 1 kHz, no pulse correction). The core correction is sound:

| metric | raw | FASTR-Python | Analyzer |
| --- | --- | --- | --- |
| slice-harmonic peak / local background, median | 2555x | 0.95x | 0.94x |
| per-30 s-block residual, median (full duration) | -- | 1.46x | 1.45x |
| Pearson r vs Analyzer, 1-40 Hz, median | -- | 0.99978 | -- |
| time lag vs Analyzer | -- | 0 samples in 101/102 | -- |

Four defects were measured. The goal is not to match Analyzer but to reach and
where possible beat it, judged on absolute criteria.

### Defect 1: every output is about 4.6 % too large

Measured at off-harmonic frequencies (5-45 Hz, at least 0.42 Hz from any 1/TR
line), where the raw contains only real EEG:

```
FASTR / raw : median 1.0455  (range 1.034 - 1.052, n = 91 channel-runs)
BVA   / raw : median 0.9995  (range 0.994 - 1.002)
```

The gain is flat from 0.1 Hz to 45 Hz on every channel in every run. It is not
an I/O scaling error: in the uncorrected boundary region the output matches
`filtfilt(butter(2, 100)) -> [::5]` of the raw to 5e-5 uV, regression slope
1.000000. It is variance inflation from the leave-one-out template mean, and it
scales as expected with the neighbour count:

```
N=20  FASTR/raw = 1.046      N=40 = 1.021      N=80 = 1.008
```

The cost is about 9 % added variance on 100 % of the data.

### Defect 2: 2.7 s of raw gradient artifact in every output

`skipped_group_indices` is 54 in all 102 runs: the first volume and the last
two. Measured on `baseline_sub0000`:

| region | FASTR RMS / pk-pk (Fp1) | Analyzer |
| --- | --- | --- |
| first 0.9 s | 383 uV / 3272 uV | 21 uV / 114 uV |
| last 1.8 s | 260 uV | 21 uV |
| ECG, first 0.9 s | 2760 uV / 19750 uV | 315 uV |

Nothing in the output `.vmrk` marks the region. Cause established: the
`step0_trimmed_raw_5khz` inputs are byte-identical to
`untrimmed[firstVolumeMarker-1 : lastVolumeMarker]`, verified at head, mid and
tail, so they carry zero margin at either end. FASTR needs 10.5 samples
(2.1 ms) before the first volume marker, and 4503 samples (0.9006 s) after a
volume marker to keep that volume. The second-to-last volume is short by
**3 samples** and loses all 18 of its groups because
`src/mri_correction/fastr.py` drops a whole volume when any one group is
invalid.

Untrimmed 5 kHz data exists for all 21 real subjects (7-8 runs each) with
16.4-28.5 s of head margin and 0.04-10.5 s of tail margin. The trim was
housekeeping, not a requirement.

### Defect 3: baseline drift leaks into the template

A full-duration scan of all 102 runs in 30 s blocks (11 417 channel-blocks,
excluding the 3rd slice harmonic because 20 Hz x 3 lands on 60 Hz mains) found
routine performance indistinguishable from Analyzer (blocks above 3x
background: 344 for FASTR, 326 for Analyzer). Seven runs, all from subjects
0004, 0007 and 0012, carry more residual line artifact than Analyzer, worst
20.9 uV on run2 sub0012 Fp1 against Analyzer's 1.2 uV. Every flagged block is a
heavy-motion block, and in every one FASTR's broadband amplitude is *lower*
than Analyzer's.

The cause is a deviation from Niazy et al. (2005) stage 2. The paper builds the
moving-average template, and fits the least-squares scalar, on `Y_h`: a 1 Hz
high-passed copy of the interpolated signal, "to ensure that the different
artifact segments used in the average artifact estimation have the same
baseline". The estimated artifact is then subtracted from the original signal,
not the high-passed one, so slow content is preserved. This implementation
estimates both the template and the scalar from the un-high-passed signal, so
baseline shifts leak into the template and bias the fit.

Measured on run2 sub0012 Fp1, current versus paper-faithful versus Analyzer:

| block | current | 1 Hz high-pass | Analyzer |
| --- | --- | --- | --- |
| 150-180 s, line residual | 20.87 uV | 0.21 uV | 1.16 uV |
| 150-180 s, broadband | 380 uV | 1403 uV | 1401 uV |
| 30-60 s, line residual | 0.83 uV | 0.18 uV | 0.29 uV |
| 30-60 s, broadband | 41 uV | 120 uV | 120 uV |
| amplitude-fit spread, whole run | 0.1397 | 0.0030 | -- |

The paper-faithful variant and Analyzer agree to within 0.5 % on broadband
amplitude; the current implementation is the outlier, discarding roughly 73 %
of the low-frequency signal amplitude on drifting channels while leaving 18x
more line artifact. Clean channels (F3, Cz) are unaffected either way.

Note that the released reference implementation, `fmrib_fastr.m`, does *not*
apply this high-pass before templating; it high-passes at 70 Hz only when
building the OBS residual matrix, which is what `_HIGH_PASS_HZ = 70.0` in
`_make_residual_high_pass` correctly mirrors. The repo follows the released
code. On this data the paper's description is measurably better.

### Defect 4: two of FASTR's four stages are not in the pipeline

Niazy et al. define FASTR as four stages: trigger alignment, moving-average
template subtraction, residual removal by optimal basis set (OBS), and adaptive
noise cancellation (ANC). `residual_obs` (`src/mri_correction/fastr.py:597`)
implements stage 3, but `pipeline.py` never imports it. Stage 4 is absent
entirely. The whole validation cohort was produced with stages 1 and 2, which
the paper states remove "more than 98%" of the artifact, leaving stages 3 and 4
to address the remainder.

This is recorded as scope, not as work: defect 3's fix is measured first, and
stages 3 and 4 are considered only if the re-measured cohort still falls short
of Analyzer.

## What changes

### A. Trimming moves into the pipeline

New optional `trim` config section:

```yaml
trim:
  mode: first_to_last_volume   # or: none
  minimum_epoch_coverage: 0.75
```

`mode` defaults to `none`, which preserves current behaviour for already-trimmed
inputs; `examples/configuration.yml` uses `first_to_last_volume`.
`minimum_epoch_coverage` is the fraction of an epoch that must fall inside the
recording for a boundary group to be corrected at all.

With `mode: first_to_last_volume`, `input.raw_vhdr` points at untrimmed data
and the pipeline:

1. Selects volume markers and computes the output window
   `[firstVolumeMarker - 1, lastVolumeMarker]` in input samples.
2. Builds FASTR geometry over the **full untrimmed sample range**, so boundary
   volumes have complete epochs and real neighbours.
3. Corrects and low-passes the full array, then slices and decimates in one
   step: `filtered[:, start:stop:decimation]`.
4. Remaps markers relative to the window start, dropping markers outside it.

Step 3 is load-bearing. Decimating from sample 0 and slicing afterwards would
shift every output sample by up to 4 samples. Slicing before decimating
reproduces today's sample grid exactly, because today's input already begins at
`firstVolumeMarker - 1`. Filtering the full array also removes the `filtfilt`
edge transient currently present at both ends of every output.

Boundary groups that still lack a complete epoch (only the final volume's first
group, which contributes one sample to the output) get a partial-epoch fit: the
template comes from the 20 same-slot neighbours as usual, the amplitude is
fitted over the available samples only, and the fitted epoch is subtracted over
those samples. Below `minimum_epoch_coverage` the group is left uncorrected and
annotated.

> **Implementation note (2026-08-26):** `minimum_epoch_coverage` and the
> partial-epoch fit it gated were dropped. Measured on real untrimmed input, the
> only group that still lacks a complete epoch is the final volume's first group,
> which contributes three output samples; those are annotated
> `Bad Interval, Bad_Gradient` instead. A configuration option that gates nothing
> is worse than no option.

### B. Neighbour count chosen from measurement

An analysis pass, not a code change, run before any default moves.

Sweep `neighbor_count` over {10, 20, 30, 40, 60, 80} on a stratified subset:
eight clean runs plus every run from subjects 0004, 0007 and 0012. Three
outcomes per setting, per channel:

- EEG transfer gain vs raw at off-harmonic frequencies (today 1.046)
- residual line artifact in uV, full duration, median
- worst 30 s block residual in uV

The default is chosen from that curve on absolute criteria, with Analyzer's
numbers (0.07-1.2 uV residual, 1.000 gain) as the line to beat. A wider window
is expected to trade lower inflation for worse tracking of a motion-modulated
artifact; the sweep establishes whether that trade is real and where it turns.

The measured transfer gain is written to the sidecar regardless of which
default is chosen, so downstream users can correct for it.

### C. Estimate the template on a high-passed signal

Add `processing.template_high_pass_hz`, default 1.0. The moving-average
template and the least-squares scalar are estimated from a high-passed copy of
each channel; the resulting artifact estimate is subtracted from the unfiltered
channel, so slow content survives. Setting it to 0.0 restores current
behaviour and is the escape hatch for reproducing the existing cohort.

Filtering is applied at the input rate before interpolation rather than on the
10x grid, where a 1 Hz Butterworth is poorly conditioned.

This changes the estimator for every epoch in every run, so it is validated by
the same cohort re-run that validates the trimming change. It is expected to
leave clean channels untouched and to change motion blocks substantially; a
clean-channel regression would mean the filter is wrong, not that the idea is.

A trimmed-mean template was considered first and rejected: it treats the
symptom, costs variance on the 97 % of blocks that already work, and is not
what the algorithm specifies.

### D. Full-duration residual QC

Independent of C, and shipped regardless.

The sidecar gains a per-channel, per-30 s-block residual measurement: excess
power at the slice harmonics expressed in **uV**, not as a ratio against local
background. The ratio form produced a 310x figure for a 1.75 uV residual and a
13.9x figure for a 0.05 uV residual, and is not a usable threshold.

The harmonic set is **derived** from `groups_per_volume / repetition_time`, and
any harmonic falling within 1 Hz of the mains frequency (configurable, default
60 Hz) is excluded. For this cohort the slice rate is 20 Hz and the 3rd
harmonic collides with mains exactly.

Blocks exceeding `residual_threshold_uv` are annotated in the output `.vmrk` as
`Bad Interval, Bad_Gradient`, alongside the same annotation for any uncorrected
boundary region. The default is 1.0 uV: on the measured cohort that flags five
of the six problem blocks (1.1 to 20.9 uV) and correctly ignores the 0.05 uV
false positive. The threshold is re-checked against the full cohort during
implementation.

## Interfaces

| unit | responsibility | depends on |
| --- | --- | --- |
| `config.TrimConfig` | parse and validate `trim` | none |
| `pipeline._resolve_output_window` | markers + mode -> `(start, stop)` in input samples | markers |
| `pipeline._lowpass_and_decimate` | filter full array, slice, decimate | window |
| `brainvision_io.resample_markers` | rescale **and** offset markers to the window | window |
| `fastr._fit_channel_noise` | partial-epoch amplitude fit at boundaries | epoch coverage |
| `fastr.apply_fastr_batch` | high-pass a channel copy before templating | `template_high_pass_hz` |
| `diagnostics.block_residuals` | per-block uV residual at derived harmonics | timing |

## Testing

Test-driven throughout.

- Window arithmetic: `_resolve_output_window` for both modes, and that
  `mode: none` on an already-trimmed input reproduces today's output byte for
  byte. This is the regression gate for the decimation-phase change.
- Marker remapping: offset and rescale together; markers outside the window
  dropped; a marker exactly on either boundary retained.
- Partial epochs: a synthetic run whose final group has 60 % coverage is
  corrected over those samples and not beyond; below threshold it is skipped
  and annotated.
- Derived harmonics: a 20 Hz slice rate excludes 60 Hz; a 17.5 Hz slice rate
  does not.
- Residual QC: a synthetic residual of known amplitude is recovered in uV to
  within 5 %.
- Template high-pass: a synthetic recording with a step baseline shift keeps the
  step after correction and holds the scalar's spread below 0.02; with
  `template_high_pass_hz=0.0` the output matches the pre-change code path
  exactly.

Cohort validation reuses the comparison harness built during the
investigation: matched-file discovery, off-harmonic transfer gain vs raw,
full-duration blockwise residual, and agreement with Analyzer.

## Risks

- Re-running the cohort costs about 76 s/run x 102, roughly 2.2 h, plus the
  sweep (6 settings x about 20 runs).
- The decimation-phase change silently shifts every output sample if the slice
  and decimate are not fused. The byte-for-byte regression test is the guard.
- A wider template window may worsen defect 3 while fixing defect 1. The sweep
  measures both, so the default is chosen with that trade visible rather than
  assumed.
- The template high-pass changes every corrected sample. Existing tests that
  assert exact corrected values must each be judged individually: unchanged on
  drift-free synthetic data, updated where the fixture contains drift.
- Defect 4 is knowingly deferred. If the re-measured cohort still trails
  Analyzer anywhere, wiring in OBS is the next step, and `residual_obs` needs
  the per-minute chunking and variance-based rank the paper specifies before it
  is trustworthy.
