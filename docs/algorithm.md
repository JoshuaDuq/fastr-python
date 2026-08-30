# Algorithm

The EEG-fMRI FASTR project implements an acquisition-slot scanner-gradient
correction pipeline.
The implementation is designed around explicit timing metadata and exact marker
validation so that acquisition geometry is never guessed from the EEG waveform.

## Processing model

For each run, the pipeline:

1. Reads the BrainVision header and losslessly parses its marker file.
2. Selects volume-start markers using the exact configured marker type and
   description.
3. Loads `RepetitionTime`, `SliceTiming`, and
   `MultibandAccelerationFactor` from the BIDS fMRI JSON.
4. Validates that volume markers are contiguous at the declared TR. When explicit
   repair is enabled, it inserts only uniquely located interior markers and checks
   the resulting count. It then expands each volume start into fractional
   acquisition-group triggers using the unique slice timing offsets.
5. Interpolates the data onto a finer temporal grid for sub-sample alignment.
6. For each acquisition-time slot, constructs a target-excluding template from the
   configured number of neighboring volumes, estimated from a high-passed copy of
   the channel.
7. Fits group shifts once from the same high-passed copy of the configured
   reference channel. Optional `residual_gate` then flags volumes whose
   first-pass leftover at the slice harmonics is an extreme outlier and drops
   those volumes from *clean* neighbours' templates. Outlier volumes keep their
   local window. The gate is off by default, and its robust thresholds, maximum
   exclusion fraction, mains frequency, and mains exclusion width are configurable.
   It then fits and subtracts a scaled template for every channel batch, except
   on the channels named by `processing.non_eeg_channels`, where the template is
   subtracted unscaled. The estimate is subtracted from the unfiltered channel,
   so slow content survives correction.
8. Optionally fits a fixed or automatically selected residual optimal basis, in
   one or more sections, over whole-volume epochs.
9. Optionally applies FMRIB's scaled-reference normalized LMS adaptive noise
   cancellation to EEG channels.
10. Applies the zero-phase low-pass filter when requested, takes the output window
   and decimates, then regresses any explicitly configured stationary line
   frequencies on the EEG channels. A zero low-pass is valid only without
   decimation.
11. Writes the corrected data, windowed markers, before/after PSD figures, and a
   provenance sidecar.

The first and last groups whose complete artifact epochs are unavailable are left
unchanged, recorded in the sidecar, and annotated in the output marker file as
`Bad Interval, Bad_Gradient`, so a corrected file never carries raw gradient
artifact without saying so.

## Trimming and boundary margin

A recording trimmed to exactly its first and last volume markers has no margin at
either end, and a boundary volume then loses all of its groups because a volume is
dropped when any one of its groups lacks a complete epoch. For a 0.9 s TR with 18
acquisition slots the epochs need 10.5 input samples before the first volume marker
and 4503 (0.9006 s) after a volume marker to keep that volume, so a first-to-last
trim costs the first volume and the last two.

Setting `trim.mode` to `first_to_last_volume` moves the trim inside the pipeline:
the input is the untrimmed recording, correction and filtering run over all of it,
and the emitted span is sliced during decimation. The decimation phase is anchored
to the window start rather than to input sample zero, so the output sample grid is
unchanged. Filtering before slicing also keeps the zero-phase filter's edge
transient outside the emitted span.

The final volume is still dropped where the recording stops before that volume
finished acquiring, which no amount of margin can recover.

## The stage-2 template high-pass

[Niazy et al. (2005)](https://pubmed.ncbi.nlm.nih.gov/16150610/) build the
moving-average template, and fit the least-squares
scalar, on a 1 Hz high-passed copy of the interpolated signal so that segments
entering the average share a baseline, then subtract the estimate from the original
signal. The released `fmrib_fastr.m` does not apply that high-pass; it high-passes
at 70 Hz only when building the residual matrix for the optimal basis set.

`processing.template_high_pass_hz` follows the paper, defaulting to 1.0 Hz. The
pipeline applies the same filter to the reference channel before alignment, so
the shifts and the template are estimated on one signal. On this
cohort, estimating the template from the unfiltered signal let baseline drift leak
into both the template and the scalar: on nine channel-runs from three subjects the
residual line artifact fell from 1.1--20.9 uV to 0.00--0.43 uV, against Analyzer's
0.02--1.16 uV, and broadband amplitude recovered to within 0.2 % of Analyzer in
every motion block. On drift-free synthetic data the high-passed template costs
about 7 % more neighbour noise. Set it to 0.0 to restore the unfiltered estimate.

## Non-EEG channels

`fmrib_fastr.m` forces the least-squares scalar to one on the channels it is told
to exclude, and its changelog names the reason: an ECG channel's QRS complex has
no counterpart in the moving-average template, so a scalar fitted through it is
biased and spreads that bias over the whole epoch.

`processing.non_eeg_channels` reproduces that rule and defaults to `[ECG]`. The
same names are kept out of the residual basis set, out of the line-noise
regression, and out of the residual-QC channel ensemble. Across the 144-run
cohort the ECG channel's per-epoch scalar varied 2.56 times as much as the median
EEG channel and was the least stable channel of all in 58 runs, so this is the
channel the rule exists for.

## The output low-pass

`fmrib_fastr.m` designs the output filter with `firls` over a 15 % transition band
and applies it with `filtfilt`. Running a linear-phase FIR twice squares its
response, which doubles the passband ripple in decibels; at a 100 Hz cutoff and
5 kHz input that design ripples by 2.1 dB between 1 and 85 Hz.

This pipeline designs the filter with MNE's `create_filter` — a Hamming-windowed
linear-phase FIR — and applies it once, compensating the group delay with a
`same`-mode convolution. The passband is then flat to 0.03 dB up to the configured
cutoff, and the worst response anywhere above the output Nyquist frequency is
−79 dB, so `lowpass_hz` describes what the output actually keeps. Filtering still
happens before the output window is sliced, so the edge transient stays outside
the emitted span.
Set `processing.lowpass_hz` to `0.0` to leave the output unfiltered. This is
accepted only when `output_sampling_rate_hz` equals the input sampling rate;
decimation without an anti-alias filter is rejected.

## Residual OBS and adaptive noise cancellation

FASTR as published has four stages. Trigger alignment and moving-average template
subtraction always run. Residual optimal-basis subtraction and adaptive noise
cancellation are separate opt-in stages, so enabling one never changes an existing
configuration silently.

The published canceller is ported from `fmrib_fastr.m` and `fastranc.c`. Its sample
update matches a MATLAB-generated fixture at `1e-13` tolerance. It was previously
measured on `sub-0001` run 1 (63 EEG channels, 443 uV artifact
RMS). It cut the volume-harmonic residual by 13 to 15 dB — 41.078 Hz fell from
-4.3 dB to -18.0 dB — while injected tones near those harmonics did not survive:
61.15 Hz kept 9.2 % of its amplitude, 82.15 Hz kept 13.4 %, 41.156 Hz kept 30.6 %.
An off-comb 7 Hz tone was untouched at 105 %. The slice-style 15 Hz high-pass and
the volume-style 2 Hz high-pass behaved the same way. The LMS reference is the
artifact estimate itself, whose spectrum is a dense comb, so the filter adapts to
cancel any narrowband EEG sitting near a tooth. That is the 1/TR limitation below
with an adaptive filter attached, and no residual-line improvement justifies it.
`processing.adaptive_noise_cancellation` therefore defaults to false.
Zero-variance references, non-finite states, and divergence raise an error; the
pipeline does not silently skip a failed EEG channel. Excluded and flat channels
bypass the stage.

`processing.residual_obs` enables the optimal basis set and defaults to off, so
the stage never changes an existing configuration silently.
`processing.residual_obs_rank` sets a positive fixed rank or `auto` and defaults
to 4. `processing.residual_obs_section_seconds` controls section duration; null
uses one basis for the run. The residual API also takes `section_seconds`, which
re-estimates the basis over consecutive stretches the way `fmrib_fastr.m` does
once per 60 s section. Shortening the sections costs signal monotonically. On a
synthetic run of 84 volume epochs, broadband 75--200 Hz content retained 0.73 under one basis for the
whole recording, 0.62 with four-second sections, 0.55 with two and 0.36 with one:
each extra section spends another `rank` degrees of freedom. Sections are split
into balanced runs rather than cut at a fixed length, because a run holding
barely more epochs than the rank spans nearly everything in that stretch — an
earlier fixed-length split left a five-epoch remainder that destroyed 95 % of the
same probe. The stage runs over whole volumes rather than acquisition groups.
What it removes is the volume-to-volume variability the template stage leaves
behind: on `sub-0001` run 1 the stationary volume-locked mean carries 0.03 % of
the residual comb's tooth power between 10 and 110 Hz, so averaging cannot reach
it, while a rank-4 basis over volume epochs cut the comb by 1.4 dB at 0.9994
injected-signal retention. The same basis estimated over acquisition-group
epochs made the comb worse, because a 50-sample epoch cannot represent a pattern
that spans a volume. Boundary volumes whose epochs would read past either end of
the recording are left uncorrected, the same treatment template subtraction gives
them, and the sidecar records how many epochs the stage actually corrected. Neither absence is hidden by the sidecar, which records exactly
what ran. PSD figures use the same interval containing
only complete corrected epochs, so uncorrected boundary data cannot dominate the
diagnostic. They use the configured PSD frequency limit, capped at the output
Nyquist, and use standard MNE spatial channel colors when channel positions can be
identified. A marker gap or more than one native sample of timing inconsistency is
a hard error by default. Explicit repair requires `expected_volume_count` and
fills only integer-multiple interior gaps; missing boundary markers remain an
error. A single sample (0.2 ms at 5 kHz) is treated as clock quantization inside
the alignment search.

## Why acquisition slots matter

In multiband acquisitions, adjacent groups can represent different acquisition-time
slots. Matching templates by the unique values in `SliceTiming` prevents those
different temporal positions from being averaged together. The number of repeated
slice groups is derived from the metadata and checked against the multiband factor.

## Relationship to the FMRIB implementation

The official FMRIB FASTR plug-in is a MATLAB/EEGLAB implementation. This project
shares the FASTR method's trigger alignment and moving-template subtraction stages,
but it is a separately organized Python implementation with a different input
contract and processing boundary. In particular, its acquisition-group geometry is
derived from BIDS multiband timing, while the FMRIB plug-in accepts EEGLAB slice or
volume events. This project is not the official FMRIB plug-in and makes no
affiliation or endorsement claim.

## Configuration and provenance

The example YAML is the single user-facing configuration surface. Protocol and
analysis choices include the interpolation factor, template neighbour count,
search radius, relative trigger position, marker-repair policy, template high-pass,
output filter/rate, exact line-noise frequencies, non-EEG channel names, residual
OBS rank and section duration, ANC, residual threshold, optional gate/adaptive
settings, trim mode, residual-QC block and mains settings, and PSD limits. Line regression deliberately removes all signal
at each configured frequency, so the YAML requires an explicit list; `[]` disables
it. Fixed implementation details include the interpolation kernel shape, the
residual OBS 70 Hz high-pass design, the output low-pass window design, and the
protected boundary volumes. The
provenance sidecar stores the FMRIB reference commit, resolved configuration,
marker repair counts, selected OBS ranks, ANC diagnostics, exact-bin and
local-sideband volume-harmonic spectra, effective PSD limit, FFT length, and
residual-QC settings.

## The 1/TR limitation

An acquisition-locked signal at the volume repetition frequency, or at one of its
harmonics, is indistinguishable from a scanner artifact when both are represented by
the same volume-locked reference. Subtracting a template therefore can reduce a
genuine neural component at exactly `1 / RepetitionTime` as well as scanner energy.
No blind post-hoc notch can restore information that was removed during subtraction.

The template window is consequently a scientific parameter, not merely a speed
setting. On this cohort, 60 neighbouring volumes cut leave-one-out transfer
inflation from 4.4% (N=20) to 1.1% without increasing residual slice-harmonic
amplitude. That value must be revalidated when the scanner sequence, electrode
montage, marker timing, or recording protocol changes.

## Inputs and units

The implementation expects voltage-valued BrainVision channels, sample positions in
the BrainVision marker convention, and BIDS timing in seconds. Internally, marker
positions are converted to zero-based input samples, correction is performed on the
original sample grid, and marker positions are mapped through the exact integer
decimation factor.

## External references

- [MNE-Python BrainVision reader](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html)
- [BIDS MRI data acquisition specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html)
- [Niazy et al. (2005), “Removal of FMRI environment artifacts from EEG data using optimal basis sets”](https://pubmed.ncbi.nlm.nih.gov/16150610/)
- [FMRIB fMRIb FASTR implementation (`fmrib_fastr.m`)](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m)
- [FMRIB fMRIb repository](https://github.com/sccn/fMRIb)
