# Algorithm

The EEG-fMRI FASTR project implements an acquisition-slot scanner-gradient
correction pipeline.
The implementation is designed around explicit timing metadata and exact marker
validation so that acquisition geometry is never guessed from the EEG waveform.

## Processing model

For each run, the pipeline:

1. Reads the BrainVision header and losslessly parses its marker file.
2. Selects markers using the exact configured marker type and description.
3. Resolves the acquisition geometry, by whichever of the two routes the
   configuration declares (see *Where acquisition groups come from* below).
4. Validates that volume boundaries are contiguous at one repetition time. When
   explicit repair is enabled on volume markers, it inserts only uniquely located
   interior markers and checks the resulting count.
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
9. Applies the zero-phase low-pass filter when requested, to the corrected data
   and, when the canceller runs, to the artifact estimate as well.
10. Optionally applies FMRIB's scaled-reference normalized LMS adaptive noise
   cancellation to EEG channels, against that low-passed reference.
11. Takes the output window and decimates, then regresses any explicitly
   configured stationary line frequencies on the EEG channels. A zero low-pass
   is valid only without decimation.
12. Writes the corrected data, windowed markers, before/after PSD figures, and a
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
Cancellation runs after the output low-pass, on a low-passed reference, as
`fmrib_fastr.m` does. Ordering matters more than it appears. Measured on this
cohort with the two orderings differing in nothing else, cancelling before the
low-pass leaves the reference carrying artifact out to the input Nyquist, which
shrinks the `0.05/(N*var(refs))` step size until the filter barely adapts: the
in-passband slice-harmonic residual fell only 20 % on `sub-0001` run 1 and 2 % on
`sub-0000` run 6. With the reference band-limited the filter adapts as intended
and those become 63 % and 36 %.

That extra suppression is not a better correction. The same runs show what it
costs. On-comb probes retained 2.0 to 6.6 % of their amplitude, against 10.6 to
19.5 % under the weaker ordering. The damage is not confined to the comb either:
off-comb probes at 33.7 Hz and 70.3 Hz retained 83.2 % and 80.5 %, where the
mis-ordered canceller left them at about 103 %. Band-limiting the reference
concentrates the filter's taps inside the EEG band, so it fits and subtracts EEG.
Only a 7 Hz probe was untouched at 104 %.

The residual improvement and the signal loss are one mechanism seen from two
sides: the canceller removes narrowband content near the comb without regard to
its origin, so a lower residual line is not evidence that it removed artifact.
`processing.adaptive_noise_cancellation` therefore defaults to false, and a
residual measurement alone must not be used to justify enabling it.

Because the reference has to be band-limited to mean anything, enabling the
canceller with `processing.lowpass_hz: 0.0` is rejected. `fmrib_fastr.m` silently
forces a 70 Hz cutoff in that case; overriding a cutoff the configuration states
is worse than refusing the pair.
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

## Where acquisition groups come from

Correction needs one trigger per acquisition group, and the number of groups in a
volume. A recording supplies these one of two ways, declared by
`timing.marker_kind`, and exactly one source of truth is accepted in each case.

**Volume markers** (`marker_kind: volume`) locate only the volume. Where each
group fires inside it is derived from declared slice timing: `RepetitionTime`,
`SliceTiming`, and `MultibandAccelerationFactor`, read either from a BIDS sidecar
at `input.fmri_metadata` or from an `acquisition:` section in the YAML. The two
carry identical fields through identical validation, so an inline declaration is
not a weaker one -- it exists because real sidecars often omit `SliceTiming` or
`MultibandAccelerationFactor`, and hand-editing a JSON file to satisfy a tool is
worse provenance than declaring the timing where the run is configured.

**Acquisition-group markers** (`marker_kind: slice`) record the group positions
directly, as a slice-triggered recording does. Nothing in a marker series says
where a volume begins, so `timing.groups_per_volume` is declared; the repetition
time and the within-volume offsets are then measured from the markers. Declared
slice timing is rejected in this mode, since it could only contradict what the
recording says. Three properties are checked, each catching a distinct failure:
the marker count must divide into whole volumes; the derived volume starts must
be one repetition time apart, because one missing or extra group marker moves
every later boundary; and each slot's offset must repeat across volumes, because
slot matching averages one acquisition time and drifting offsets would mix
different slots.

A `groups_per_volume` that is wrong but still divides the marker count is
self-consistent: counting two volumes' worth of groups measures twice the
repetition time and offsets that repeat just as well. The markers cannot refute
that reading, which is why the count must be declared rather than inferred, and
why `timing.expected_repetition_time_seconds` exists to check it. The likeliest
mistake it catches is counting slices where the scanner marks excitations.

Neither route infers acquisition geometry from the EEG waveform. The sidecar
records which route ran, the timing declared to it, and the geometry resolved
from it, so a measured offset can be told from a derived one after the fact.

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
analysis choices include the marker convention and its timing source, the
interpolation factor, template neighbour count,
search radius, relative trigger position, marker-repair policy, template high-pass,
output filter/rate, exact line-noise frequencies, non-EEG channel names, residual
OBS rank and section duration, ANC, residual threshold, optional gate/adaptive
settings, trim mode, residual-QC block, mains, and volume-spectrum settings, and
PSD limits. Line regression deliberately removes all signal
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

## Trying the pipeline without a recording

`eegfmri-fastr demo --output-dir DIR` writes a BrainVision recording carrying a
simulated multiband gradient artifact, marked both per volume and per
acquisition group, alongside a BIDS sidecar and a commented configuration. It
exists so that an installation can be exercised end to end, and so that the two
marker conventions can be compared on one recording. Its 10.5 Hz probe tone sits
off the volume-harmonic comb of its 0.9 s repetition time; a 10.0 Hz tone would
land on the ninth harmonic and be removed with the artifact, which is the 1/TR
limitation below rather than a defect. Simulated numbers are evidence about the
implementation, never about an acquisition.

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
