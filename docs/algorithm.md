# Algorithm

FASTR-Python implements an acquisition-slot scanner-gradient correction pipeline.
The implementation is designed around explicit timing metadata and exact marker
validation so that acquisition geometry is never guessed from the EEG waveform.

## Processing model

For each run, the pipeline:

1. Reads the BrainVision header and losslessly parses its marker file.
2. Selects volume-start markers using the exact configured marker type and
   description.
3. Loads `RepetitionTime`, `SliceTiming`, and
   `MultibandAccelerationFactor` from the BIDS fMRI JSON.
4. Validates that volume markers are contiguous at the declared TR and expands each
   volume start into fractional acquisition-group triggers using the unique slice
   timing offsets.
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
   It then fits and subtracts a scaled template for every channel batch. The
   estimate is subtracted from the unfiltered channel, so slow content survives
   correction.
8. Applies the configured zero-phase low-pass filter, then takes the output window
   and decimates.
9. Writes the corrected data, windowed markers, before/after PSD figures, and a
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

Niazy et al. (2005) build the moving-average template, and fit the least-squares
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

## Stages this implementation does not perform

FASTR as published has four stages. This pipeline performs the first two: trigger
alignment and moving-average template subtraction, which the paper reports removes
more than 98 % of the artifact. Residual removal by optimal basis set is available
as `residual_obs` but is not part of the pipeline, and adaptive noise cancellation
is not implemented. Neither absence is hidden by the sidecar, which records exactly
what ran. PSD figures use the same interval containing
only complete corrected epochs, so uncorrected boundary data cannot dominate the
diagnostic. They use the configured PSD frequency limit, capped at the output
Nyquist, and use standard MNE spatial channel colors when channel positions can be
identified. A marker gap or more than one native
sample of timing inconsistency is a hard error; the pipeline does not interpolate
missing acquisition events. A single sample (0.2 ms at 5 kHz) is treated as clock
quantization inside the alignment search.

## Why acquisition slots matter

In multiband acquisitions, adjacent groups can represent different acquisition-time
slots. Matching templates by the unique values in `SliceTiming` prevents those
different temporal positions from being averaged together. The number of repeated
slice groups is derived from the metadata and checked against the multiband factor.

## Configuration and provenance

The example YAML is the single user-facing configuration surface. Protocol and
analysis choices include the interpolation factor, template neighbour count,
search radius, template high-pass, output filter/rate, residual threshold, optional
gate/adaptive settings, trim mode, residual-QC block and mains settings, and PSD
limits. Fixed implementation details include the interpolation kernel shape, the
residual OBS 70 Hz high-pass design, and the protected boundary volumes. The
provenance sidecar stores the resolved configuration plus the effective PSD limit,
FFT length, and residual-QC mains/block settings used for the run.

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
- [FASTR implementation reference](https://github.com/jesuslmc/FMRIB-FASTR)
