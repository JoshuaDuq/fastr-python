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
   configured number of neighboring volumes.
7. Fits group shifts once from the configured reference channel, then fits and
   subtracts a scaled template for every channel batch.
8. Applies the configured zero-phase low-pass filter and integer decimation.
9. Writes the corrected data, resampled markers, before/after PSD figures, and a
   provenance sidecar.

The first and last groups whose complete artifact epochs are unavailable are left
unchanged and recorded in the sidecar. PSD figures use the same interval containing
only complete corrected epochs, so uncorrected boundary data cannot dominate the
diagnostic. They are limited to 0--100 Hz and use standard MNE spatial channel colors
when channel positions can be identified. A marker gap or timing inconsistency is a
hard error; the pipeline does not interpolate missing acquisition events.

## Why acquisition slots matter

In multiband acquisitions, adjacent groups can represent different acquisition-time
slots. Matching templates by the unique values in `SliceTiming` prevents those
different temporal positions from being averaged together. The number of repeated
slice groups is derived from the metadata and checked against the multiband factor.

## The 1/TR limitation

An acquisition-locked signal at the volume repetition frequency, or at one of its
harmonics, is indistinguishable from a scanner artifact when both are represented by
the same volume-locked reference. Subtracting a template therefore can reduce a
genuine neural component at exactly `1 / RepetitionTime` as well as scanner energy.
No blind post-hoc notch can restore information that was removed during subtraction.

The template window is consequently a scientific parameter, not merely a speed
setting. The example value is a conservative starting point selected by comparing
residual artifact and signal-transfer behavior on representative recordings. It
must be revalidated when the scanner sequence, electrode montage, marker timing, or
recording protocol changes.

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
