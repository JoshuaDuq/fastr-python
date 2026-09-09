# Algorithm

## Scope and assumptions

FASTR removes scanner-gradient artifact from simultaneous EEG-fMRI recordings.
It requires a BrainVision recording, an exact scanner-marker stream, and one
declared acquisition-timing interpretation. BCG, bad electrodes, motion, and
other physiological artifacts are outside its scope. See the
[validation checklist](validation.md).

## Processing model

For one configured run, FASTR:

1. Reads the BrainVision files without overwriting them.
2. Selects the configured marker type and description, optionally selecting a
   contiguous volume-marker block.
3. Resolves acquisition geometry from declared volume timing or measured
   acquisition-group markers.
4. Validates marker spacing, ordering, and complete artifact epochs.
5. Interpolates the signal for sub-sample alignment.
6. Estimates target-excluding acquisition-slot templates and fits one amplitude
   per channel.
7. Applies optional residual gating and adaptive/local-window policies.
8. Applies optional residual OBS over complete whole-volume epochs.
9. Applies optional normalized LMS adaptive noise cancellation.
10. Low-passes and decimates the corrected signal when configured, then applies
    stationary line-noise regression to EEG channels.
11. Resamples markers into the emitted output window and annotates skipped
    gradient spans and residual-QC blocks.
12. Writes BrainVision output, PSD figures, and JSON provenance.

Correction and filtering use the available input context before the output
window is sliced. Incomplete boundary groups remain uncorrected, are recorded
in provenance, and receive a `Bad_Gradient` marker.

## Trimming and boundary margin

`trim.mode: none` emits the full recording. `first_to_last_volume` starts at the
first selected volume marker and ends one declared TR after the last selected
marker, clipped to the recording end. Correction and filtering still run on the
untrimmed input first. A partial final TR is retained, with uncorrected boundary
samples marked `Bad_Gradient`.

The final volume is not synthesized when its artifact epoch is incomplete. A
marker gap is an error by default. Explicit repair can fill uniquely located
interior volume markers when an expected count is supplied. See the
[configuration reference](configuration.md#trim).

## Template estimation and alignment

The moving template and least-squares amplitude follow
[Niazy et al. (2005)](references.md#niazy-et-al-2005). For each acquisition
slot, neighboring volumes are averaged after excluding the target volume. A
high-pass copy is used for template estimation and shared alignment; the
estimate is subtracted from the original signal so slow content is retained.

Alignment uses the configured reference channel and is applied consistently to
all channel batches. `neighbor_count` must be even. Local modes require a
smaller even `local_neighbor_count`. Adaptive decisions are recorded in
provenance.

## Acquisition timing and geometry

With `marker_kind: volume`, one marker starts each volume. Group positions are
expanded from `RepetitionTime`, `SliceTiming`, and
`MultibandAccelerationFactor`, read from one BIDS JSON sidecar or the inline
`acquisition` section. These fields follow the
[BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html).

With `marker_kind: slice`, every acquisition group is marked. Marker positions
provide group offsets; `groups_per_volume` declares the number of groups per
volume. FASTR measures repetition time and checks repeated spacing and
within-volume offsets. It does not infer the group count.

The timing sources are not merged. A configuration selects exactly one valid
interpretation so provenance can distinguish declared from measured timing.

## Non-EEG channels

Channels in `processing.non_eeg_channels` default to `ECG`. Their template
subtraction uses an unscaled estimate. They are excluded from residual OBS,
ANC, line-noise regression, and residual-QC channel statistics. Output filtering,
decimation, and trimming still apply.

## Residual OBS and adaptive noise cancellation

Residual OBS is optional. It fits a fixed or automatic rank over complete
whole-volume epochs; `residual_obs_section_seconds` refits by sections.
Automatic selection rejects an unstable rank rather than choosing one silently.

Optional normalized LMS ANC follows the
[FMRIB `fmrib_fastr.m` implementation](references.md#fmrib-fastr-implementation)
and uses the low-passed artifact estimate as its reference. ANC can remove
narrowband EEG near scanner harmonics, so assess residual suppression with
signal transfer. Zero-variance and divergent states raise an error.

## Output filtering and decimation

The output low-pass is a zero-phase FIR designed through MNE-Python and applied
before integer decimation. Its passband edge must be below both input and output
Nyquist frequencies. The transition width is the smaller of MNE's automatic
width and the distance from the passband edge to output Nyquist. This keeps the
stopband at or below output Nyquist without moving the requested passband edge;
narrow transitions require longer filters. The same filter is applied to the
ANC artifact reference. A zero cutoff is allowed only when the output rate
equals the input rate. See [MNE filtering documentation](references.md#mne-python).

Frequencies in `line_noise_frequencies_hz` are regressed from EEG channels
after filtering and decimation. An empty list disables regression.

## Quality control and provenance

Residual QC measures acquisition-group harmonic excess in microvolts over complete
blocks. Temporal flags count EEG channels only; optional spatial
flags identify isolated channel-block outliers. The failure policy may retry a
candidate with a local window and recommend a bad channel. It never drops or
interpolates channels.

The separate `residual_qc.volume_harmonic_qc` report covers integer multiples of
`1 / RepetitionTime`, including harmonics below the group rate. Each complete
block uses a detrended rectangular window and Fourier-series normalization to
report the RMS of components at the declared harmonic frequencies. Mains
collisions are excluded and only EEG channels enter this report. Its flags use
the same configured floor, robust multiplier, and minimum EEG-channel count,
but remain separate from group-residual flags and automatic local retries.
These measurements include possible neural activity and must not be interpreted
as artifact alone. Finite block lengths also admit leakage from nearby frequencies;
longer blocks distinguish nearby frequencies more clearly. Rectangular windows
avoid counting a harmonic twice through adjacent harmonics even for one-volume
blocks. Measurements are reported in JSON, without automatic data rejection.

The whole-run volume spectrum evaluates power at exact harmonic frequencies with
SciPy ZoomFFT and Welch density normalization. Local peak searches use a separate
Welch spectrum. Segment durations are sampled on the output grid; acquisition
periods and harmonic frequencies are not rounded. Fractional output samples per
volume are supported, such as TR 0.9 s at 625 Hz. Zero padding samples the local
peak search on short recordings but does not improve their spectral resolution.

The JSON sidecar records input SHA-256 hashes, resolved timing and geometry,
configuration, output window, alignment, correction counts, PSD interval,
residual measurements, channel decisions, and runtime. `software_environment`
records Python, NumPy, SciPy, MNE, and pybv versions. Preserve the sidecar with
the corrected recording.

## The 1/TR limitation

Scanner artifact and its harmonics are locked to the acquisition period:
`1 / RepetitionTime` and integer multiples. Neural or physiological activity at
those frequencies cannot be separated by frequency alone. Template, OBS, and
ANC stages can reduce signal as well as artifact. Report suppression with an
independent signal-transfer measure.

For example, continuous 10 Hz activity is exactly the ninth volume harmonic at
TR 0.9 s and may be removed with the artifact, even while nearby 10.5 Hz activity
survives. Validate the frequencies and time structure used by the intended
analysis, including bursts and task-locked responses. Lower residual power alone
does not establish that EEG was preserved.

## Known limitations

- Marker errors, timing gaps, and incompatible timing sources are not repaired
  implicitly.
- Boundary groups without complete epochs remain uncorrected.
- A fixed artifact model may not capture motion or scanner-state changes;
  inspect block-level residuals.
- Residual QC is advisory and does not establish that a channel is unusable.
- BCG correction and general bad-channel handling are outside this package.
- This Python implementation is compared with, but is not a drop-in
  replacement for, the FMRIB EEGLAB interface.

## Trying the pipeline without a recording

Generate and correct the deterministic synthetic demo:

```text
fastr-python demo --output-dir /path/to/demo
fastr-python run --config /path/to/demo/demo.yml
```

The demo validates the software path and an injected off-comb signal. It is not
protocol-specific validation.

## Inputs and units

Input and output use BrainVision Core Data Format. Signal arrays follow MNE's
volt convention. BIDS timing and configuration durations use seconds;
frequencies use hertz; residual reports use microvolts; internal sample indices
are zero-based; and BrainVision marker positions on disk are one-based.

## External references

See [References](references.md) for the FASTR paper, BIDS specification,
MNE-Python, BrainVision format, SciPy diagnostics, and the FMRIB source audit.
