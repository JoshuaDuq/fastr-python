# Algorithm

## Scope and assumptions

FASTR removes scanner-gradient artifact from simultaneous EEG-fMRI EEG. It
requires a BrainVision recording with an exact scanner-marker stream and a
declared acquisition-timing interpretation. It does not identify BCG artifact,
bad electrodes, motion, or other physiological signal. Use the
[validation checklist](validation.md) for protocol-specific evidence.

## Processing model

For one configured run, the pipeline:

1. Reads the BrainVision header, data, and marker files without overwriting
   them.
2. Selects the exact configured marker type and description. An optional
   contiguous volume-marker block is selected before timing validation.
3. Resolves one acquisition geometry from declared volume timing or measured
   acquisition-group markers.
4. Validates volume spacing, marker ordering, and complete acquisition epochs.
5. Interpolates the signal for sub-sample alignment.
6. Estimates target-excluding, acquisition-slot templates from neighboring
   volumes and fits an amplitude for each channel.
7. Applies optional residual gating and adaptive/local-window policies.
8. Applies optional residual optimal-basis subtraction over whole-volume epochs.
9. Applies optional normalized LMS adaptive noise cancellation.
10. Low-passes and decimates the corrected signal when configured, then applies
    explicit stationary line-noise regression to EEG channels.
11. Resamples markers into the emitted output window and annotates skipped
    gradient spans and advisory residual-QC blocks.
12. Writes BrainVision output, before/after PSD figures, and JSON provenance.

Correction and filtering run over the available input context before the output
window is sliced. Boundary groups that lack a complete epoch remain
uncorrected, are recorded in provenance, and receive a `Bad_Gradient` marker.

## Trimming and boundary margin

`trim.mode: none` emits the full corrected recording. With
`first_to_last_volume`, the first and last selected volume markers define the
zero-based, half-open output span. The pipeline still corrects and filters the
untrimmed input first, preserving the context required by boundary epochs.

The final volume is not synthesized when the recording ends before its complete
artifact epoch. A marker gap is an error by default; explicit repair can fill
only uniquely located interior volume markers when an expected count is given.
See the [configuration reference](configuration.md#trim) for the exact rules.

## Template estimation and alignment

The moving template and least-squares amplitude follow the FASTR method of
[Niazy et al. (2005)](references.md#niazy-et-al-2005). For each acquisition
slot, neighboring volumes are averaged while excluding the target volume. A
configurable high-pass copy is used for template estimation and shared
alignment; the fitted estimate is subtracted from the original signal so slow
content is retained.

The alignment search is performed on the configured reference channel and then
applied consistently to the channel batches. `neighbor_count` must be even;
local modes use a smaller even `local_neighbor_count`. Adaptive choices are
explicitly recorded in provenance.

## Acquisition timing and geometry

With `marker_kind: volume`, one marker starts each volume. Group positions are
expanded from `RepetitionTime`, `SliceTiming`, and
`MultibandAccelerationFactor`, read from one BIDS JSON sidecar or the equivalent
inline `acquisition` section. These fields follow the
[BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html).

With `marker_kind: slice`, every acquisition group is marked. The marker
positions provide group offsets; `groups_per_volume` declares how many groups
make a volume. The pipeline measures repetition time and checks that volume
spacing and within-volume offsets repeat. It does not infer the group count.

The two sources are intentionally not merged or prioritized. A configuration
must select exactly one valid interpretation so provenance can distinguish
declared from measured timing.

## Non-EEG channels

Channels listed in `processing.non_eeg_channels` default to `ECG`. Their
template subtraction uses an unscaled estimate, and they are excluded from
residual OBS, ANC, line-noise regression, and residual-QC channel statistics.
This prevents physiological channel structure from biasing scanner-artifact
fits. Exclusion does not remove or alter those channels' recorded signal beyond
the configured template subtraction.

## Residual OBS and adaptive noise cancellation

Residual optimal-basis subtraction is optional. When enabled, the pipeline
fits the configured fixed or automatic rank over complete whole-volume epochs;
`residual_obs_section_seconds` can refit by sections. Automatic selection
rejects an unstable rank rather than silently choosing one.

Normalized LMS adaptive noise cancellation is also optional and follows the
[FMRIB `fmrib_fastr.m` implementation](references.md#fmrib-fastr-implementation).
The reference is the low-passed artifact estimate. ANC can remove genuine
narrowband EEG near scanner harmonics, so residual suppression must be assessed
together with signal transfer. Zero-variance and divergent states raise an
error.

## Output filtering and decimation

The output low-pass is a zero-phase FIR designed through MNE-Python and is
applied before integer decimation. The cutoff must be below both input and
output Nyquist frequencies. A zero cutoff is allowed only when the output rate
equals the input rate, because decimation without anti-alias filtering is
rejected. See [MNE filtering documentation](references.md#mne-python).

Stationary frequencies in `line_noise_frequencies_hz` are regressed from EEG
channels after filtering and decimation. An empty list disables that step.

## Quality control and provenance

Residual QC measures scanner-locked harmonic excess in microvolts over complete
blocks. Robust temporal flags identify coherent multi-channel blocks; optional
spatial flags identify isolated channel-block outliers. The automatic failure
policy may retry a candidate with a local window and recommend a persistent bad
channel, but it never drops or interpolates channels.

The JSON sidecar records input SHA-256 hashes, resolved timing, geometry,
configuration, output window, alignment, correction counts, PSD interval,
residual measurements, channel decisions, and runtime. Treat it as part of the
scientific output and preserve it with the corrected recording.

## The 1/TR limitation

The scanner-gradient artifact is strongly locked to the acquisition period, so
its fundamental and harmonics occur at `1 / RepetitionTime` and integer
multiples. Neural or physiological activity at those frequencies is not
distinguishable from scanner artifact by frequency alone. Template, OBS, and
ANC stages can therefore reduce signal as well as residual artifact. Report
both suppression and independent signal-transfer measurements.

## Known limitations

- Marker errors, timing gaps, and incompatible timing sources are not repaired
  implicitly.
- Boundary groups without complete epochs are left uncorrected.
- A fixed artifact model does not account for every motion or scanner-state
  change; inspect block-level residuals.
- Residual QC is advisory and does not establish that a channel is unusable.
- BCG correction and general bad-channel handling are outside this package.
- The Python implementation is scientifically compared with, but is not a
  drop-in replacement for, the FMRIB EEGLAB interface.

## Trying the pipeline without a recording

Generate and correct the deterministic synthetic demo:

```text
eegfmri-fastr demo --output-dir /path/to/demo
eegfmri-fastr run --config /path/to/demo/demo.yml
```

The demo validates the software path and an injected off-comb signal. It is not
a substitute for protocol-specific validation on real data.

## Inputs and units

The input and output formats are BrainVision Core Data Format files. Signal
arrays follow MNE's convention of volts. BIDS timing fields and configuration
durations use seconds; frequencies use hertz; residual reports use microvolts;
internal sample indices are zero-based; BrainVision marker positions on disk are
one-based.

## External references

See the [references](references.md) for the FASTR paper, BIDS specification,
MNE-Python, BrainVision format, SciPy diagnostics, and the FMRIB source audit.
