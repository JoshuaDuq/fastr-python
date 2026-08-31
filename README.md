# EEG-fMRI FASTR

`eegfmri-fastr` is a Python implementation of the FASTR scanner-gradient artifact
correction method for simultaneous EEG-fMRI recordings. It accepts a BrainVision
recording marked either once per volume or once per acquisition group, validates
the acquisition timing, applies acquisition-slot FASTR correction, and writes a
corrected BrainVision recording with preserved markers and provenance.

The distribution is `eegfmri-fastr`, the command is `eegfmri-fastr`, and the
importable package is `eegfmri_fastr`.

This is research software. Inspect the provenance and validate the correction for
each acquisition protocol before using the output for inference.

It is not the FMRIB EEGLAB plug-in and is not affiliated with, sponsored by, or
endorsed by the FMRIB Centre or the University of Oxford. The project is released
under GPL-2.0-only; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Installation

Python 3.12 is the tested interpreter. Install the runtime and development
dependencies with `uv`, which reproduces the exact versions in `uv.lock`:

```text
uv sync
```

Or install the package into an existing environment. Runtime dependencies are
declared as compatible ranges, so it installs beside other analysis tools:

```text
uv pip install .
```

## Try it without a recording

Write a self-contained synthetic dataset -- a BrainVision recording carrying a
simulated multiband gradient artifact, both marker conventions, a BIDS sidecar,
and a commented configuration -- and correct it:

```text
eegfmri-fastr demo --output-dir /path/to/demo
eegfmri-fastr run --config /path/to/demo/demo.yml
```

The recording is simulated, so it demonstrates the interface and nothing about
any acquisition. It carries a 10.5 Hz tone placed off the volume-harmonic comb,
which a correct run keeps while suppressing the artifact.

## Correct a recording

Copy [`examples/configuration.yml`](examples/configuration.yml), set the paths and
processing values for the recording, then run:

```text
eegfmri-fastr run --config /path/to/configuration.yml
```

All important run settings are defined in YAML rather than hardcoded. The strict
configuration rejects ambiguous markers, invalid timing, unsuitable filters, and
existing output files. The run produces BrainVision files, before/after PSD figures,
and a JSON provenance sidecar. `line_noise_frequencies_hz` is required: use
`[60.0]` for explicit 60 Hz sinusoidal regression or `[]` to retain every line.
Missing volume markers can be repaired only when the YAML explicitly selects
`repair` and supplies the expected volume count; only uniquely located interior
markers are accepted. If one BrainVision marker stream contains multiple scanner
acquisitions, configure `timing.volume_marker_start_index` and
`timing.volume_marker_count` together to select the exact block belonging to the
BOLD run. Selection happens before timing validation and never repairs or bridges
a gap inside the chosen block.
Explicit marker selection requires `trim.mode: first_to_last_volume`, so later
unmatched acquisitions cannot remain uncorrected in the emitted recording.

`processing.channel_adaptive_window` is an opt-in alternative to the shared
reference-channel adaptive window. It keeps the configured wide template on
stable channels and uses `local_neighbor_count` only for groups where that
channel's template-fit residual improves by `adaptive_improvement_ratio`.
Non-EEG channels remain on the wide, unscaled correction. The residual-QC
sidecar reports isolated channel/block outliers by default; these advisory flags
do not change samples or create bad-interval annotations. Multichannel motion
still requires downstream review, rejection, or interpolation.

`processing.local_window_channels` is the explicit mode for a confirmed,
isolated channel failure. Listed EEG channels use `local_neighbor_count` for
every acquisition group, while all other channels retain the configured wide
window. It cannot be combined with either adaptive-window mode. For the
provided residual dataset, forcing AF4 to 20 neighbours reduced the peak
scanner-locked residual from 59.9 to 3.5 uV in `sub-0000` run 4 and from 51.9
to 9.7 uV in run 6. Because this is deliberately targeted, channel names must
exist in every recording processed by that configuration.

`processing.channel_failure_policy` is the automatic form of that same repair,
for recordings whose bad channel is not known in advance. The default `report`
changes nothing. `retry_local_and_recommend_bad` measures the corrected output,
retries the channels that failed with `local_neighbor_count`, installs a retry
only when it is materially better, and writes the channels that still fail into
`fastr.channel_failure_policy.recommended_bad_channels` in the JSON sidecar.

The policy detects isolated scanner-gradient residuals only. It does not detect
every bad-electrode condition, never interpolates or drops channels, and writes
`recommended_bad_channels` solely to JSON provenance for downstream review or
interpolation.

Four rules make up the decision, and all four are in the sidecar:

- **Candidate.** A channel-block is a candidate when its residual exceeds both
  `quality_control.bad_channel_residual_uv` and the median of the other EEG
  channels in the *same block* by `quality_control.residual_mad_multiplier`
  robust sigma. The comparison is spatial, not temporal, because a channel that
  was bad for the whole run has no good past to stand out from. A uniformly
  noisy block raises the median with it and nominates nobody. Non-EEG channels
  are outside the statistic and can never be nominated.
- **Frozen threshold.** Those per-block thresholds are computed once, from the
  wide pass, and every later comparison uses them. A retry is judged against the
  bar its own channel failed, not against one the retry itself moved.
- **Acceptance.** A local retry is installed only when it both fails fewer
  blocks and lowers that channel's worst residual to at most 85 % of the wide
  value. Counting blocks alone would install a retry that nudged a residual from
  just above the threshold to just below it; comparing maxima alone would
  install one that halved a spike while leaving every failed block failed. A
  rejected retry is discarded whole: the wide samples and every diagnostic
  belonging to them are kept.
- **Persistence.** After retries, a channel is recommended bad when it still
  fails at least two blocks and at least a tenth of them. One bad block stays
  advisory. If the recommended channel is the one named by
  `processing.reference_channel`, `reference_channel_recommended_bad` is true in
  the sidecar: alignment was fitted on that channel, so the whole run deserves a
  second look.

The policy owns the local window it decides to install, so it cannot be combined
with `adaptive_window`, `channel_adaptive_window`, or `local_window_channels`,
and it requires `quality_control.report_channel_outliers` and a
`local_neighbor_count` smaller than `neighbor_count`.

On the 144-run residual dataset the spatial rule nominates a channel in 24 runs,
and the automatic policy reaches the same answer the hand-written
`local_window_channels` configuration above reached: AF4 in `sub-0000` run 4
falls from 59.9 to 3.5 uV and its five failed blocks to none, and in run 6 from
51.9 to 9.7 uV. In `sub-0013` run 6 the retry of AF7 is installed on a smaller
gain — 41.4 to 31.6 uV, four failed blocks to three — and the channel is still
recommended bad, which is the intended outcome: the improvement is kept and the
channel is still flagged for review. FC6 in both `sub-0000` runs falls from
about 15 to 7 uV without clearing the threshold in its one failed block, so the
retry is rejected and the wide samples are kept unchanged; the sidecar records
both numbers, so overriding that with `local_window_channels` remains a
deliberate, informed choice.

### Where the acquisition timing comes from

Exactly one source, declared in the YAML. Naming two is an error rather than a
precedence rule, because the second could only contradict the first.

| Recording marks | `timing.marker_kind` | Timing source |
| --- | --- | --- |
| one event per volume | `volume` | `input.fmri_metadata`, a BIDS sidecar |
| one event per volume | `volume` | the `acquisition:` section, for a sidecar missing `SliceTiming` or `MultibandAccelerationFactor` |
| one event per acquisition group | `slice` | the markers themselves, plus `timing.groups_per_volume` |

With volume markers, where each acquisition group fires inside a volume is
derived from the declared slice timing. With acquisition-group markers those
positions are recorded, so only the number of groups in a volume has to be
declared -- nothing in a marker series says where a volume begins. The
repetition time and the within-volume offsets are then measured and checked for
the periodicity that acquisition-slot matching depends on. A wrong
`groups_per_volume` that still divides the marker count is self-consistent, so
set `timing.expected_repetition_time_seconds` to catch it. Every resolved
number, and whether it was declared or measured, is written to the provenance
sidecar.

## Compare uncorrected vs FASTR folders

Pair an uncorrected scanner-artifact folder with FASTR-corrected recordings
and write PSD/epoch overlays plus a CSV of band-power ratios:

```text
eegfmri-fastr compare --config examples/compare.yaml
```

Recordings pair by the identifier left after stripping the suffixes each
processing stage appends. Those suffixes, the subject directory prefix, and the
run-ordering rule are all declared in the config's `naming:` section: no export
convention is assumed, and the two sides pair only once the naming in use is
named. The example carries one lab's BrainVision Analyzer convention; replace it.

## Validate timing only

Resolve the acquisition geometry and write it out, correcting nothing. Volume
markers are checked against declared slice timing:

```text
eegfmri-fastr validate-timing \
  --metadata /path/to/bold.json \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Volume \
  --marker-description volume-start \
  --output /path/to/timing-validation.json
```

Acquisition-group markers are measured instead, so they take no metadata:

```text
eegfmri-fastr validate-timing \
  --marker-kind slice \
  --groups-per-volume 18 \
  --expected-repetition-time-seconds 0.9 \
  --sampling-rate 5000 \
  --vhdr /path/to/raw.vhdr \
  --marker-type Slice \
  --marker-description slice-start \
  --output /path/to/timing-validation.json
```

The command fails on missing or duplicate markers, marker gaps, excessive timing
jitter, an inconsistent TR-to-sample conversion, and — for acquisition-group
markers — a partial volume, a shifted volume boundary, or offsets that do not
repeat. It does not infer markers from the EEG waveform. The report records the
resolved repetition time, groups per volume, and group offsets.

## Method

The pipeline implements the acquisition-group variant of FASTR for multiband data,
following the published method and using the
[FMRIB fMRIb FASTR implementation](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m)
as a reference. It is a separate Python implementation, not a drop-in port of the
FMRIB plug-in. See [`docs/algorithm.md`](docs/algorithm.md) for the processing model,
limitations, and configuration details. The pipeline exposes fixed, automatic, or
disabled residual OBS; sectioned basis fitting; FMRIB-style adaptive noise
cancellation; relative trigger positioning; explicit marker repair; and an optional
output low-pass. ANC is opt-in because it can remove genuine narrowband activity
near scanner harmonics. See
[`docs/fmrib-parity-validation.md`](docs/fmrib-parity-validation.md) for the complete
MATLAB source audit and real-recording comparison.

## Development

Run the test and lint checks:

```text
uv run pytest
uv run ruff check src tests validation
git diff --check
```

See [`docs/validation.md`](docs/validation.md) for the validation checklist.

`metrics`, `diagnostics`, `simulation`, and `matlab_comparison` are validation
instrumentation. They are imported by the tests, the demo, and the validation
runners, never by the correction pipeline.

## Related pipelines

Cardiac detection and AAS/PCA-OBS BCG correction live in **BCG-Correction**
(`bcg-correct`). The deep-learning BCGNet path lives in **BCGNet-Python**
(`bcgnet`). This package only removes scanner-gradient artifacts.
