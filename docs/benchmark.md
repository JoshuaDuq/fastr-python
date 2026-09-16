# Three-way correction benchmark

`benchmark/` compares FASTR-Python against MATLAB FMRIB FASTR and FACETpy, both
its averaging corrections and its pretrained networks, on identical recordings.
It is research instrumentation: it is not part of the installed package, and
nothing in `src/fastr_python/` imports it.

## What it measures

Five axes, of which the first two are never reported apart.

| Axis | Measurement |
| --- | --- |
| Scanner-artifact suppression | Harmonic-locked residual per channel in 30 s blocks |
| Neuronal signal preservation | Transfer of injected tones, on and between volume harmonics |
| Sensitivity to movement | Residual against per-block framewise displacement |
| Processing time | Wall clock and peak resident memory, one machine, per arm |
| Robustness | Failure rate and residual spread across participants |

Signal transfer is measured by difference. Each arm corrects the recording
twice, once plain and once on a copy carrying tones of known amplitude and
phase, and the two corrections are subtracted. The corrected recording still
holds residual artifact and real EEG at the probe frequency, so projecting one
pass alone would measure all three at once.

Tones are placed at two distances from the artifact comb: exactly on a volume
harmonic, the worst case for any comb-shaped correction, and halfway between
two, the best. Reporting a single averaged transfer would sit between the two
and describe neither.

## Arms

| Arm | Method |
| --- | --- |
| `fastr_python` | acquisition-slot FASTR at the cohort's settings |
| `matlab_fmrib` | `fmrib_fastr.m` at one trigger per volume |
| `facetpy_volume_averaged` | FACETpy's own default shape, one artifact per volume |
| `ml_*` | one arm per pretrained FACETpy network |

`fmrib_fastr` is driven at volume triggers because that is what it is built
for. Given one trigger per acquisition group it suppresses nothing measurable:
it has no acquisition-slot concept, so averaging a group against its neighbours
in a multiband sequence averages different slice sets together. Reporting that
as an implementation result would report a missing capability as a broken one.

FACETpy runs at its own documented shape for the same reason. It does ship a
slot-matched mode, but it cannot represent this cohort's acquisition: it cuts
every epoch to one fixed length, taken from the median trigger spacing, and
subtracts each template in place and cumulatively. The multiband slots here are
spaced 237, 238, 250 and 350 samples apart, so no single length tiles a volume.
At the inferred 250 the epochs overlap and about half the slot boundaries are
subtracted twice, leaving 113.7 uV where the other arms leave 2.4; at 237, the
widest length that cannot overlap, the gaps still leave 66.9.

Neither reference implementation, then, can represent non-uniform multiband
acquisition timing -- FMRIB has no acquisition-slot concept at all, and
FACETpy's fixed-length epochs cannot tile an unevenly spaced volume. That is a
result of the comparison rather than a configuration to keep tuning, and it is
why no arm here is a pure implementation comparison.

## What is held constant

Every arm is handed the same acquisition geometry, resolved once from the same
marker stream and BIDS sidecar, so no two arms can disagree about where the
scanner fired.

No arm applies its own output filter. Each tool ships a different one --
`fmrib_fastr.m` builds a least-squares FIR and runs it through `filtfilt`,
reaching -4.4 dB at 90 Hz where this project's once-applied FIR is flat -- and
probe tones reach 93 Hz, so per-tool filtering would report a filter as a
correction. The arms correct at the recorded rate, and one shared anti-alias
low-pass and decimation is applied to all of them afterwards.

Every arm is then cropped to one span of an even number of whole volumes. Whole
volumes keep each artifact harmonic on an exact Fourier bin; the even count
does the same for the between-harmonic tones, which complete whole cycles only
over an even number of repetitions. One whole volume is dropped at each end,
sized from the filter actually designed, because the shared filter smears its
own edge and this project's arm starts its output on the first volume marker.

## Failures are results

An arm that fails on a recording is recorded as having failed, and the run
continues to the next arm. Nothing is retried and nothing is substituted. How
often an arm fails, and on which recordings, is the robustness measurement, so
it is written to the outcome log exactly as a success is.

## Running it

MATLAB, FACETpy, and the pretrained models are each optional; omitting a flag
omits its arms.

```text
python -m benchmark.cli select --config benchmark.json
python -m benchmark.cli run --config benchmark.json \
  --matlab /Applications/MATLAB_R2026a.app/bin/matlab \
  --eeglab "/path/to/EEGLAB" \
  --facetpy /path/to/facetpy-env/bin/python \
  --facetpy-source /path/to/FACETpy
python -m benchmark.cli report --config benchmark.json
```

`select` freezes the chosen recordings to a manifest that hashes each header
and marker stream before anything is corrected against it. `--participant`
narrows it to one participant for a trial run without changing what is chosen,
since stratification always ranks a participant against their own runs.

The configuration names four roots and how many runs each participant
contributes:

```json
{
  "source_root": "/path/to/source_data",
  "confounds_root": "/path/to/fmriprep",
  "protocol_root": "/path/to/bids/fmri",
  "output_root": "/path/to/benchmark_output",
  "runs_per_participant": 3
}
```

Recordings are chosen by ranking each participant's task runs on mean
framewise displacement and taking evenly spaced strata, so the spread within a
participant is motion rather than everything that differs between people.

## Environments

FACETpy pins `numpy==2.1.3` and `mne==1.10.2` against this project's versions
and is GPL-3.0-only against its GPL-2.0-only, so it runs in its own
interpreter behind a subprocess boundary and the benchmark aggregates the two
rather than combining them. Its deep-learning code is not in the published
package and its model weights are Git LFS objects, so both come from a clone;
each weight file is fetched over HTTPS and checked against the SHA-256 its
pointer declares.

Statistics need `pandas` and `statsmodels`, installed with
`uv sync --group benchmark`. Correction itself needs neither.

## Reading the result

Suppression alone does not rank a correction. On every recording measured so
far, the arm with the lowest residual also keeps the least signal at the
frequencies it cleaned, so a ranking on residual inverts the ranking on
transfer. Read both columns, and read the two tone placements apart.
