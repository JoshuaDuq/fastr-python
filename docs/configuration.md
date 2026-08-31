# Configuration reference

## File format and path resolution

Configurations are YAML mappings loaded by `fastr_python.config.load_config`.
Relative paths are resolved against the directory containing the YAML file;
they are not resolved against the current shell directory. Unknown fields,
missing required fields, invalid scalar types, and incompatible combinations
raise `ConfigurationError`. Loading a configuration does not create files or
require the input recording to exist.

## Timing source rule

The timing source is explicit and exclusive. With `marker_kind: volume`, set
exactly one of `input.fmri_metadata` or the inline `acquisition` section. With
`marker_kind: slice`, set neither: acquisition-group marker positions are
measured from the recording, while `groups_per_volume` is declared. The
`RepetitionTime`, `SliceTiming`, and `MultibandAccelerationFactor` fields
follow the [BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html).

## Top-level sections

| Section | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `input` | mapping | required | paths | Must contain `raw_vhdr`; may contain `fmri_metadata`. |
| `output` | mapping | required | path | Must contain `vhdr`; all derived outputs use its stem. |
| `timing` | mapping | required | marker metadata | Must contain `marker_type` and `marker_description`. |
| `acquisition` | mapping | absent | BIDS timing | Valid only as the volume-marker timing source. |
| `processing` | mapping | required | mixed | Contains all required FASTR and optional stage settings. |
| `quality_control` | mapping | absent | mixed | Missing fields use the defaults below. |
| `diagnostics` | mapping | absent | mixed | Missing fields use the defaults below. |
| `trim` | mapping | absent | mode | Defaults to `none`. |

## input

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `input.raw_vhdr` | string/path | required | path | BrainVision `.vhdr`; the referenced `.eeg` and `.vmrk` files must be local to the header and present at run time. |
| `input.fmri_metadata` | string/path or null | default `null` | path | BIDS JSON timing source for volume markers; mutually exclusive with `acquisition` and invalid for slice markers. |

## output

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `output.vhdr` | string/path | required | path | Must end in `.vhdr`; the `.vhdr`, `.eeg`, `.vmrk`, `.json`, and PSD outputs must not already exist. |

## timing

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `timing.marker_type` | string | required | BrainVision marker type | Exact, case-sensitive marker selection. |
| `timing.marker_description` | string | required | BrainVision description | Exact, case-sensitive marker selection. |
| `timing.marker_kind` | `volume` or `slice` | default `volume` | — | `volume` means one marker starts each volume; `slice` means one marker records each acquisition group. |
| `timing.groups_per_volume` | positive integer or null | default `null` | groups/volume | Required for `slice`; invalid for `volume`. It is declared rather than inferred. |
| `timing.expected_repetition_time_seconds` | positive number or null | default `null` | seconds | Optional check for measured slice-marker timing; invalid for `volume`. |
| `timing.missing_volume_markers` | `error` or `repair` | default `error` | — | Applies only to volume markers. `repair` fills uniquely located interior gaps and requires `expected_volume_count`. |
| `timing.expected_volume_count` | positive integer or null | default `null` | volumes | Required with `missing_volume_markers: repair`; invalid with `error`. Boundary markers cannot be inferred. |
| `timing.volume_marker_start_index` | nonnegative integer or null | default `null` | zero-based marker index | Selects the start of an explicit contiguous volume-marker block; configure with `volume_marker_count`. Invalid for slice markers and repair mode. |
| `timing.volume_marker_count` | positive integer or null | default `null` | markers | Selects the length of an explicit volume-marker block; configure with `volume_marker_start_index`. |

## acquisition

The inline section carries the same three timing concepts as the BIDS sidecar;
it goes through the same validation. Use it when a sidecar does not provide
complete slice timing metadata.

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `acquisition.repetition_time_seconds` | positive number | required | seconds | Corresponds to BIDS `RepetitionTime`; must convert to an integer number of input samples. |
| `acquisition.slice_timing_seconds` | nonempty list of numbers | required | seconds | Corresponds to BIDS `SliceTiming`; each offset is nonnegative and less than the repetition time. |
| `acquisition.multiband_acceleration_factor` | positive integer | required | slices/group | Corresponds to BIDS `MultibandAccelerationFactor`; slice timing length must be divisible by it and every unique offset must occur exactly that many times. |

## processing

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `processing.method` | string | required; `acquisition_group_fastr` | — | The supported configuration-driven method. |
| `processing.interpolation_factor` | positive integer | required | samples/grid | Temporal interpolation factor used for sub-sample alignment. |
| `processing.neighbor_count` | positive even integer | required | volumes/groups | Wide moving-template width; must be even. |
| `processing.search_radius_samples` | nonnegative integer | required | input samples | Alignment search radius around each trigger. |
| `processing.pre_trigger_fraction` | number, default `0.03` | fraction | fraction of epoch | Relative trigger location within the artifact epoch; must be in `[0, 1]`. |
| `processing.lowpass_hz` | nonnegative number | required | Hz | Output anti-alias cutoff; must be below both input and output Nyquist frequencies unless zero. Zero is allowed only without decimation. |
| `processing.output_sampling_rate_hz` | positive number | required | Hz | Output rate; input/output rates must have an integer ratio and output may not exceed input. |
| `processing.channel_batch_size` | positive integer | required | channels/batch | Controls memory use without changing the numerical path. |
| `processing.reference_channel` | string or integer | required | channel name/index | Alignment reference; names must exist and integer indices must be in range. |
| `processing.line_noise_frequencies_hz` | list of positive numbers | required | Hz | Stationary sinusoidal regression frequencies below output Nyquist; use `[]` to disable. Applied to EEG channels after output filtering/decimation. |
| `processing.non_eeg_channels` | list of strings, default `[ECG]` | channel names | channels | Excluded from template scaling, residual OBS, ANC, line-noise regression, and residual-QC channel statistics. |
| `processing.template_high_pass_hz` | nonnegative number, default `1.0` | Hz | High-pass used for template estimation and alignment; `0.0` uses the unfiltered estimate. |
| `processing.residual_threshold_uv` | nonnegative number, default `1.0` | µV | Legacy absolute residual threshold used by residual gating. It does not replace the robust QC thresholds. |
| `processing.residual_gate` | boolean, default `false` | — | Excludes extreme residual volumes from clean-neighbour templates; disabled by default. |
| `processing.residual_obs` | boolean, default `false` | — | Enables residual optimal-basis correction after template subtraction. |
| `processing.residual_obs_rank` | positive integer or `auto`, default `4` | components | Fixed OBS rank or FMRIB-style automatic rank selection. |
| `processing.residual_obs_section_seconds` | positive number or null, default `null` | seconds | Refit OBS by sections; null fits one basis over the run. |
| `processing.adaptive_noise_cancellation` | boolean, default `false` | — | Enables normalized LMS ANC against the filtered artifact reference; opt in only after signal-transfer checks. Requires nonzero `lowpass_hz`. |
| `processing.adaptive_window` | boolean, default `false` | — | Chooses wide or local windows from the shared reference-channel residual. Mutually exclusive with channel-adaptive and explicit local modes. |
| `processing.channel_adaptive_window` | boolean, default `false` | — | Chooses wide or local windows independently per EEG channel. Mutually exclusive with `adaptive_window` and explicit local channels. |
| `processing.local_neighbor_count` | positive even integer, default `20` | volumes/groups | Local moving-template width; must be even and smaller than `neighbor_count` for local modes. |
| `processing.local_window_channels` | list of strings, default `[]` | channel names | Forces the local window across the run for named EEG channels; names must exist and cannot be non-EEG channels. |
| `processing.residual_gate_mad_multiplier` | positive number, default `8.0` | robust sigma | Residual-gate outlier multiplier. |
| `processing.residual_gate_ratio` | positive number, default `8.0` | ratio | Maximum allowed fraction of residual-gated volumes. |
| `processing.residual_gate_max_fraction` | number in `(0, 1]`, default `0.02` | fraction | Upper bound on how many volumes the residual gate may exclude. |
| `processing.adaptive_improvement_ratio` | number in `(0, 1]`, default `0.85` | ratio | Local window must reduce the residual to this fraction of the wide score. |
| `processing.channel_failure_policy` | `report` or `retry_local_and_recommend_bad`, default `report` | — | Retry policy owns the local window it installs; it requires channel-outlier reporting and cannot combine with other local/adaptive window modes. It never drops or interpolates a channel. |

## quality_control

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `quality_control.block_seconds` | positive number, default `30.0` | seconds | Residual-QC block duration, rounded to complete volumes. |
| `quality_control.mains_frequency_hz` | positive number, default `60.0` | Hz | Mains frequency excluded from scanner-harmonic attribution. |
| `quality_control.mains_exclusion_hz` | nonnegative number, default `1.0` | Hz | Width around mains harmonics excluded from attribution. |
| `quality_control.residual_mad_multiplier` | nonnegative number, default `6.0` | robust sigma | Per-channel temporal residual multiplier used for coherent block flags. |
| `quality_control.residual_minimum_channels` | positive integer, default `4` | channels | Minimum simultaneous EEG channels required to flag a residual block. |
| `quality_control.volume_spectrum_max_hz` | positive number, default `110.0` | Hz | Highest reported volume harmonic, capped by output Nyquist. |
| `quality_control.report_channel_outliers` | boolean, default `true` | — | Reports isolated channel/block outliers in provenance; does not alter samples. Required by the automatic channel-failure policy. |
| `quality_control.bad_channel_residual_uv` | positive number, default `5.0` | µV | Absolute floor for automatic spatial channel-failure candidates. |

## diagnostics

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `diagnostics.psd_max_frequency_hz` | positive number, default `100.0` | Hz | PSD plot/report limit, capped by output Nyquist. |
| `diagnostics.psd_n_fft` | positive integer or null, default `null` | FFT samples | Optional PSD FFT length; null uses the diagnostic default. |

## trim

| Field | Type | Required or default | Units | Validation and interaction |
| --- | --- | --- | --- | --- |
| `trim.mode` | `none` or `first_to_last_volume`, default `none` | — | `none` emits the full corrected recording; `first_to_last_volume` emits the span from the first through last selected volume marker after correction. Explicit volume-marker selection requires the latter. |

## Interaction rules

- Keep exactly one timing source. Volume markers need BIDS metadata or inline
  acquisition timing; slice markers need neither and require
  `groups_per_volume`.
- `missing_volume_markers: repair` requires an expected count and repairs only
  uniquely located interior gaps. Explicit marker selection and repair cannot
  be combined.
- `neighbor_count` and `local_neighbor_count` are even. Any local mode requires
  the local count to be smaller than the wide count.
- `adaptive_window`, `channel_adaptive_window`, and nonempty
  `local_window_channels` are pairwise incompatible. The automatic channel
  failure policy is also incompatible with all three.
- A nonzero low-pass must be below both Nyquist frequencies. Decimation needs
  that anti-alias filter; a zero cutoff is valid only when input and output
  rates are equal.
- Configuration values select or enable stages; they do not create a silent
  fallback when an input violates a requirement.

## Units and channel names

BIDS timing fields use seconds: `RepetitionTime` is the volume period and
`SliceTiming` contains within-volume offsets. Internally, signal arrays are in
volts following MNE conventions. Residual reports use microvolts (`µV`) for
readability. Sample indices are zero-based internally; BrainVision marker
positions on disk are one-based. Channel names are exact strings, including
spaces and capitalization.
