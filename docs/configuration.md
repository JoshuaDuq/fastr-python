# Configuration reference

Configurations are YAML mappings loaded by `fastr_python.config.load_config`.
Relative paths resolve from the YAML file's directory. Unknown fields, missing
fields, invalid types, and incompatible combinations raise
`ConfigurationError`. Loading does not create files or require the input to
exist.

## Timing source rule

Timing is explicit and exclusive. For `marker_kind: volume`, use exactly one of
`input.fmri_metadata` or the inline `acquisition` section. For
`marker_kind: slice`, use neither: group positions are measured from the
recording and `groups_per_volume` is declared. The BIDS fields
`RepetitionTime`, `SliceTiming`, and `MultibandAccelerationFactor` follow the
[BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html).

## Top-level sections

| Section | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `input` | mapping | required | paths | Requires `raw_vhdr`; may contain `fmri_metadata`. |
| `output` | mapping | required | path | Requires `vhdr`; derived outputs use its stem. |
| `timing` | mapping | required | marker metadata | Requires `marker_type` and `marker_description`. |
| `acquisition` | mapping | absent | BIDS timing | Valid only as the volume-marker timing source. |
| `processing` | mapping | required | mixed | FASTR and optional-stage settings. |
| `quality_control` | mapping | absent | mixed | Missing fields use the defaults below. |
| `diagnostics` | mapping | absent | mixed | Missing fields use the defaults below. |
| `trim` | mapping | absent | mode | Defaults to `none`. |

## input

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `input.raw_vhdr` | string/path | required | path | BrainVision `.vhdr`; referenced `.eeg` and `.vmrk` files must be present beside it. |
| `input.fmri_metadata` | string/path or null | `null` | path | BIDS timing source for volume markers; mutually exclusive with `acquisition` and invalid for slice markers. |

## output

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `output.vhdr` | string/path | required | path | Must end in `.vhdr`; `.vhdr`, `.eeg`, `.vmrk`, `.json`, and PSD outputs must not exist. |

## timing

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `timing.marker_type` | string | required | BrainVision marker type | Exact, case-sensitive selection. |
| `timing.marker_description` | string | required | BrainVision description | Exact, case-sensitive selection. |
| `timing.marker_kind` | `volume` or `slice` | `volume` | — | `volume`: one marker per volume. `slice`: one marker per acquisition group. |
| `timing.groups_per_volume` | positive integer or null | `null` | groups/volume | Required for `slice`; invalid for `volume`. Declared, not inferred. |
| `timing.expected_repetition_time_seconds` | positive number or null | `null` | seconds | Optional check for measured slice timing; invalid for `volume`. |
| `timing.missing_volume_markers` | `error` or `repair` | `error` | — | Volume markers only. `repair` fills uniquely located interior gaps and requires `expected_volume_count`. |
| `timing.expected_volume_count` | positive integer or null | `null` | volumes | Required with `missing_volume_markers: repair`; invalid with `error`. |
| `timing.volume_marker_start_index` | nonnegative integer or null | `null` | zero-based marker index | Start of an explicit contiguous volume-marker block; use with `volume_marker_count`. Invalid for slice markers and repair mode. |
| `timing.volume_marker_count` | positive integer or null | `null` | markers | Length of an explicit volume-marker block; use with `volume_marker_start_index`. |

## acquisition

The inline section carries the same timing fields as the BIDS sidecar and uses
the same validation.

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `acquisition.repetition_time_seconds` | positive number | required | seconds | BIDS `RepetitionTime`; must convert to an integer number of input samples. |
| `acquisition.slice_timing_seconds` | nonempty list of numbers | required | seconds | BIDS `SliceTiming`; offsets must be nonnegative and less than the repetition time. |
| `acquisition.multiband_acceleration_factor` | positive integer | required | slices/group | BIDS `MultibandAccelerationFactor`; timing length must be divisible by it and each unique offset must occur exactly that many times. |

## processing

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `processing.method` | string | required | — | Must be `acquisition_group_fastr`. |
| `processing.interpolation_factor` | positive integer | required | samples/grid | Temporal factor for sub-sample alignment. |
| `processing.neighbor_count` | positive even integer | required | volumes/groups | Wide moving-template width. |
| `processing.search_radius_samples` | nonnegative integer | required | input samples | Alignment search radius around each trigger. |
| `processing.pre_trigger_fraction` | number | `0.03` | fraction | Trigger location within the artifact epoch; must be in `[0, 1]`. |
| `processing.lowpass_hz` | nonnegative number | required | Hz | Anti-alias cutoff below both Nyquist frequencies unless zero. Zero is allowed only without decimation. |
| `processing.output_sampling_rate_hz` | positive number | required | Hz | Input/output rates must have an integer ratio; output cannot exceed input. |
| `processing.channel_batch_size` | positive integer | required | channels/batch | Controls memory use without changing the numerical path. |
| `processing.reference_channel` | string or integer | required | channel name/index | Alignment reference; names and indices must be valid. |
| `processing.line_noise_frequencies_hz` | list of positive numbers | required | Hz | Frequencies below output Nyquist; `[]` disables regression. Applied to EEG after filtering/decimation. |
| `processing.non_eeg_channels` | list of strings | `[ECG]` | channel names | Excluded from template scaling, residual OBS, ANC, line-noise regression, and residual-QC statistics. |
| `processing.template_high_pass_hz` | nonnegative number | `1.0` | Hz | High-pass for template estimation and alignment; `0.0` uses the unfiltered estimate. |
| `processing.residual_threshold_uv` | nonnegative number | `1.0` | µV | Legacy absolute residual threshold for residual gating; does not replace robust QC thresholds. |
| `processing.residual_gate` | boolean | `false` | — | Excludes extreme residual volumes from clean-neighbour templates. |
| `processing.residual_obs` | boolean | `false` | — | Enables residual optimal-basis correction after template subtraction. |
| `processing.residual_obs_rank` | positive integer or `auto` | `4` | components | Fixed OBS rank or FMRIB-style automatic selection. |
| `processing.residual_obs_section_seconds` | positive number or null | `null` | seconds | Refit OBS by sections; null fits one basis for the run. |
| `processing.adaptive_noise_cancellation` | boolean | `false` | — | Enables normalized LMS ANC against the filtered artifact reference. Requires nonzero `lowpass_hz`; use only after signal-transfer checks. |
| `processing.adaptive_window` | boolean | `false` | — | Chooses wide or local windows from reference-channel residuals. Incompatible with other local modes. |
| `processing.channel_adaptive_window` | boolean | `false` | — | Chooses wide or local windows per EEG channel. Incompatible with other local modes. |
| `processing.local_neighbor_count` | positive even integer | `20` | volumes/groups | Local window width; must be smaller than `neighbor_count` for local modes. |
| `processing.local_window_channels` | list of strings | `[]` | channel names | Forces the local window for named EEG channels; names must exist and cannot be non-EEG. |
| `processing.residual_gate_mad_multiplier` | positive number | `8.0` | robust sigma | Residual-gate outlier multiplier. |
| `processing.residual_gate_ratio` | positive number | `8.0` | ratio | Maximum allowed fraction of residual-gated volumes. |
| `processing.residual_gate_max_fraction` | number in `(0, 1]` | `0.02` | fraction | Upper bound on excluded volumes. |
| `processing.adaptive_improvement_ratio` | number in `(0, 1]` | `0.85` | ratio | Local residual must be at most this fraction of the wide score. |
| `processing.channel_failure_policy` | `report` or `retry_local_and_recommend_bad` | `report` | — | Retry policy owns its local window; requires channel-outlier reporting and is incompatible with other local/adaptive modes. Never drops or interpolates a channel. |

## quality_control

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `quality_control.block_seconds` | positive number | `30.0` | seconds | Residual-QC block duration, rounded to complete volumes. |
| `quality_control.mains_frequency_hz` | positive number | `60.0` | Hz | Mains frequency excluded from scanner-harmonic attribution. |
| `quality_control.mains_exclusion_hz` | nonnegative number | `1.0` | Hz | Width around mains harmonics excluded from attribution. |
| `quality_control.residual_mad_multiplier` | nonnegative number | `6.0` | robust sigma | Per-channel temporal residual multiplier for coherent block flags. |
| `quality_control.residual_minimum_channels` | positive integer | `4` | channels | Minimum simultaneous EEG channels for a residual-block flag. |
| `quality_control.volume_spectrum_max_hz` | positive number | `110.0` | Hz | Highest reported volume harmonic, capped by output Nyquist. |
| `quality_control.report_channel_outliers` | boolean | `true` | — | Reports isolated channel/block outliers; does not alter samples. Required by the automatic channel-failure policy. |
| `quality_control.bad_channel_residual_uv` | positive number | `5.0` | µV | Absolute floor for spatial channel-failure candidates. |

## diagnostics

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `diagnostics.psd_max_frequency_hz` | positive number | `100.0` | Hz | PSD limit, capped by output Nyquist. |
| `diagnostics.psd_n_fft` | positive integer or null | `null` | FFT samples | Optional PSD FFT length; null uses the diagnostic default. |

## trim

| Field | Type | Required or default | Units | Rules |
| --- | --- | --- | --- | --- |
| `trim.mode` | `none` or `first_to_last_volume` | `none` | — | `none` emits the full recording. The other mode emits the span from the first through last selected volume marker after correction. Explicit volume-marker selection requires it. |

## Interaction rules

- Use one timing source. Volume markers need BIDS metadata or inline timing;
  slice markers need neither and require `groups_per_volume`.
- Repair requires an expected count and only fills uniquely located interior
  gaps. Explicit marker selection and repair are incompatible.
- `neighbor_count` and `local_neighbor_count` are even. Local modes require
  the local count to be smaller than the wide count.
- `adaptive_window`, `channel_adaptive_window`, and nonempty
  `local_window_channels` are mutually exclusive. The automatic channel
  failure policy is also incompatible with them.
- A nonzero low-pass must be below both Nyquist frequencies. Decimation needs
  anti-alias filtering; zero cutoff is valid only when rates are equal.
- Values select or enable stages; invalid inputs raise errors rather than
  triggering a fallback.

## Units and channel names

BIDS timing and configuration durations use seconds. Signal arrays use volts
following MNE conventions; residual reports use microvolts (`µV`). Internal
sample indices are zero-based; BrainVision marker positions are one-based on
disk. Channel names are exact strings, including spaces and capitalization.
