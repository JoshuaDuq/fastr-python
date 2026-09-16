# Configuration Reference

FASTR-Python configurations are defined in YAML mappings and parsed by `fastr_python.config.load_config`.

All relative filesystem paths resolve relative to the directory of the YAML configuration file. Loading performs strict validation: missing required fields, unknown keys, invalid value types, out-of-bounds ranges, or contradictory parameter combinations immediately raise `ConfigurationError`. Configuration parsing is pure and does not write to disk.

---

## 1. Minimal YAML Templates

### Volume-Triggered Recording (BIDS Timing Source)

Use this pattern when your EEG recording contains **one scanner trigger per whole volume** (e.g. at the start of each TR), and BIDS fMRI metadata provides slice timing:

```yaml
input:
  raw_vhdr: "sub-01/eeg/sub-01_task-rest_eeg.vhdr"
  fmri_metadata: "sub-01/func/sub-01_task-rest_bold.json"

output:
  vhdr: "derivatives/fastr/sub-01/sub-01_task-rest_desc-fastr_eeg.vhdr"

timing:
  marker_type: "Response"
  marker_description: "R128"
  marker_kind: "volume"

processing:
  method: "acquisition_group_fastr"
  interpolation_factor: 10
  neighbor_count: 30
  search_radius_samples: 5
  pre_trigger_fraction: 0.03
  lowpass_hz: 70.0
  output_sampling_rate_hz: 500.0
  channel_batch_size: 16
  reference_channel: "Cz"
  line_noise_frequencies_hz: [60.0]
  non_eeg_channels: ["ECG"]
```

### Slice-Triggered Recording (Measured Marker Timing)

Use this pattern when your EEG recording contains **one scanner trigger per slice or multiband acquisition group**:

```yaml
input:
  raw_vhdr: "sub-01/eeg/sub-01_task-rest_eeg.vhdr"

output:
  vhdr: "derivatives/fastr/sub-01/sub-01_task-rest_desc-fastr_eeg.vhdr"

timing:
  marker_type: "Response"
  marker_description: "R128"
  marker_kind: "slice"
  groups_per_volume: 18
  expected_repetition_time_seconds: 0.9

processing:
  method: "acquisition_group_fastr"
  interpolation_factor: 10
  neighbor_count: 30
  search_radius_samples: 5
  pre_trigger_fraction: 0.03
  lowpass_hz: 70.0
  output_sampling_rate_hz: 500.0
  channel_batch_size: 16
  reference_channel: "Cz"
  line_noise_frequencies_hz: [60.0]
  non_eeg_channels: ["ECG"]
```

---

## 2. Top-Level Sections

| Section | Type | Required? | Purpose |
| --- | --- | :---: | --- |
| [`input`](#input) | mapping | Yes | Path to raw BrainVision recording and optional BIDS metadata. |
| [`output`](#output) | mapping | Yes | Path to destination `.vhdr` file (stem governs derived outputs). |
| [`timing`](#timing) | mapping | Yes | Marker selection and acquisition geometry source (`volume` vs `slice`). |
| [`acquisition`](#acquisition) | mapping | Optional | Inline BIDS acquisition timing (alternative to external BIDS JSON). |
| [`processing`](#processing) | mapping | Yes | Numerical FASTR parameters (AAS, OBS, ANC, filtering, decimation). |
| [`quality_control`](#quality_control) | mapping | Optional | Parameters for residual and harmonic QC block detection. |
| [`diagnostics`](#diagnostics) | mapping | Optional | PSD estimation parameters for before/after summary plots. |
| [`trim`](#trim) | mapping | Optional | Recording window trimming mode (defaults to `none`). |

---

## 3. Section Specifications

### input

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `input.raw_vhdr` | string / path | *required* | path | Path to existing BrainVision `.vhdr`. Companion `.eeg` and `.vmrk` files must exist alongside it. |
| `input.fmri_metadata` | string / path or null | `null` | path | Path to BIDS JSON sidecar. Allowed **only** when `timing.marker_kind: volume`; mutually exclusive with the inline `acquisition` section. |

---

### output

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `output.vhdr` | string / path | *required* | path | Target `.vhdr` path. Must end with `.vhdr`. Destination files (`.vhdr`, `.eeg`, `.vmrk`, `.json`, `_psd_*.png`) must not already exist. |

---

### timing

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `timing.marker_type` | string | *required* | string | BrainVision marker type (e.g. `Response`, `Stimulus`). Exact case-sensitive match. |
| `timing.marker_description` | string | *required* | string | BrainVision description (e.g. `R128`, `V`). Exact case-sensitive match. |
| `timing.marker_kind` | `volume` or `slice` | `volume` | — | `volume`: One marker per whole volume TR. `slice`: One marker per slice/multiband group. |
| `timing.groups_per_volume` | integer > 0 or null | `null` | groups/volume | **Required** when `marker_kind: slice`; forbidden when `marker_kind: volume`. Declared by operator, never inferred. |
| `timing.expected_repetition_time_seconds` | number > 0 or null | `null` | seconds | Optional validation check for slice markers; forbidden when `marker_kind: volume`. |
| `timing.missing_volume_markers` | `error` or `repair` | `error` | — | Volume markers only. `repair` interpolates uniquely identifiable interior missing volume markers. |
| `timing.expected_volume_count` | integer > 0 or null | `null` | volumes | **Required** when `missing_volume_markers: repair`; forbidden with `error`. |
| `timing.volume_marker_start_index` | integer $\ge 0$ or null | `null` | index | Zero-based index of first volume marker in an explicit contiguous block. Requires `volume_marker_count`. |
| `timing.volume_marker_count` | integer > 0 or null | `null` | markers | Number of volume markers in explicit contiguous block. Requires `volume_marker_start_index`. |

---

### acquisition

Used **only** when `timing.marker_kind: volume` as an alternative to an external BIDS JSON sidecar. Conforms strictly to the [BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html).

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `acquisition.repetition_time_seconds` | number > 0 | *required* | seconds | BIDS `RepetitionTime`. Must convert to an integer number of raw EEG samples ($T_R \times f_{s,\text{in}} \in \mathbb{Z}^+$). |
| `acquisition.slice_timing_seconds` | list of numbers | *required* | seconds | BIDS `SliceTiming`. Non-empty list of slice offsets in $[0, T_R)$. |
| `acquisition.multiband_acceleration_factor` | integer > 0 | *required* | factor | BIDS `MultibandAccelerationFactor`. Number of slices acquired simultaneously. `len(slice_timing_seconds)` must be divisible by this factor. |

---

### processing

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `processing.method` | string | *required* | — | Must be `acquisition_group_fastr`. |
| `processing.interpolation_factor` | integer > 0 | *required* | factor | Temporal upsampling factor for sub-sample cross-correlation alignment (typically 10). |
| `processing.neighbor_count` | even integer > 0 | *required* | epochs | Number of adjacent epochs included in the moving average template (excluding the target epoch). Must be even. |
| `processing.search_radius_samples` | integer $\ge 0$ | *required* | input samples | Search window around expected trigger for cross-correlation alignment (typically 3–5 samples). |
| `processing.pre_trigger_fraction` | number | `0.03` | fraction | Fraction of the epoch prior to the trigger timestamp. Must be in $[0.0, 1.0]$. |
| `processing.lowpass_hz` | number $\ge 0$ | *required* | Hz | Output low-pass filter cutoff. Must be strictly below both input and output Nyquist frequencies. Can be `0.0` only when not decimating. |
| `processing.output_sampling_rate_hz` | number > 0 | *required* | Hz | Output sampling rate. Input rate must be an exact integer multiple of output rate ($f_{s,\text{in}} / f_{s,\text{out}} \in \mathbb{Z}^+$). |
| `processing.channel_batch_size` | integer > 0 | *required* | channels | Channel chunk size for batch processing. Conserves RAM without changing numerical outputs. |
| `processing.reference_channel` | string or integer | *required* | channel | Channel used to estimate shared sub-sample alignment delays across channels. Must exist in montage. |
| `processing.line_noise_frequencies_hz` | list of numbers > 0 | *required* | Hz | Mains frequencies below output Nyquist regressed from EEG channels post-filtering. `[]` disables regression. |
| `processing.non_eeg_channels` | list of strings | `["ECG"]` | channel names | Channels excluded from template amplitude scaling, OBS, ANC, line regression, and residual QC statistics. |
| `processing.template_high_pass_hz` | number $\ge 0$ | `1.0` | Hz | High-pass cutoff applied to signal copies before template estimation and alignment. `0.0` disables. |
| `processing.residual_threshold_uv` | number $\ge 0$ | `1.0` | $\mu\mathrm{V}$ | Absolute noise floor threshold for residual and volume harmonic QC flags. |
| `processing.residual_gate` | boolean | `false` | — | Excludes outlier residual volumes from moving templates of neighboring volumes. |
| `processing.residual_gate_mad_multiplier` | number > 0 | `8.0` | robust $\sigma$ | Multiplier on median absolute deviation (MAD) for residual gating. |
| `processing.residual_gate_ratio` | number > 0 | `8.0` | ratio | Ratio of local residual to background level for residual gating. |
| `processing.residual_gate_max_fraction` | number in $(0, 1]$ | `0.02` | fraction | Maximum allowable fraction of excluded volumes before error is raised. |
| `processing.residual_obs` | boolean | `false` | — | Enables Optimal Basis Set (OBS) PCA residual artifact projection. |
| `processing.residual_obs_rank` | integer > 0 or `auto` | `4` | rank | Number of principal components to project out, or `auto` for FMRIB eigenvalue slope heuristic. |
| `processing.residual_obs_section_seconds` | number > 0 or null | `null` | seconds | Time interval over which OBS bases are re-estimated. `null` fits one basis across the run. |
| `processing.adaptive_noise_cancellation` | boolean | `false` | — | Enables normalized LMS adaptive noise cancellation against artifact reference. Requires `lowpass_hz > 0`. |
| `processing.adaptive_window` | boolean | `false` | — | Dynamically chooses wide vs local template window based on reference channel residuals. Mutually exclusive with other local window modes. |
| `processing.channel_adaptive_window` | boolean | `false` | — | Dynamically chooses wide vs local template window independently per channel. Mutually exclusive with other local window modes. |
| `processing.local_neighbor_count` | even integer > 0 | `20` | epochs | Template window width for local mode. Must be strictly less than `neighbor_count`. |
| `processing.local_window_channels` | list of strings | `[]` | channel names | Forces local window on named EEG channels. Mutually exclusive with adaptive modes. |
| `processing.adaptive_improvement_ratio` | number in $(0, 1]$ | `0.85` | ratio | Threshold ratio of local to wide residual required to switch to local window. |
| `processing.channel_failure_policy` | `report` or `retry_local_and_recommend_bad` | `report` | — | Strategy when a channel exhibits residual failure. `retry_local_and_recommend_bad` attempts a local window retry and adds channel to recommendations. Never drops or interpolates data. |

---

### quality_control

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `quality_control.block_seconds` | number > 0 | `30.0` | seconds | Block duration for residual QC, rounded to complete volume TRs. |
| `quality_control.mains_frequency_hz` | number > 0 | `60.0` | Hz | Mains line frequency excluded from scanner harmonic attribution. |
| `quality_control.mains_exclusion_hz` | number $\ge 0$ | `1.0` | Hz | Half-width around mains harmonics excluded from attribution. |
| `quality_control.residual_mad_multiplier` | number $\ge 0$ | `6.0` | robust $\sigma$ | MAD multiplier for coherent multi-channel residual detection. |
| `quality_control.residual_minimum_channels` | integer > 0 | `4` | channels | Minimum number of EEG channels simultaneously flagged to mark a block outlier. |
| `quality_control.volume_spectrum_max_hz` | number > 0 | `110.0` | Hz | Upper frequency limit for whole-run volume harmonic evaluation (capped at output Nyquist). |
| `quality_control.report_channel_outliers` | boolean | `true` | — | Detects and logs spatial channel outliers in QC report. |
| `quality_control.bad_channel_residual_uv` | number > 0 | `5.0` | $\mu\mathrm{V}$ | Absolute microvolt floor for candidate channel failure recommendations. |

---

### diagnostics

| Field | Type | Default | Units | Rules & Validation |
| --- | --- | --- | --- | --- |
| `diagnostics.psd_max_frequency_hz` | number > 0 | `100.0` | Hz | Maximum frequency plotted in pre/post Welch PSD figures (capped by output Nyquist). |
| `diagnostics.psd_n_fft` | integer > 0 or null | `null` | samples | FFT window length for PSD estimation. `null` selects an optimal power-of-2 default based on sampling rate. |

---

### trim

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `trim.mode` | `none` or `first_to_last_volume` | `none` | `none`: Retains and exports the entire raw recording length. Boundary samples outside valid artifact templates are marked `Bad_Gradient`.<br>`first_to_last_volume`: Trims output recording to start at the first selected volume marker and end exactly one $T_R$ after the last selected volume marker. |

---

## 4. Invariance & Mutual Exclusivity Rules

To prevent accidental misconfiguration, FASTR-Python enforces strict mutual exclusivity:

1. **Timing Source Exclusivity**:
   - `marker_kind: volume` $\implies$ Requires external `input.fmri_metadata` OR inline `acquisition`. `timing.groups_per_volume` is forbidden.
   - `marker_kind: slice` $\implies$ Requires `timing.groups_per_volume`. `input.fmri_metadata` and inline `acquisition` are strictly forbidden.
2. **Missing Marker Repair vs Manual Selection**:
   - `missing_volume_markers: repair` requires `expected_volume_count` and is incompatible with explicit volume slicing (`volume_marker_start_index` / `volume_marker_count`).
   - Slicing modes require `trim.mode: first_to_last_volume`.
3. **Window Policies**:
   - `adaptive_window`, `channel_adaptive_window`, and non-empty `local_window_channels` are mutually exclusive.
   - `channel_failure_policy: retry_local_and_recommend_bad` is incompatible with other local/adaptive window flags.
4. **Anti-Aliasing Constraints**:
   - If $f_{s,\text{in}} > f_{s,\text{out}}$, `lowpass_hz` must be $> 0$ and its transition band must reach full stopband attenuation strictly at or before output Nyquist ($f_{s,\text{out}} / 2$).
   - `lowpass_hz: 0.0` is permitted if and only if $f_{s,\text{in}} == f_{s,\text{out}}$ (no decimation).

