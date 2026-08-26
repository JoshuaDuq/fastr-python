# Independent ECG R-marker and BCG benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, ECG-only cardiac-event detector and a scientifically controlled BCG benchmark that compares FASTR-Python with the BrainVision Analyzer reference.

**Architecture:** Keep the existing scanner-gradient correction unchanged. Add a pure ECG detector that accepts only samples, sampling rate, and typed detector configuration; a separate BrainVision marker/audit layer; a BCG correction layer with AAS and MNE PCA-OBS; and a cohort benchmark that pairs FASTR-only inputs with Analyzer-corrected references. Analyzer markers are read only after independent detection for audit metrics.

**Tech Stack:** Python 3.12, NumPy, SciPy, MNE-Python 1.12.1, PyYAML, existing strict BrainVision I/O, pytest, Ruff, JSON/CSV provenance.

---

## Research basis and implementation decisions

The implementation should preserve these method distinctions:

- Allen et al. (1998) established heartbeat-locked pulse-artifact subtraction and emphasized reliable ECG peak detection and artifact-template quality. The local-average arm follows that family of methods: [paper](https://doi.org/10.1006/nimg.1998.0361).
- Niazy et al. (2005) introduced the optimal-basis-set approach: PCA basis functions are derived from heartbeat-locked artifact residuals and fitted to individual occurrences. The MNE arm uses the documented `apply_pca_obs` implementation: [paper](https://doi.org/10.1016/j.neuroimage.2005.06.067), [MNE API](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html).
- Abi-Abdallah et al. (2007) documented that the MRI magnetohydrodynamic effect can enlarge the T wave and obstruct R-peak detection. This is why amplitude-only peak picking is not acceptable here: [paper](https://pubmed.ncbi.nlm.nih.gov/18002339/).
- The FMRIB/EEGLAB detector provides the most directly relevant single-channel MRI reference: it combines a 7--40 Hz ECG representation, short smoothing, a k-Teager energy operator, Christov's combined adaptive MFR threshold, and a separate false-positive/false-negative correction with correlation alignment. The published Niazy method was validated on poor-quality ECG collected during fMRI: [paper](https://doi.org/10.1016/j.neuroimage.2005.06.067), [FMRIB reference implementation](https://github.com/sccn/fMRIb/blob/master/fmrib_qrsdetect.m), [peak correction implementation](https://github.com/sccn/fMRIb/blob/master/qrscorrect.m). The MATLAB source is GPL-licensed and is a reference for behavior, not source to copy.
- Christov's adaptive-threshold detector supplies the underlying MFR logic and was evaluated on all 48 full-length MIT-BIH arrhythmia records; the MRI-specific adaptation remains the relevant reference for this study: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC516783/).
- Brain Products documents CB Correction as supporting multiple R-peak strategies, including template/coherence matching, with configurable pulse-rate, correlation, delay, and pulse-interval settings. The supplied Analyzer output is consequently an empirical black-box comparator unless its transformation history/settings are available: [Brain Products documentation](https://pressrelease.brainproducts.com/sensor-data-analysis/), [Analyzer manual discussion](https://www.nmr.mgh.harvard.edu/~tatiana/BrainVisionManuals/RecView/20200204_RecView.pdf).
- Ganassin et al. (2024) demonstrated a patient-independent MRI R-peak strategy using ICA, derivative-based detection, adaptive thresholding, and automatic component selection. Their validation used multi-lead ECG without imaging gradients, so the plan uses the signal-processing principles but does not assume that single-channel study conditions transfer to this cohort: [paper](https://doi.org/10.1088/1361-6579/ad3b3d).
- Wong et al. (2018) detected cardiac cycles from an ICA-derived EEG BCG component. That is a useful independent audit concept, but it is outside the agreed production boundary because this detector must use ECG only: [paper](https://pubmed.ncbi.nlm.nih.gov/29614296/).
- MNE's official ECG and PCA-OBS examples are the package-level references for API behavior and correction validation: [ECG artifact workflow](https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html), [PCA-OBS example](https://mne.tools/stable/auto_examples/preprocessing/esg_rm_heart_artefact_pcaobs.html).

Niazy et al. validated QRS sensitivity and specificity against manual heartbeat counts.
Analyzer annotations cannot serve that role in this cohort because they are the reference
system being challenged and are known to miss beats. A detector-level sensitivity or
positive-predictive-value claim therefore requires blinded manual adjudication of a
stratified subset; otherwise the report must use marker agreement, ECG self-consistency,
and downstream held-out BCG residuals as its explicitly limited evidence.

The earlier `pain_study` implementation remains a design reference for QRS template construction, double-mark rejection, physiological interval checks, held-out event-locked metrics, and circular-shift nulls. Its Analyzer-seeded gap search, `read_analyzer_beats`, and marker-driven recovery path must not be called by the new production detector.

## File map

Create these focused modules:

- `src/mri_correction/bcg_config.py` — strict YAML dataclasses for one-recording detection and cohort benchmarking.
- `src/mri_correction/cardiac.py` — pure ECG conditioning, candidate generation, independent train selection, and detector QC.
- `src/mri_correction/cardiac_markers.py` — replacement of pulse markers, BrainVision sidecar copying, and one-to-one Analyzer audit.
- `src/mri_correction/cardiac_pipeline.py` — single-recording detection orchestration and provenance writing.
- `src/mri_correction/bcg.py` — local-average/AAS and MNE PCA-OBS correction on EEG picks, with explicit epoch confinement.
- `src/mri_correction/bcg_benchmark.py` — recording pairing, per-run orchestration, metrics, and JSON/CSV report writing.

Create these tests and fixtures:

- `tests/test_bcg_config.py`
- `tests/test_cardiac.py`
- `tests/test_cardiac_markers.py`
- `tests/test_bcg.py`
- `tests/test_bcg_benchmark.py`

Create or update these user-facing files:

- `examples/cardiac_detection.yml`
- `examples/bcg_benchmark.yml`
- `docs/bcg_methods.md`
- `docs/validation.md`
- `README.md`
- `src/mri_correction/cli.py`

The existing `src/mri_correction/fastr.py`, scanner-gradient correction path, and public
configuration schema must remain behaviorally unchanged except where the new command
registration is required.

### Task 1: Add strict cardiac/benchmark configuration

**Files:**
- Create: `src/mri_correction/bcg_config.py`
- Create: `tests/test_bcg_config.py`
- Create: `examples/cardiac_detection.yml`
- Create: `examples/bcg_benchmark.yml`

- [ ] **Step 1: Write failing configuration tests.** Cover the accepted shape, relative-path resolution, immutable dataclasses, unknown-key rejection, invalid frequency ordering, invalid interval ordering, unsupported correction methods, and non-existent paths being deferred until execution. The accepted YAML fields must be exactly:

```yaml
input:
  vhdr: data/gradient_corrected.vhdr
output:
  vhdr: output/with_independent_pulse_markers.vhdr
detector:
  ecg_channel: ECG
  preprocessing_band_hz: [7.0, 40.0]
  teager_emphasis_hz: 10.0
  teager_smoothing_seconds: 0.028
  template_window_seconds: [-0.2, 0.4]
  minimum_rr_seconds: 0.4
  maximum_rr_seconds: 1.5
  candidate_refractory_seconds: 0.25
  candidate_prominence_mad: 3.0
  correlation_threshold: 0.5
  refinement_iterations: 2

benchmark:
  fastr_root: /data/fastr_only
  analyzer_reference_root: /data/step3_bcg_corrected
  output_root: /data/bcg_benchmark
  marker_tolerance_seconds: 0.1
  correction_methods: [aas, pca_obs]
  correction_window_seconds: [-0.2, 0.7]
  aas_neighbor_count: 20
  pca_obs_components: 4
  null_surrogate_count: 20
  random_seed: 20260826
```

The detector document contains `input`, `output`, and `detector`. The benchmark document
contains top-level `benchmark` and `detector` blocks; the detector block has the same
fields in both documents. Do not make the parser accept arbitrary extra fields.

- [ ] **Step 2: Run the focused tests and verify the expected failure.**

Run:

```bash
uv run pytest tests/test_bcg_config.py -q
```

Expected: collection or import failure because `mri_correction.bcg_config` does not yet
exist.

- [ ] **Step 3: Implement the strict dataclasses and parsers.** Expose these exact public types and functions:

```python
@dataclass(frozen=True, slots=True)
class DetectorConfig:
    ecg_channel: str
    preprocessing_band_hz: tuple[float, float]
    teager_emphasis_hz: float
    teager_smoothing_seconds: float
    template_window_seconds: tuple[float, float]
    minimum_rr_seconds: float
    maximum_rr_seconds: float
    candidate_refractory_seconds: float
    candidate_prominence_mad: float
    correlation_threshold: float
    refinement_iterations: int


@dataclass(frozen=True, slots=True)
class DetectionRunConfig:
    input_vhdr: Path
    output_vhdr: Path
    detector: DetectorConfig


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    fastr_root: Path
    analyzer_reference_root: Path
    output_root: Path
    detector: DetectorConfig
    marker_tolerance_seconds: float
    correction_methods: tuple[str, ...]
    correction_window_seconds: tuple[float, float]
    aas_neighbor_count: int
    pca_obs_components: int
    null_surrogate_count: int
    random_seed: int


def load_detection_config(path: str | Path) -> DetectionRunConfig:
    raise NotImplementedError


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raise NotImplementedError
```

Use the existing `ConfigurationError` style: reject booleans where numbers are expected,
require finite positive frequencies, require `low < high`, require `minimum_rr <
maximum_rr`, require at least one correction method, and accept only `aas` and `pca_obs`.
Resolve relative paths relative to the YAML file without creating directories or checking
input existence during parsing.

- [ ] **Step 4: Run the focused tests and lint.**

Run:

```bash
uv run pytest tests/test_bcg_config.py -q
uv run ruff check src/mri_correction/bcg_config.py tests/test_bcg_config.py
```

Expected: all configuration tests pass and Ruff reports no findings.

- [ ] **Step 5: Commit the configuration boundary.**

```bash
git add src/mri_correction/bcg_config.py tests/test_bcg_config.py examples/cardiac_detection.yml examples/bcg_benchmark.yml
git commit -m "feat: add strict cardiac benchmark configuration"
```

### Task 2: Implement and test the independent ECG detector

**Files:**
- Create: `src/mri_correction/cardiac.py`
- Create: `tests/test_cardiac.py`

- [ ] **Step 1: Write failing synthetic detector tests.** Build a deterministic one-channel ECG fixture at 1 kHz containing known QRS complexes, amplitude drift, polarity inversion in a second fixture, T-wave-like deflections, noise, and a long interval in which a beat is present but an external marker train would be absent. Tests must assert that the detector returns the known QRS sample positions within 10 ms, does not return the T-wave positions, produces strictly increasing samples, and never places two events closer than `candidate_refractory_seconds`.

Include a structural independence test: call `detect_r_peaks` twice with exactly the same ECG and configuration and assert byte-identical peak samples and QC; there must be no function parameter for annotations or Analyzer markers. A separate test may construct arbitrary Analyzer marker arrays, but the detector call must not receive them.

Use a fixture helper with explicit known event times rather than a hidden random generator:

```python
def make_ecg(sampling_rate_hz: float, duration_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(round(duration_seconds * sampling_rate_hz), dtype=float)
    signal = np.zeros(samples.size, dtype=float)
    peak_seconds = np.array([0.8, 1.65, 2.53, 3.44, 4.37, 5.31])
    for index, peak_second in enumerate(peak_seconds):
        centre = peak_second * sampling_rate_hz
        width = 0.008 * sampling_rate_hz
        sign = -1.0 if index == 4 else 1.0
        signal += sign * np.exp(-0.5 * ((samples - centre) / width) ** 2)
        t_wave = centre + 0.28 * sampling_rate_hz
        signal += 0.65 * np.exp(-0.5 * ((samples - t_wave) / (0.035 * sampling_rate_hz)) ** 2)
    signal += 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    return signal, np.rint(peak_seconds * sampling_rate_hz).astype(np.int64)
```

- [ ] **Step 2: Run the detector tests and verify the expected failure.**

Run:

```bash
uv run pytest tests/test_cardiac.py -q
```

Expected: import failure because `mri_correction.cardiac` does not yet exist.

- [ ] **Step 3: Implement the pure detector with explicit validation.** Expose these exact public types and functions:

```python
@dataclass(frozen=True, slots=True)
class CardiacDetectionQuality:
    candidate_count: int
    accepted_count: int
    rejected_count: int
    median_rr_seconds: float
    rr_iqr_seconds: float
    minimum_rr_seconds: float
    maximum_rr_seconds: float
    implied_rate_bpm: float
    template_correlation_median: float
    rejected_low_correlation: int
    rejected_double_mark: int
    rejected_interval: int
    status: str


@dataclass(frozen=True, slots=True)
class CardiacDetection:
    peak_samples: npt.NDArray[np.int64]
    quality: CardiacDetectionQuality


def detect_r_peaks(
    ecg: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    config: DetectorConfig,
) -> CardiacDetection:
    raise NotImplementedError
```

Implement the algorithm in small private functions with the following fixed sequence:

1. Validate a one-dimensional finite numeric ECG, positive finite sampling rate, and a
   record long enough for the configured filter padding and template window.
2. Remove the robust centre and scale only for detector evidence; retain the original ECG
   for final timing localization.
3. Apply a zero-phase Butterworth band-pass over `preprocessing_band_hz`, smooth the
   result over `teager_smoothing_seconds`, and construct the nonnegative k-Teager energy
   complex lead. Derive `k` from `sampling_rate_hz` and `teager_emphasis_hz`, following
   the published MRI detector. Use `scipy.signal.sosfiltfilt`, and let an invalid
   short-record condition raise a specific detector input error.
4. Generate permissive QRS candidates with a deterministic implementation of the
   combined adaptive MFR threshold. Consolidate threshold crossings into local maxima
   and enforce the configured candidate refractory interval. Do not use a fixed
   amplitude threshold or any annotation-derived threshold.
5. Estimate the dominant period from the ECG-derived candidate intervals using a bounded
   interval-density mode. Detect and resolve a half-period cluster before constructing a
   template, so T-wave/double candidates cannot define the period.
6. Construct a mean-removed QRS template only from mutually consistent ECG-derived
   candidates. Reject the run explicitly when the configured minimum seed count cannot be
   reached; do not use Analyzer markers as seeds.
7. Compute normalized sliding template correlation over the conditioned ECG, choose local
   maxima with the period-derived refractory interval, and align each accepted event to the
   strongest original-ECG QRS feature in the local fitting window.
8. Refine the template and accepted train for exactly `refinement_iterations` passes.
   Reject low-correlation candidates, double detections, and RR intervals outside the
   configured physiological range. Record every rejection category in QC.

The implementation may adapt the prior `qrs_template`, normalized-correlation, modal
interval, and physiological-floor concepts, but every seed and threshold must originate
from the ECG vector passed to this function. Do not import MNE annotations in this module.

- [ ] **Step 4: Run the synthetic tests and inspect detector diagnostics.**

Run:

```bash
uv run pytest tests/test_cardiac.py -q
uv run ruff check src/mri_correction/cardiac.py tests/test_cardiac.py
```

Expected: all synthetic cases pass, including T-wave suppression, polarity handling,
determinism, and refractory guarantees.

- [ ] **Step 5: Commit the independent detector.**

```bash
git add src/mri_correction/cardiac.py tests/test_cardiac.py
git commit -m "feat: add independent ECG R-peak detector"
```

### Task 3: Write independent pulse markers and Analyzer audit metrics

**Files:**
- Create: `src/mri_correction/cardiac_markers.py`
- Create: `src/mri_correction/cardiac_pipeline.py`
- Create: `tests/test_cardiac_markers.py`

- [ ] **Step 1: Write failing marker and audit tests.** Test that existing `Pulse Artifact,R`
markers are replaced by detector events while all other markers are preserved, that
positions use BrainVision's one-based convention, that events outside the recording and
duplicate samples raise errors, and that writing refuses any existing output sidecar.

Test one-to-one audit matching with a marker train containing a nearby extra marker; one
Analyzer marker must not support multiple detected events. Test empty trains as explicit
audit results rather than exceptions.

- [ ] **Step 2: Run the focused tests and verify the expected failure.**

```bash
uv run pytest tests/test_cardiac_markers.py -q
```

Expected: import failure because `mri_correction.cardiac_markers` does not yet exist.

- [ ] **Step 3: Implement the marker/audit boundary.** Expose:

```python
PULSE_MARKER_TYPE = "Pulse Artifact"
PULSE_MARKER_DESCRIPTION = "R"


@dataclass(frozen=True, slots=True)
class MarkerAudit:
    analyzer_samples: npt.NDArray[np.int64]
    detected_samples: npt.NDArray[np.int64]
    matched_count: int
    tolerance_samples: int
    median_lag_samples: float | None
    lag_iqr_samples: float | None


@dataclass(frozen=True, slots=True)
class DetectionSummary:
    output_vhdr: Path
    provenance_json: Path
    marker_count: int
    status: str


def replace_pulse_markers(
    markers: Sequence[BrainVisionMarker],
    peak_samples: npt.ArrayLike,
    *,
    sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    raise NotImplementedError


def audit_marker_trains(
    analyzer_samples: npt.ArrayLike,
    detected_samples: npt.ArrayLike,
    *,
    tolerance_samples: int,
) -> MarkerAudit:
    raise NotImplementedError


def write_marker_recording(
    source_vhdr: str | Path,
    output_vhdr: str | Path,
    *,
    peak_samples: npt.ArrayLike,
) -> Path:
    raise NotImplementedError


def run_cardiac_detection(config: DetectionRunConfig) -> DetectionSummary:
    raise NotImplementedError
```

`replace_pulse_markers` must remove only exact `Pulse Artifact`/`R` markers and append the
new detector markers in sample order. `write_marker_recording` must copy `.eeg` and the
header into the destination, rewrite only the `DataFile` and `MarkerFile` references when
the output stem differs, and write the new strict marker file through the existing
BrainVision marker writer. It must not re-encode the data merely to add markers.

The audit function must sort validated samples, consume each marker at most once, and
report signed detector-minus-Analyzer lags for nearest markers. It is the only module
allowed to read Analyzer pulse annotations.

`run_cardiac_detection` belongs in `cardiac_pipeline.py`. It reads the configured
BrainVision recording, extracts the typed ECG samples, calls `detect_r_peaks` without
passing the annotation collection, replaces only the existing pulse markers, writes the
marker recording, and writes a sibling JSON provenance file containing the detector result.
It must not calculate a detector template from any marker annotation.

- [ ] **Step 4: Run the focused tests and perform a BrainVision round trip.**

```bash
uv run pytest tests/test_cardiac_markers.py tests/test_brainvision.py tests/test_brainvision_io.py -q
uv run ruff check src/mri_correction/cardiac_markers.py tests/test_cardiac_markers.py
```

Expected: all marker tests pass, and re-reading the written recording returns identical
`.eeg` bytes plus the expected replacement R markers.

- [ ] **Step 5: Commit marker output and audit.**

```bash
git add src/mri_correction/cardiac_markers.py tests/test_cardiac_markers.py
git commit -m "feat: write independent pulse markers and audits"
```

### Task 4: Implement bounded AAS and MNE PCA-OBS correction

**Files:**
- Create: `src/mri_correction/bcg.py`
- Create: `tests/test_bcg.py`

- [ ] **Step 1: Write failing correction tests.** Use a synthetic multichannel recording in
volts containing a known heartbeat-locked artifact, unrelated broadband signal, and a
separate ECG channel. Assert that both methods reduce held-out heartbeat-locked residual
energy, that samples outside the configured union of beat windows are unchanged, and that
the ECG channel is byte-identical. Test that PCA-OBS with fewer than `n_components + 1`
valid beats raises a specific `BcgInputError`.

- [ ] **Step 2: Run the focused tests and verify the expected failure.**

```bash
uv run pytest tests/test_bcg.py -q
```

Expected: import failure because `mri_correction.bcg` does not yet exist.

- [ ] **Step 3: Implement the correction API.** Expose:

```python
@dataclass(frozen=True, slots=True)
class BcgCorrectionConfig:
    method: str
    window_seconds: tuple[float, float]
    aas_neighbor_count: int
    pca_obs_components: int


@dataclass(frozen=True, slots=True)
class BcgCorrectionResult:
    data_volts: npt.NDArray[np.float64]
    corrected_samples: npt.NDArray[np.int64]
    method: str


def correct_bcg(
    data_volts: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    channel_names: Sequence[str],
    eeg_picks: npt.ArrayLike,
    ecg_channel_index: int,
    config: BcgCorrectionConfig,
) -> BcgCorrectionResult:
    raise NotImplementedError
```

Implement `_correct_aas` as local-neighbour heartbeat subtraction: extract complete
epochs, form the mean of the configured neighbouring epochs excluding the target, subtract
that template only from the target window, and blend overlapping windows by the existing
sample-wise correction accumulation rule. Do not use the target epoch in its own AAS
template.

Implement `_correct_pca_obs` with a temporary `mne.io.RawArray` containing only EEG picks,
call `mne.preprocessing.apply_pca_obs` with QRS times in seconds and the configured number
of components, then splice the corrected values only into the configured heartbeat-window
union. Restore the original ECG and all non-EEG channels exactly. Validate all array shapes,
finite values, event ordering, complete epochs, and MNE's minimum beat requirement before
calling the library.

- [ ] **Step 4: Run tests and compare both methods on the synthetic artifact.**

```bash
uv run pytest tests/test_bcg.py -q
uv run ruff check src/mri_correction/bcg.py tests/test_bcg.py
```

Expected: both methods satisfy the confinement and ECG-preservation tests; PCA-OBS tests
exercise the installed MNE 1.12.1 API rather than a local reimplementation.

- [ ] **Step 5: Commit the BCG correction layer.**

```bash
git add src/mri_correction/bcg.py tests/test_bcg.py
git commit -m "feat: add bounded AAS and PCA-OBS BCG correction"
```

### Task 5: Add held-out cardiac metrics and validated recording pairing

**Files:**
- Modify: `src/mri_correction/metrics.py`
- Create: `src/mri_correction/bcg_benchmark.py`
- Create: `tests/test_bcg_benchmark.py`

- [ ] **Step 1: Write failing metric and pairing tests.** Test the following:

  - held-out cardiac residual uses an even-beat template and odd-beat scoring, with no
    single-trial noise leakage;
  - circular-shift nulls preserve the event interval structure but destroy cardiac phase
    locking and are deterministic for a fixed seed;
  - pair discovery keys recordings by the study's subject/run identity, rejects duplicate
    keys, and refuses missing or mismatched FASTR/Analyzer geometry;
  - Analyzer and FASTR ECG vectors must match within `1e-3` microvolts before a pair is
    scored;
  - the benchmark does not call `audit_marker_trains` until after `detect_r_peaks` returns.

- [ ] **Step 2: Run the focused tests and verify the expected failure.**

```bash
uv run pytest tests/test_bcg_benchmark.py -q
```

Expected: import failure for the new benchmark module or missing metric symbols.

- [ ] **Step 3: Add the metrics without changing existing metric behavior.** Add focused functions to `src/mri_correction/metrics.py`:

```python
def held_out_cardiac_rms(
    data_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> np.ndarray:
    raise NotImplementedError


def cardiac_residual_ratio(
    before_uv: npt.ArrayLike,
    after_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> np.ndarray:
    raise NotImplementedError


def circular_shifted_cardiac_null(
    data_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
    surrogate_count: int,
    seed: int,
) -> np.ndarray:
    raise NotImplementedError
```

Reuse current validation and fractional epoch extraction helpers. Return per-channel arrays
and raise `MetricInputError` for insufficient complete epochs instead of returning a
misleading zero. Keep the existing public metric outputs unchanged.

- [ ] **Step 4: Implement the benchmark orchestration and strict pairing.** Expose:

```python
@dataclass(frozen=True, slots=True)
class RecordingPair:
    recording_id: str
    fastr_vhdr: Path
    analyzer_vhdr: Path


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    report_json: Path
    report_csv: Path
    run_count: int
    successful_count: int
    failed_count: int


def discover_recording_pairs(
    fastr_root: str | Path,
    analyzer_reference_root: str | Path,
) -> tuple[RecordingPair, ...]:
    raise NotImplementedError


def run_bcg_benchmark(config: BenchmarkConfig) -> BenchmarkSummary:
    raise NotImplementedError
```

Use a single explicit run-key parser for `runN_subXXXX` and baseline recordings. Reject
duplicate keys rather than taking the first path. For each pair, load the FASTR and
Analyzer recordings with MNE, validate channel names/order, sample count, sampling rate,
and ECG identity, then run `detect_r_peaks` on the FASTR ECG vector alone. Only after that
call `audit_marker_trains` against Analyzer's `Pulse Artifact,R` annotations.

Score the Analyzer reference by comparing the same FASTR input against the Analyzer output,
and score each own-method output by comparing the same FASTR input against its corrected
copy. Write each corrected method result with `write_brainvision_recording`, using the
independent marker set and a method-specific output directory. Exclude ECG from correction
and metrics. Store per-run detector QC, marker audit, method settings, held-out cardiac
residuals, null maxima, preservation metrics, input paths, hashes, and failure messages. A
failed run must be explicit in the report; it must not be silently omitted from the
denominator.

- [ ] **Step 5: Run benchmark unit tests and lint.**

```bash
uv run pytest tests/test_metrics.py tests/test_bcg_benchmark.py -q
uv run ruff check src/mri_correction/metrics.py src/mri_correction/bcg_benchmark.py tests/test_bcg_benchmark.py
```

Expected: all prior metric tests remain green, new held-out/null tests pass, and pairing
tests reject the intentionally malformed fixtures.

- [ ] **Step 6: Commit metrics and benchmark pairing.**

```bash
git add src/mri_correction/metrics.py src/mri_correction/bcg_benchmark.py tests/test_bcg_benchmark.py
git commit -m "feat: add held-out BCG benchmark metrics"
```

### Task 6: Add command-line entry points and provenance outputs

**Files:**
- Modify: `src/mri_correction/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests.** Add tests for:

  - `mri-correct detect-cardiac --config examples/cardiac_detection.yml` returning JSON
    with output marker recording and detector QC;
  - `mri-correct benchmark-bcg --config examples/bcg_benchmark.yml` returning JSON with
    report paths and run counts;
  - missing ECG and existing output paths returning exit code 1 with a concise error;
  - the existing `run` and `validate-timing` commands retaining their current behavior.

- [ ] **Step 2: Run the CLI tests and verify the expected failure.**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: parser failure because the new subcommands do not exist.

- [ ] **Step 3: Add separate subcommands with config-only algorithm settings.** Register:

```python
detect_cardiac = commands.add_parser(
    "detect-cardiac",
    help="derive independent ECG R markers and write a BrainVision recording",
)
detect_cardiac.add_argument("--config", type=Path, required=True)

benchmark_bcg = commands.add_parser(
    "benchmark-bcg",
    help="benchmark independent-marker BCG correction against Analyzer",
)
benchmark_bcg.add_argument("--config", type=Path, required=True)
```

Dispatch `detect-cardiac` to `load_detection_config` and
`run_cardiac_detection`; dispatch `benchmark-bcg` to `load_benchmark_config` and
`run_bcg_benchmark`. Catch only the new documented input/config exceptions alongside the
existing CLI exceptions. Print `dataclasses.asdict` JSON with absolute paths and no
progress text on stdout.

- [ ] **Step 4: Run the CLI and full test suite.**

```bash
uv run pytest -q
uv run ruff check src tests
git diff --check
```

Expected: the complete existing suite and all new tests pass with no Ruff or whitespace
findings.

- [ ] **Step 5: Commit the command-line boundary.**

```bash
git add src/mri_correction/cli.py tests/test_cli.py
git commit -m "feat: expose cardiac detection and BCG benchmark commands"
```

### Task 7: Document methods, run a real diagnostic, and produce the cohort report

**Files:**
- Create: `docs/bcg_methods.md`
- Modify: `README.md`
- Modify: `docs/validation.md`
- Modify: `examples/cardiac_detection.yml`
- Modify: `examples/bcg_benchmark.yml`

- [ ] **Step 1: Document the method and literature choices.** `docs/bcg_methods.md` must
state the independent input boundary, the detector stages, why MNE `find_ecg_events` is a
baseline rather than a silent fallback, why multi-lead ICA and EEG-derived cycle detection
are not the production path, the AAS/PCA-OBS distinction, the Analyzer-reference limitation,
and the held-out/null validation design. Include the primary-paper DOI/PMID links listed
at the beginning of this plan.

- [ ] **Step 2: Update user documentation and example configurations.** Explain the two
data roots explicitly: FASTR-only input for our correction and `step3_bcg_corrected` as
Analyzer reference. Document that `step3_bcg_corrected` must not be passed as the own-method
EEG input. Include exact commands:

```bash
mri-correct detect-cardiac --config /path/to/cardiac_detection.yml
mri-correct benchmark-bcg --config /path/to/bcg_benchmark.yml
```

- [ ] **Step 3: Run one real-recording diagnostic.** Resolve and record the exact FASTR-only
input root before execution. Run the detector on one representative run and inspect its
marker count, RR distribution, lock statistics, and a plotted ECG window containing a
QRS/T-wave pair. Compare Analyzer markers only in the audit report. If the only available
candidate is `/Volumes/KINGSTON/EEG_fMRI_data/source_data/step3_bcg_corrected`, stop with an
input-stage error rather than using it as the own-method EEG source.

- [ ] **Step 4: Create the detector truth audit before making sensitivity claims.** Select a
stratified, blinded subset of FASTR ECG traces covering clear, difficult, and low-marker
runs. Record manually adjudicated QRS sample positions and the adjudication protocol.
Use this subset for sensitivity, positive predictive value, and timing error; do not use
Analyzer annotations as truth. If manual adjudication is not available, omit those
metrics and state the limitation explicitly.

- [ ] **Step 5: Run the full paired benchmark.** Execute the locked configuration across all
validated pairs. Confirm that every report row contains either complete metrics or an
explicit failure reason. Inspect representative clean, difficult, and low-marker runs.

- [ ] **Step 6: Apply the predeclared interpretation rule.** Report paired run-level
differences for Analyzer, AAS, and PCA-OBS. A claim of outperformance requires lower
primary held-out cardiac residual together with acceptable preservation and null controls;
marker count or spectral notch depth alone is insufficient. Do not retune thresholds on
the headline evaluation after viewing the results.

- [ ] **Step 7: Run final verification and commit documentation/results tooling.**

```bash
uv run pytest -q
uv run ruff check src tests
git diff --check
git status --short
git add README.md docs/validation.md docs/bcg_methods.md examples/cardiac_detection.yml examples/bcg_benchmark.yml
git commit -m "docs: describe independent BCG benchmark methods"
```

Expected: tests, lint, and whitespace checks pass; only the intended documentation and
example changes are staged.

## Plan self-review

Spec coverage is mapped as follows: independent ECG-only input and no Analyzer leakage in
Tasks 1–2; marker output and post-hoc audit in Task 3; AAS/PCA-OBS correction and ECG
preservation in Task 4; held-out/null metrics, validated pairing, and explicit failures in
Task 5; reproducible commands/provenance in Task 6; literature and cohort interpretation in
Task 7. The plan contains no fallback detector, fallback input root, or implicit output
overwrite. Public function signatures use consistent names (`peak_samples`,
`sampling_rate_hz`, `window_seconds`) across modules.
