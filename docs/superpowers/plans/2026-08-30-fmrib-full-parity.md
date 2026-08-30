# FMRIB FASTR Full-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose all scientifically relevant `fmrib_fastr.m` capabilities through the existing strict BIDS/BrainVision Python pipeline while preserving current default outputs.

**Architecture:** Extend the current timing, geometry, OBS, and pipeline modules rather than introducing a second correction engine. Add ANC as one focused module, keep every legacy-risky behavior opt-in, and validate shared signal-processing contracts against MATLAB-generated fixtures.

**Tech Stack:** Python 3.12, NumPy, SciPy, MNE-Python, PyYAML, Pytest, Ruff, MATLAB R2026a with EEGLAB/FMRIB 2.1 for external oracle generation.

---

## File structure

- `src/mri_correction/fastr_timing.py`: strict volume-marker validation and explicit repair.
- `src/mri_correction/fastr_geometry.py`: configurable artifact timing geometry.
- `src/mri_correction/fastr_types.py`: immutable OBS and ANC result values.
- `src/mri_correction/fastr_processing.py`: fixed/automatic OBS fitting and selected-rank reporting.
- `src/mri_correction/fastr_anc.py`: FMRIB LMS adaptive noise cancellation only.
- `src/mri_correction/config.py`: YAML contract for parity options.
- `src/mri_correction/pipeline.py`: stage ordering and batch orchestration.
- `src/mri_correction/pipeline_io.py`: optional output low-pass and strict rate validation.
- `src/mri_correction/pipeline_provenance.py`: parity settings and stage diagnostics.
- `validation/fmrib_reference.m`: reproducible, non-runtime MATLAB oracle runner.
- `validation/compare_fmrib_reference.py`: Python comparison report for oracle outputs.
- `tests/test_marker_repair.py`: volume-marker repair behavior.
- `tests/test_fastr_geometry.py`: configurable trigger fraction.
- `tests/test_obs_rank.py`: automatic OBS rank and detailed OBS results.
- `tests/test_fastr_anc.py`: sample-level LMS and ANC validation.
- `tests/test_pipeline_parity.py`: configuration, stage order, provenance, and default preservation.
- `tests/test_pipeline_modules.py`: disabled-low-pass behavior.

### Task 1: Explicit missing-volume-marker repair

**Files:**
- Modify: `src/mri_correction/fastr_timing.py`
- Modify: `src/mri_correction/fastr.py`
- Create: `tests/test_marker_repair.py`

- [ ] **Step 1: Write the failing marker-repair tests**

```python
import numpy as np
import pytest

from mri_correction.fastr import FastrInputError, repair_volume_starts


def test_repair_volume_starts_fills_unique_interior_gaps() -> None:
    starts = np.array([100, 200, 400, 500], dtype=np.int64)

    repaired = repair_volume_starts(
        starts,
        samples_per_volume=100,
        expected_volume_count=5,
    )

    np.testing.assert_array_equal(repaired, [100, 200, 300, 400, 500])


def test_repair_volume_starts_accepts_one_sample_clock_ticks() -> None:
    starts = np.array([100, 201, 401, 501], dtype=np.int64)

    repaired = repair_volume_starts(
        starts,
        samples_per_volume=100,
        expected_volume_count=5,
    )

    np.testing.assert_array_equal(repaired, [100, 201, 301, 401, 501])


@pytest.mark.parametrize(
    ("starts", "count", "message"),
    [
        ([100, 250, 350], 4, "not an integer multiple"),
        ([100, 200, 400], 5, "expected volume count"),
        ([100, 200, 300], 4, "boundary"),
    ],
)
def test_repair_volume_starts_rejects_ambiguous_repairs(
    starts: list[int], count: int, message: str
) -> None:
    with pytest.raises(FastrInputError, match=message):
        repair_volume_starts(
            np.asarray(starts),
            samples_per_volume=100,
            expected_volume_count=count,
        )
```

- [ ] **Step 2: Verify the tests fail because the API is absent**

Run: `uv run pytest tests/test_marker_repair.py -v`

Expected: collection fails because `repair_volume_starts` cannot be imported.

- [ ] **Step 3: Implement strict interior repair and export it**

Add a public `repair_volume_starts` in `fastr_timing.py`. Validate starts with
`_validate_volume_starts`, require positive integer inputs, round each observed
interval to the nearest repetition multiple, accept at most one sample of clock
error, and insert positions from the left observed marker. Reject any repaired
count different from `expected_volume_count`; if no interior gap exists, report
that missing boundary markers cannot be inferred.

```python
def repair_volume_starts(
    volume_starts: object,
    *,
    samples_per_volume: int,
    expected_volume_count: int,
) -> np.ndarray:
    starts = _validate_volume_starts(volume_starts)
    period = _validate_positive_integer(samples_per_volume, "samples per volume")
    expected = _validate_positive_integer(
        expected_volume_count,
        "expected volume count",
    )
    repaired = [int(starts[0])]
    for left, right in pairwise(starts):
        interval = int(right - left)
        multiple = round(interval / period)
        if multiple < 1 or abs(interval - multiple * period) > _CLOCK_TICK_SAMPLES:
            raise FastrInputError(
                "volume marker interval is not an integer multiple of the "
                "repetition time"
            )
        repaired.extend(int(left + step * period) for step in range(1, multiple))
        repaired.append(int(right))
    if len(repaired) != expected:
        detail = "boundary markers cannot be inferred" if len(repaired) < expected else "too many markers"
        raise FastrInputError(f"repaired marker count does not match expected volume count: {detail}")
    return np.asarray(repaired, dtype=np.int64)


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise FastrInputError(f"{name} must be a positive integer")
    return int(value)
```

Import `pairwise` from `itertools` and `Integral` from `numbers`. Add
`repair_volume_starts` to `fastr.py` imports and `__all__`.

- [ ] **Step 4: Run focused timing tests**

Run: `uv run pytest tests/test_marker_repair.py tests/test_fastr.py -v`

Expected: all focused tests pass.

- [ ] **Step 5: Commit marker repair**

```bash
git add src/mri_correction/fastr.py src/mri_correction/fastr_timing.py tests/test_marker_repair.py
git commit -m "feat: add explicit volume marker repair"
```

### Task 2: Configurable trigger position within the artifact

**Files:**
- Modify: `src/mri_correction/fastr_geometry.py`
- Modify: `src/mri_correction/fastr_types.py`
- Modify: `src/mri_correction/fastr.py`
- Modify: `src/mri_correction/fastr_processing.py`
- Create: `tests/test_fastr_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

```python
import numpy as np
import pytest

from mri_correction.fastr import FastrInputError, prepare_fastr_geometry


def make_geometry(fraction: float):
    return prepare_fastr_geometry(
        np.arange(40, dtype=np.float64) * 100.0 + 200.0,
        sample_count=4_400,
        interpolation_factor=10,
        neighbor_count=10,
        search_radius_samples=3,
        pre_trigger_fraction=fraction,
    )


@pytest.mark.parametrize(("fraction", "before"), [(0.0, 10), (0.03, 40), (1.0, 1_010)])
def test_geometry_uses_configured_trigger_fraction(
    fraction: float, before: int
) -> None:
    geometry = make_geometry(fraction)
    assert geometry.pre_trigger_fraction == fraction
    assert geometry.epoch.samples_before == before


@pytest.mark.parametrize("fraction", [-0.01, 1.01, np.inf, np.nan, True])
def test_geometry_rejects_invalid_trigger_fraction(fraction: object) -> None:
    with pytest.raises(FastrInputError, match="pre-trigger fraction"):
        make_geometry(fraction)
```

- [ ] **Step 2: Verify the new keyword fails**

Run: `uv run pytest tests/test_fastr_geometry.py -v`

Expected: failures report an unexpected `pre_trigger_fraction` keyword.

- [ ] **Step 3: Thread the validated fraction through geometry**

Add `pre_trigger_fraction: float` to `FastrGeometry`. Add a keyword with default
`0.03` to `prepare_fastr_geometry`, `_build_fastr_geometry`, the four public
FASTR convenience functions, and the two private runners. Change
`_measure_artifact_epoch` to accept the fraction.

```python
def _validate_pre_trigger_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FastrInputError("pre-trigger fraction must be a finite number")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise FastrInputError("pre-trigger fraction must lie between zero and one")
    return fraction


def _measure_artifact_epoch(
    fine_triggers: np.ndarray,
    *,
    cover_full_gap: bool,
    pre_trigger_fraction: float,
) -> _ArtifactEpoch:
    intervals = np.diff(fine_triggers)
    interval = math.ceil(np.median(intervals))
    before = _round_half_up(interval * pre_trigger_fraction)
    if cover_full_gap:
        return _ArtifactEpoch(before, int(np.max(intervals)), 0)
    slack = math.ceil((1.0 + _ARTIFACT_SLACK_FRACTION) * interval) - interval
    after = _round_half_up((1.0 - pre_trigger_fraction) * interval)
    return _ArtifactEpoch(before + slack, after, slack)
```

These expectations include the ten interpolated samples of artifact slack used
by the classical slice-trigger geometry.

- [ ] **Step 4: Run geometry and FASTR regression tests**

Run: `uv run pytest tests/test_fastr_geometry.py tests/test_slice_fastr.py tests/test_fastr_batch.py -v`

Expected: all tests pass, including unchanged default geometry.

- [ ] **Step 5: Commit artifact timing**

```bash
git add src/mri_correction/fastr.py src/mri_correction/fastr_geometry.py src/mri_correction/fastr_processing.py src/mri_correction/fastr_types.py tests/test_fastr_geometry.py
git commit -m "feat: configure FASTR trigger position"
```

### Task 3: Automatic and section-wise OBS

**Files:**
- Modify: `src/mri_correction/fastr_types.py`
- Modify: `src/mri_correction/fastr_processing.py`
- Modify: `src/mri_correction/fastr.py`
- Create: `tests/test_obs_rank.py`

- [ ] **Step 1: Write failing automatic-rank and result tests**

```python
import numpy as np
import pytest

from mri_correction.fastr import (
    FastrInputError,
    fit_residual_obs,
    select_obs_rank,
)


def test_select_obs_rank_matches_fmrib_three_criterion_rule() -> None:
    explained = np.array([45.0, 25.0, 12.0, 7.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    assert select_obs_rank(explained) == 3


def test_select_obs_rank_rejects_a_spectrum_without_a_stable_knee() -> None:
    with pytest.raises(FastrInputError, match="automatic OBS rank"):
        select_obs_rank(np.array([60.0, 40.0]))


def test_fit_residual_obs_reports_selected_rank_by_channel_and_section() -> None:
    rate = 500.0
    triggers = np.arange(20, 100) * 50
    samples = np.arange(6_000)
    residual = np.vstack(
        [
            np.sin(2 * np.pi * samples / 50),
            np.sin(2 * np.pi * samples / 50) + 0.3 * np.cos(4 * np.pi * samples / 50),
        ]
    )

    result = fit_residual_obs(
        residual,
        triggers,
        sampling_rate=rate,
        excluded_channels=(),
        rank=2,
        interpolation_factor=1,
        section_seconds=4.0,
    )

    assert result.data.shape == residual.shape
    assert result.selected_ranks.shape == (2, 2)
    np.testing.assert_array_equal(result.selected_ranks, 2)
```

- [ ] **Step 2: Verify the APIs are absent**

Run: `uv run pytest tests/test_obs_rank.py -v`

Expected: collection fails for missing `fit_residual_obs` and `select_obs_rank`.

- [ ] **Step 3: Add immutable detailed OBS results and rank selection**

```python
@dataclass(frozen=True, slots=True, eq=False)
class ResidualObsCorrection:
    data: np.ndarray
    selected_ranks: np.ndarray

    def __post_init__(self) -> None:
        for name in ("data", "selected_ranks"):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
```

```python
def select_obs_rank(explained_variance_percent: npt.ArrayLike) -> int:
    values = np.asarray(explained_variance_percent, dtype=np.float64)
    if values.ndim != 1 or values.size < 5 or not np.all(np.isfinite(values)):
        raise FastrInputError("automatic OBS rank needs at least five finite components")
    stable = np.flatnonzero(np.abs(np.diff(values)) < 2.0)
    consecutive = np.flatnonzero(
        (np.diff(stable)[:-2] == 1)
        & (np.diff(stable)[1:-1] == 1)
        & (np.diff(stable)[2:] == 1)
    )
    cumulative = np.flatnonzero(np.cumsum(values) > 80.0)
    below_five = np.flatnonzero(values < 5.0)
    if not consecutive.size or not cumulative.size or not below_five.size:
        raise FastrInputError("automatic OBS rank criteria did not identify a stable rank")
    slope_rank = max(int(stable[consecutive[0]]), 1)
    cumulative_rank = int(cumulative[0]) + 1
    variance_rank = max(int(below_five[0]), 1)
    return max(int(np.floor(np.mean([slope_rank, cumulative_rank, variance_rank]))), 1)
```

Add `fit_residual_obs` as the detailed implementation. Let `rank` accept
`int | Literal["auto"]`; for every channel and section, compute singular values,
convert them to explained percentages, choose/validate the rank, fit the basis,
and record it. Keep `residual_obs` as the stable ndarray-returning wrapper:

```python
def residual_obs(*args: object, **kwargs: object) -> np.ndarray:
    return np.array(fit_residual_obs(*args, **kwargs).data, copy=True)
```

Export the new value and functions from `fastr.py`.

- [ ] **Step 4: Run OBS tests and correct only evidence-backed discrepancies**

Run: `uv run pytest tests/test_obs_rank.py tests/test_residual_obs_stage.py -v`

Expected: automatic-rank tests and all existing fixed-rank tests pass.

- [ ] **Step 5: Commit automatic OBS**

```bash
git add src/mri_correction/fastr.py src/mri_correction/fastr_processing.py src/mri_correction/fastr_types.py tests/test_obs_rank.py
git commit -m "feat: add automatic sectioned OBS"
```

### Task 4: Faithful FMRIB ANC primitive

**Files:**
- Create: `src/mri_correction/fastr_anc.py`
- Modify: `src/mri_correction/fastr_types.py`
- Modify: `src/mri_correction/fastr.py`
- Create: `tests/test_fastr_anc.py`

- [ ] **Step 1: Generate and freeze a deterministic MATLAB LMS fixture**

Run MATLAB with fixed arrays and the installed `fastranc.m`; save `refs`, `d`,
`N`, `mu`, `out`, and `y` as decimal arrays in `tests/test_fastr_anc.py`. Use:

```matlab
addpath('/Users/joduq24/Library/Application Support/MathWorks/MATLAB Add-Ons/Collections/EEGLAB/plugins/fMRIb2.1');
refs = sin((0:39)' * 0.37) + 0.2 * cos((0:39)' * 0.11);
d = 0.4 * refs + cos((0:39)' * 0.23);
[out, y] = fastranc(refs, d, 4, 0.01);
format longG;
disp(refs'); disp(d'); disp(out'); disp(y');
```

- [ ] **Step 2: Write failing primitive and validation tests**

```python
import numpy as np
import pytest

from mri_correction.fastr import FastrInputError, adaptive_noise_cancel, fmrib_lms


def test_fmrib_lms_matches_matlab_fixture() -> None:
    refs = MATLAB_REFS
    desired = MATLAB_DESIRED
    error, noise = fmrib_lms(refs, desired, filter_order=4, step_size=0.01)
    np.testing.assert_allclose(error, MATLAB_ERROR, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(noise, MATLAB_NOISE, rtol=1e-12, atol=1e-12)


def test_adaptive_noise_cancel_rejects_zero_variance_reference() -> None:
    with pytest.raises(FastrInputError, match="reference variance"):
        adaptive_noise_cancel(
            np.ones((1, 1_000)),
            np.zeros((1, 1_000)),
            sampling_rate=500.0,
            filter_order=50,
            excluded_channels=(),
        )


def test_adaptive_noise_cancel_leaves_excluded_channels_untouched() -> None:
    samples = np.arange(2_000)
    artifact = np.sin(2 * np.pi * samples / 50)[np.newaxis, :]
    corrected = np.cos(2 * np.pi * samples / 73)[np.newaxis, :]
    result = adaptive_noise_cancel(
        corrected,
        artifact,
        sampling_rate=500.0,
        filter_order=50,
        excluded_channels=(0,),
    )
    np.testing.assert_array_equal(result.data, corrected)
    assert np.isnan(result.reference_scales[0])


def test_adaptive_noise_cancel_bypasses_flat_channels() -> None:
    corrected = np.vstack([np.zeros(2_000), np.sin(np.arange(2_000) / 11.0)])
    artifact = np.vstack([np.zeros(2_000), np.sin(np.arange(2_000) / 7.0)])
    result = adaptive_noise_cancel(
        corrected,
        artifact,
        sampling_rate=500.0,
        filter_order=50,
        excluded_channels=(),
    )
    np.testing.assert_array_equal(result.data[0], corrected[0])
    assert np.isnan(result.reference_scales[0])
```

- [ ] **Step 3: Verify the ANC tests fail on missing APIs**

Run: `uv run pytest tests/test_fastr_anc.py -v`

Expected: collection fails because the ANC functions are absent.

- [ ] **Step 4: Implement the focused ANC module**

Define `AncCorrection(data, reference_scales, step_sizes, filter_order)` as an
immutable dataclass. Implement `fmrib_lms` with the exact update order from
`fastranc.c` and `adaptive_noise_cancel` with the FMRIB 2 Hz high-pass,
least-squares reference scaling, and `0.05 / (order * variance)` step size.

```python
def fmrib_lms(
    reference: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    filter_order: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    refs = _validate_vector(reference, "reference")
    target = _validate_vector(desired, "desired")
    if refs.shape != target.shape:
        raise FastrInputError("reference and desired signals must have equal length")
    order = _validate_filter_order(filter_order, refs.size)
    step = validate_positive_finite(step_size, name="ANC step size")
    weights = np.zeros(order + 1)
    error = np.zeros(refs.size)
    noise = np.zeros(refs.size)
    for index in range(order, refs.size):
        window = refs[index - order : index + 1]
        noise[index] = weights @ window
        error[index] = target[index] - noise[index]
        weights += 2.0 * step * error[index] * window
    return error, noise
```

The high-pass must use `scipy.signal.firls` and `filtfilt` with the MATLAB order
formula. Validate every array, channel index, variance, scale, step, and result.
Export the ANC APIs from `fastr.py`.

- [ ] **Step 5: Run ANC tests**

Run: `uv run pytest tests/test_fastr_anc.py -v`

Expected: MATLAB fixture agreement and all validation tests pass.

- [ ] **Step 6: Commit ANC primitive**

```bash
git add src/mri_correction/fastr.py src/mri_correction/fastr_anc.py src/mri_correction/fastr_types.py tests/test_fastr_anc.py
git commit -m "feat: implement FMRIB adaptive noise cancellation"
```

### Task 5: Strict parity configuration

**Files:**
- Modify: `src/mri_correction/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add failing configuration tests**

Add tests proving these exact contracts:

```python
def test_parity_options_keep_conservative_defaults(tmp_path: Path) -> None:
    config = write_and_load(tmp_path, valid_document())
    assert config.timing.missing_volume_markers == "error"
    assert config.timing.expected_volume_count is None
    assert config.processing.pre_trigger_fraction == 0.03
    assert config.processing.residual_obs_rank == 4
    assert config.processing.residual_obs_section_seconds is None
    assert config.processing.adaptive_noise_cancellation is False


def test_config_accepts_all_fmrib_parity_options(tmp_path: Path) -> None:
    document = valid_document().replace(
        "  marker_description: volume-start\n",
        "  marker_description: volume-start\n"
        "  missing_volume_markers: repair\n"
        "  expected_volume_count: 120\n",
    ) + (
        "  pre_trigger_fraction: 0.05\n"
        "  residual_obs_rank: auto\n"
        "  residual_obs_section_seconds: 60.0\n"
        "  adaptive_noise_cancellation: true\n"
    )
    config = write_and_load(tmp_path, document)
    assert config.timing.expected_volume_count == 120
    assert config.processing.residual_obs_rank == "auto"
    assert config.processing.adaptive_noise_cancellation


def test_marker_repair_requires_expected_count(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="expected_volume_count"):
        write_and_load(
            tmp_path,
            valid_document().replace(
                "  marker_description: volume-start\n",
                "  marker_description: volume-start\n"
                "  missing_volume_markers: repair\n",
            ),
        )
```

Also parameterize invalid marker modes, invalid counts, invalid fractions,
invalid OBS rank strings, nonpositive section lengths, and non-boolean ANC.

- [ ] **Step 2: Verify configuration tests fail**

Run: `uv run pytest tests/test_config.py -q`

Expected: new fields are rejected as unknown or absent.

- [ ] **Step 3: Extend immutable config values and strict YAML parsing**

```python
@dataclass(frozen=True, slots=True)
class TimingConfig:
    marker_type: str
    marker_description: str
    missing_volume_markers: str = "error"
    expected_volume_count: int | None = None
```

Add processing fields:

```python
pre_trigger_fraction: float = 0.03
residual_obs_rank: int | str = 4
residual_obs_section_seconds: float | None = None
adaptive_noise_cancellation: bool = False
```

Update the allowed/optional key sets. Parse `residual_obs_rank` as either a
positive integer or exactly `auto`. Require expected count only in repair mode,
and reject an expected count in error mode so configuration intent is unique.

- [ ] **Step 4: Run all config tests**

Run: `uv run pytest tests/test_config.py -q`

Expected: all configuration tests pass.

- [ ] **Step 5: Commit parity configuration**

```bash
git add src/mri_correction/config.py tests/test_config.py
git commit -m "feat: configure FMRIB parity stages"
```

### Task 6: Integrate marker repair, OBS detail, and ANC into the pipeline

**Files:**
- Modify: `src/mri_correction/pipeline.py`
- Modify: `src/mri_correction/pipeline_provenance.py`
- Create: `tests/test_pipeline_parity.py`

- [ ] **Step 1: Write failing orchestration tests with real stage seams**

Use the existing tiny BrainVision builders from `tests/test_pipeline.py`. Add:

```python
def test_pipeline_repairs_markers_only_when_explicitly_enabled(tmp_path: Path) -> None:
    config = make_pipeline_config(
        tmp_path,
        missing_volume_markers="repair",
        expected_volume_count=40,
    )
    remove_one_interior_volume_marker(config.input.raw_vhdr, index=20)
    summary = run_correction(config)
    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["markers"]["repaired_volume_count"] == 1


def test_pipeline_runs_obs_before_anc(monkeypatch, pipeline_case) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "fit_residual_obs",
        recording_stage("obs", calls),
    )
    monkeypatch.setattr(
        pipeline_module,
        "adaptive_noise_cancel",
        recording_stage("anc", calls),
    )
    run_correction(pipeline_case.config)
    assert calls == ["obs", "anc"] * pipeline_case.batch_count


def test_default_pipeline_does_not_run_new_optional_stages(monkeypatch, pipeline_case) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "adaptive_noise_cancel",
        lambda *args, **kwargs: pytest.fail("ANC ran under defaults"),
    )
    run_correction(pipeline_case.config)
```

Add a provenance assertion for repaired counts, pre-trigger fraction, OBS rank
mode/selected ranks/section duration, ANC enabled/order/scales/steps, and the
FMRIB reference commit.

- [ ] **Step 2: Verify integration tests fail at the first missing behavior**

Run: `uv run pytest tests/test_pipeline_parity.py -v`

Expected: failures identify absent repair, detailed OBS, ANC, or provenance.

- [ ] **Step 3: Integrate stages in explicit order**

After selecting volume markers, compute integer samples per volume and invoke
`repair_volume_starts` only in repair mode. Pass `pre_trigger_fraction` into
geometry. In each batch:

```python
template_result = apply_fastr_batch(
    batch,
    geometry,
    alignment,
    template_high_pass_hz=config.processing.template_high_pass_hz,
    sampling_rate=input_rate,
    unscaled_channels=batch_non_eeg_rows,
)
corrected_batch = template_result.data
selected_obs_ranks = np.empty((stop - start, 0), dtype=np.int64)
if config.processing.residual_obs:
    obs_result = fit_residual_obs(
        corrected_batch,
        obs_triggers,
        sampling_rate=input_rate,
        excluded_channels=batch_non_eeg_rows,
        rank=config.processing.residual_obs_rank,
        interpolation_factor=config.processing.interpolation_factor,
        section_seconds=config.processing.residual_obs_section_seconds,
    )
    corrected_batch = obs_result.data
    selected_obs_ranks = obs_result.selected_ranks
if config.processing.adaptive_noise_cancellation:
    artifact_estimate = batch - corrected_batch
    anc_result = adaptive_noise_cancel(
        corrected_batch,
        artifact_estimate,
        sampling_rate=input_rate,
        filter_order=math.ceil(geometry.epoch.length / geometry.interpolation_factor),
        excluded_channels=batch_non_eeg_rows,
        sample_slice=corrected_input_span,
    )
    corrected_batch = anc_result.data
```

Aggregate batch diagnostics into channel-ordered arrays. Keep all current
default branches byte-for-byte unchanged.

- [ ] **Step 4: Extend provenance without private validation paths**

Add constant:

```python
FMRIB_REFERENCE_COMMIT = "2aa522bc5ec4215f42b3ba8efdb2b84d2a312935"
```

Report repair mode/count, geometry fraction, OBS rank mode and selected ranks,
OBS section seconds, ANC enabled/filter order/scales/steps. Extend
`make_provenance` arguments with explicit typed values rather than a generic
diagnostics dictionary.

- [ ] **Step 5: Run focused pipeline tests**

Run: `uv run pytest tests/test_pipeline_parity.py tests/test_pipeline.py tests/test_pipeline_modules.py -v`

Expected: all focused tests pass and defaults do not call repair or ANC.

- [ ] **Step 6: Commit pipeline integration**

```bash
git add src/mri_correction/pipeline.py src/mri_correction/pipeline_provenance.py tests/test_pipeline_parity.py
git commit -m "feat: integrate complete FMRIB stage options"
```

### Task 7: Support disabled output low-pass safely

**Files:**
- Modify: `src/mri_correction/config.py`
- Modify: `src/mri_correction/pipeline_io.py`
- Modify: `src/mri_correction/pipeline.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_pipeline_modules.py`

- [ ] **Step 1: Write failing filter-disable tests**

```python
def test_no_lowpass_returns_the_window_unchanged_at_input_rate() -> None:
    data = np.arange(20, dtype=np.float64)[np.newaxis, :]
    result = pipeline_io.lowpass_and_decimate(
        data,
        sampling_rate=500.0,
        output_sampling_rate=500.0,
        lowpass_hz=0.0,
        window=OutputWindow(start=3, stop=17),
    )
    np.testing.assert_array_equal(result, data[:, 3:17])


def test_no_lowpass_rejects_decimation() -> None:
    with pytest.raises(PipelineInputError, match="anti-alias"):
        pipeline_io.validate_rates(5_000.0, 1_000.0, 0.0)
```

Change the config test that currently rejects `lowpass_hz: 0.0` to accept it.

- [ ] **Step 2: Verify the filter tests fail**

Run: `uv run pytest tests/test_pipeline_modules.py tests/test_config.py -q`

Expected: zero cutoff is rejected or passed to MNE filter design.

- [ ] **Step 3: Implement the explicit no-filter branch**

Validate `lowpass_hz` as finite and nonnegative. In `validate_rates`, reject
zero when decimation is greater than one. In `lowpass_and_decimate`:

```python
if lowpass_hz == 0.0:
    return np.array(data[:, window.start : window.stop], copy=True)
```

In the pipeline, calculate diagnostic `fmax` from output Nyquist when low-pass
is zero; otherwise retain the existing minimum with the cutoff.

- [ ] **Step 4: Run filter and pipeline tests**

Run: `uv run pytest tests/test_pipeline_modules.py tests/test_pipeline.py tests/test_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit disabled-filter support**

```bash
git add src/mri_correction/config.py src/mri_correction/pipeline.py src/mri_correction/pipeline_io.py tests/test_config.py tests/test_pipeline_modules.py
git commit -m "feat: allow explicit unfiltered FASTR output"
```

### Task 8: MATLAB oracle and real-recording comparison

**Files:**
- Create: `validation/fmrib_reference.m`
- Create: `validation/compare_fmrib_reference.py`
- Create: `tests/test_matlab_comparison.py`
- Create: `docs/fmrib-parity-validation.md`

- [ ] **Step 1: Write failing tests for comparison metrics and schema**

```python
def test_comparison_reports_residual_transfer_and_ecg_metrics() -> None:
    report = compare_arrays(
        raw=RAW_FIXTURE,
        matlab=MATLAB_FIXTURE,
        python=PYTHON_FIXTURE,
        sampling_rate=500.0,
        repetition_time_seconds=0.9,
        ecg_index=1,
    )
    assert set(report) == {
        "sample_rmse",
        "scanner_harmonic_rms",
        "broadband_transfer",
        "ecg_correlation",
    }
```

- [ ] **Step 2: Verify comparison helper is absent**

Run: `uv run pytest tests/test_matlab_comparison.py -v`

Expected: import fails for `validation.compare_fmrib_reference`.

- [ ] **Step 3: Add reusable MATLAB runner**

The MATLAB function accepts input/output paths and explicit FASTR parameters,
starts EEGLAB without the GUI, loads BrainVision, selects requested channels and
sample span, extracts exact volume events, invokes `fmrib_fastr`, and saves raw,
corrected, triggers, sampling rate, and parameters to MAT v7.3. It must not
contain any subject-specific path or default.

Function signature:

```matlab
function fmrib_reference(input_vhdr, output_mat, channels, first_volume, volume_count, lowpass_hz, interpolation_factor, window, anc_enabled, pre_trigger_fraction, excluded_channels, obs_rank)
```

- [ ] **Step 4: Add Python comparison CLI**

Implement `compare_arrays` using repository metrics, and a CLI taking raw
BrainVision, BIDS JSON, MATLAB MAT, and Python BrainVision paths. Emit one JSON
report. Reject mismatched channel counts, sample rates, or spans.

- [ ] **Step 5: Run MATLAB on a bounded representative recording**

Use sub-0001 run 1, channels `Fp1`, `Cz`, and `ECG`, and a bounded run of at
least 80 volume markers. Generate outputs outside the repository. Run equivalent
Python settings with template high-pass and line regression disabled for the
shared-stage comparison.

- [ ] **Step 6: Record aggregate evidence**

Write `docs/fmrib-parity-validation.md` with the exact source hashes, MATLAB
version, parameters, bounded span, stage-level tolerances, residual suppression,
signal transfer, ECG preservation, and the reason whole-pipeline samples differ
under BIDS acquisition-slot geometry. Do not include subject data or reusable
absolute private paths.

- [ ] **Step 7: Run the comparison tests**

Run: `uv run pytest tests/test_matlab_comparison.py -v`

Expected: all comparison schema and metric tests pass.

- [ ] **Step 8: Commit the validation harness and evidence**

```bash
git add validation docs/fmrib-parity-validation.md tests/test_matlab_comparison.py
git commit -m "test: validate FASTR stages against MATLAB"
```

### Task 9: User documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `docs/algorithm.md`
- Modify: `docs/validation.md`
- Modify: `examples/configuration.yml`

- [ ] **Step 1: Update public documentation and example configuration**

Document the exact MATLAB-to-YAML mapping, conservative defaults, strict marker
repair, trigger fraction, fixed/automatic/disabled OBS, section duration, ANC
risk, and disabled-low-pass constraint. Remove the statement that ANC is not
implemented. Preserve the existing warning that ANC removed injected tones near
scanner harmonics.

- [ ] **Step 2: Run documentation consistency searches**

Run:

```bash
rg -n "not implemented|adaptive noise cancellation|residual_obs_rank|missing_volume_markers|pre_trigger_fraction" README.md docs examples
```

Expected: no stale claim says ANC or automatic OBS is unavailable, and every new
configuration key appears in algorithm and example documentation.

- [ ] **Step 3: Run the full automated verification**

Run:

```bash
uv run pytest
uv run ruff check src tests validation
git diff --check
```

Expected: zero test failures, zero Ruff violations, and no whitespace errors.

- [ ] **Step 4: Review the complete diff against the design checklist**

Run:

```bash
git diff 389b8f9b73a6756a1273574b02b569d896acea4d --stat
git diff 389b8f9b73a6756a1273574b02b569d896acea4d -- README.md docs examples src tests validation
```

Confirm every in-scope reference-audit row is implemented or documented, no
private recording is tracked, and current defaults remain unchanged.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/algorithm.md docs/validation.md examples/configuration.yml
git commit -m "docs: document complete FMRIB FASTR parity"
```

- [ ] **Step 6: Request independent code review**

Request review of commits after
`389b8f9b73a6756a1273574b02b569d896acea4d` against the design and this plan.
Resolve every critical or important finding with a failing regression test and a
focused fix.

- [ ] **Step 7: Re-run fresh final verification after review fixes**

Run:

```bash
uv run pytest
uv run ruff check src tests validation
git diff --check
git status --short --branch
```

Expected: all tests and lint pass, whitespace is clean, and only the feature
branch is checked out with no uncommitted changes.
