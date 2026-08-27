# Gradient Boundaries and Template Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every volume in the emitted window by reading untrimmed input and trimming inside the pipeline, then measure and improve template quality with evidence.

**Architecture:** The pipeline gains an optional `trim` section. With it enabled it reads untrimmed BrainVision data, builds FASTR geometry over the full recording so boundary volumes have complete epochs, and slices the output window during decimation rather than before correction. A second phase adds a microvolt-valued residual-artifact measurement written to the sidecar and annotated in the output `.vmrk`, then uses it to pick `neighbor_count`.

**Tech Stack:** Python 3.12, numpy 2.4.6, scipy 1.17.1, mne 1.12.1, pybv 0.8.1, pytest 9.1.1, ruff 0.16.3, uv.

## Global Constraints

- Python `>=3.12,<3.13`. Dependency versions are pinned exactly in `pyproject.toml`; do not change them.
- Run tests with `uv run pytest` and lint with `uv run ruff check src tests`. Both must pass before every commit.
- ruff line-length 88, rules `E, F, I, B, UP, RUF`.
- BrainVision marker positions are **one-based** everywhere they appear in `BrainVisionMarker.position`. Sample indices inside numpy arrays are **zero-based**. Every conversion must be explicit.
- `trim.mode` defaults to `none`, which must reproduce today's output **byte for byte**. This is the regression gate for the decimation change.
- No correction function may accept an estimated acquisition period. Group timing comes from BIDS metadata only (see `docs/algorithm.md`).
- Phase 1 (Tasks 1-9) is independently shippable and closes with a cohort re-run. Phase 2 (Tasks 10-11) depends on it.

## Verified facts this plan relies on

Established by measurement during the investigation. Do not re-derive; do not assume they generalise beyond this cohort.

- `step0_trimmed_raw_5khz` files are byte-identical to `untrimmed[firstVolumeMarker - 1 : lastVolumeMarker]` (verified at head, mid, tail on sub-0006 baseline).
- FASTR needs 10.5 input samples before the first volume marker and 4503 input samples (0.9006 s) after a volume marker to keep that volume, for TR 0.9 s / 18 groups / slice offsets 0 to 0.8325 s.
- Untrimmed data exists for all 21 real subjects under `source_data/sub-XXXX/eeg/original_untrimmed_5khz/`, head margin 16.4-28.5 s, tail margin 0.04-10.5 s.
- The slice rate is `groups_per_volume / repetition_time` = 20 Hz here, and its 3rd harmonic lands exactly on 60 Hz mains.

## File Structure

| file | responsibility | status |
| --- | --- | --- |
| `src/mri_correction/config.py` | add `TrimConfig`, optional `trim` section | modify |
| `src/mri_correction/window.py` | resolve the output window from markers and mode | create |
| `src/mri_correction/brainvision_io.py:126` | `resample_markers` gains `window`, drops out-of-window markers | modify |
| `src/mri_correction/pipeline.py:329` | `_lowpass_and_decimate` fuses slice and decimate | modify |
| `src/mri_correction/pipeline.py:78` | `_run_correction` wiring | modify |
| `src/mri_correction/fastr.py:1234` | partial-epoch amplitude fit | modify |
| `src/mri_correction/fastr.py:362` | estimate the template on a high-passed copy | modify |
| `src/mri_correction/residual_qc.py` | derived harmonics, per-block residual in uV | create |
| `scripts/sweep_neighbor_count.py` | analysis harness for Task 11 | create |
| `examples/configuration.yml` | use `first_to_last_volume` | modify |
| `docs/algorithm.md` | document trimming and boundary handling | modify |

---

## Phase 1: boundary correctness

### Task 1: `TrimConfig` and an optional `trim` section

**Files:**
- Modify: `src/mri_correction/config.py:65` (`_TOP_LEVEL_KEYS`), `:18-62` (dataclasses), `:101-133` (`load_config`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TrimConfig(mode: str, minimum_epoch_coverage: float)`; `CorrectionConfig.trim: TrimConfig`. `mode` is one of `"none"`, `"first_to_last_volume"`. Defaults: `mode="none"`, `minimum_epoch_coverage=0.75`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_trim_section_defaults_to_none(tmp_path):
    config = load_config(_write_config(tmp_path, trim=None))
    assert config.trim.mode == "none"
    assert config.trim.minimum_epoch_coverage == 0.75


def test_trim_section_is_parsed(tmp_path):
    config = load_config(
        _write_config(
            tmp_path,
            trim={"mode": "first_to_last_volume", "minimum_epoch_coverage": 0.5},
        )
    )
    assert config.trim.mode == "first_to_last_volume"
    assert config.trim.minimum_epoch_coverage == 0.5


def test_trim_rejects_unknown_mode(tmp_path):
    with pytest.raises(ConfigurationError, match="trim.mode"):
        load_config(_write_config(tmp_path, trim={"mode": "everything"}))


def test_trim_rejects_coverage_outside_unit_interval(tmp_path):
    with pytest.raises(ConfigurationError, match="minimum_epoch_coverage"):
        load_config(
            _write_config(tmp_path, trim={"minimum_epoch_coverage": 1.5})
        )
```

`_write_config` is a helper you add in the same file: it writes the four existing sections from `examples/configuration.yml` and, when `trim` is not `None`, adds a `trim:` mapping. Read the existing tests in `tests/test_config.py` first and match their fixture style.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k trim -v`
Expected: FAIL with `AttributeError: 'CorrectionConfig' object has no attribute 'trim'`.

- [ ] **Step 3: Implement**

In `src/mri_correction/config.py`, add near the other dataclasses:

```python
_TRIM_MODES = frozenset({"none", "first_to_last_volume"})
_TRIM_KEYS = frozenset({"mode", "minimum_epoch_coverage"})


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """How the pipeline restricts its output to the scanning period."""

    mode: str = "none"
    minimum_epoch_coverage: float = 0.75

    def __post_init__(self) -> None:
        if self.mode not in _TRIM_MODES:
            raise ConfigurationError(
                f"trim.mode must be one of {sorted(_TRIM_MODES)}"
            )
        if not 0.0 < self.minimum_epoch_coverage <= 1.0:
            raise ConfigurationError(
                "trim.minimum_epoch_coverage must be within (0, 1]"
            )
```

Add `trim: TrimConfig` to `CorrectionConfig`. Change `_TOP_LEVEL_KEYS` to include `"trim"`, and change the required-section loop so `trim` is optional:

```python
_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing", "trim"})
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"input", "output", "timing", "processing"})
```

```python
    for section in _REQUIRED_TOP_LEVEL_KEYS:
        if section not in root:
            raise ConfigurationError(f"missing required field: {section}")
```

Add a loader that tolerates a missing section and missing keys within it:

```python
def _trim_config(root: Mapping[str, object]) -> TrimConfig:
    if "trim" not in root:
        return TrimConfig()
    values = _require_mapping(root["trim"], "trim")
    _reject_unknown_keys(values, _TRIM_KEYS, "trim")
    mode = _string_value(values, "mode") if "mode" in values else "none"
    coverage = (
        _finite_number(values, "minimum_epoch_coverage", minimum=0.0)
        if "minimum_epoch_coverage" in values
        else 0.75
    )
    return TrimConfig(mode=mode, minimum_epoch_coverage=coverage)
```

Pass `trim=_trim_config(root)` in the `CorrectionConfig(...)` construction.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check src tests`
Expected: PASS, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/config.py tests/test_config.py
git commit -m "feat: add optional trim configuration section"
```

---

### Task 2: Output window resolution

**Files:**
- Create: `src/mri_correction/window.py`
- Test: `tests/test_window.py`

**Interfaces:**
- Consumes: `TrimConfig` from Task 1.
- Produces: `OutputWindow(start: int, stop: int)` with zero-based half-open input-sample bounds and a `length` property; `resolve_output_window(volume_starts, *, mode, input_sample_count) -> OutputWindow`. `volume_starts` is the zero-based array returned by `select_marker_samples`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_window.py`:

```python
import numpy as np
import pytest

from mri_correction.window import (
    OutputWindow,
    WindowError,
    resolve_output_window,
)


def test_none_mode_covers_the_whole_recording():
    window = resolve_output_window(
        np.array([0, 4500], dtype=np.int64),
        mode="none",
        input_sample_count=10_000,
    )
    assert (window.start, window.stop) == (0, 10_000)
    assert window.length == 10_000


def test_first_to_last_volume_spans_marker_to_marker_inclusive():
    window = resolve_output_window(
        np.array([142_276, 3_107_776], dtype=np.int64),
        mode="first_to_last_volume",
        input_sample_count=3_160_200,
    )
    assert (window.start, window.stop) == (142_276, 3_107_777)
    assert window.length == 2_965_501


def test_already_trimmed_input_is_a_no_op_window():
    window = resolve_output_window(
        np.array([0, 4500, 9000], dtype=np.int64),
        mode="first_to_last_volume",
        input_sample_count=9001,
    )
    assert (window.start, window.stop) == (0, 9001)


def test_window_beyond_the_recording_is_rejected():
    with pytest.raises(WindowError, match="outside the recording"):
        resolve_output_window(
            np.array([0, 9000], dtype=np.int64),
            mode="first_to_last_volume",
            input_sample_count=5000,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_window.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mri_correction.window'`.

- [ ] **Step 3: Implement**

Create `src/mri_correction/window.py`:

```python
"""Resolve which span of an input recording a correction run emits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class WindowError(ValueError):
    """Raised when an output window cannot be resolved."""


@dataclass(frozen=True, slots=True)
class OutputWindow:
    """Zero-based, half-open bounds of the emitted span in input samples."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise WindowError("output window must be a non-empty forward span")

    @property
    def length(self) -> int:
        return self.stop - self.start


def resolve_output_window(
    volume_starts: npt.ArrayLike,
    *,
    mode: str,
    input_sample_count: int,
) -> OutputWindow:
    """Resolve the emitted span for one trim mode.

    ``first_to_last_volume`` reproduces the external trim this cohort was
    prepared with: the first volume marker through the last one, inclusive.
    """
    if not isinstance(input_sample_count, int) or input_sample_count < 1:
        raise WindowError("input sample count must be a positive integer")
    if mode == "none":
        return OutputWindow(start=0, stop=input_sample_count)
    if mode != "first_to_last_volume":
        raise WindowError(f"unsupported trim mode: {mode!r}")

    starts = np.asarray(volume_starts)
    if starts.ndim != 1 or starts.size < 1:
        raise WindowError("volume starts must be a non-empty one-dimensional array")
    start = int(starts[0])
    stop = int(starts[-1]) + 1
    if start < 0 or stop > input_sample_count:
        raise WindowError("resolved output window falls outside the recording")
    return OutputWindow(start=start, stop=stop)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_window.py -v && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/window.py tests/test_window.py
git commit -m "feat: resolve the emitted output window from trim mode"
```

---

### Task 3: Window-aware marker remapping

**Files:**
- Modify: `src/mri_correction/brainvision_io.py:126-150` (`resample_markers`)
- Test: `tests/test_brainvision_io.py`

**Interfaces:**
- Consumes: `OutputWindow` from Task 2.
- Produces: `resample_markers(markers, *, factor, window=None) -> tuple[BrainVisionMarker, ...]`. When `window` is given, markers outside `[window.start, window.stop)` in zero-based input samples are dropped and the rest are shifted to be window-relative before decimation. When `window` is `None` behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_brainvision_io.py`:

```python
def test_resample_markers_shifts_positions_into_the_window():
    markers = (
        BrainVisionMarker("Volume", "V  1", position=101, size=1, channel=0),
        BrainVisionMarker("Volume", "V  1", position=601, size=1, channel=0),
    )
    window = OutputWindow(start=100, stop=1100)
    resampled = resample_markers(markers, factor=5, window=window)
    assert [marker.position for marker in resampled] == [1, 101]


def test_resample_markers_drops_markers_outside_the_window():
    markers = (
        BrainVisionMarker("Volume", "V  1", position=50, size=1, channel=0),
        BrainVisionMarker("Volume", "V  1", position=101, size=1, channel=0),
        BrainVisionMarker("Volume", "V  1", position=2000, size=1, channel=0),
    )
    window = OutputWindow(start=100, stop=1100)
    resampled = resample_markers(markers, factor=5, window=window)
    assert [marker.position for marker in resampled] == [1]


def test_resample_markers_keeps_a_marker_on_the_window_start():
    markers = (BrainVisionMarker("Volume", "V  1", 101, 1, 0),)
    window = OutputWindow(start=100, stop=200)
    assert resample_markers(markers, factor=1, window=window)[0].position == 1


def test_resample_markers_without_a_window_is_unchanged():
    markers = (BrainVisionMarker("Volume", "V  1", 101, 1, 0),)
    assert resample_markers(markers, factor=5)[0].position == 21
```

Import `OutputWindow` from `mri_correction.window` at the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_brainvision_io.py -k resample_markers -v`
Expected: FAIL with `TypeError: resample_markers() got an unexpected keyword argument 'window'`.

- [ ] **Step 3: Implement**

Replace the body of `resample_markers` in `src/mri_correction/brainvision_io.py`:

```python
def resample_markers(
    markers: Iterable[BrainVisionMarker],
    *,
    factor: int,
    window: OutputWindow | None = None,
) -> tuple[BrainVisionMarker, ...]:
    """Map marker positions through an output window and integer decimation.

    Positions are one-based; ``window`` bounds are zero-based and half-open.
    Markers outside the window are dropped rather than clamped, so a corrected
    file never claims an event it does not contain.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise BrainVisionInputError("resampling factor must be a positive integer")
    transformed = []
    for marker in markers:
        position = marker.position
        if window is not None:
            index = position - 1
            if index < window.start or index >= window.stop:
                continue
            position = index - window.start + 1
        transformed.append(
            BrainVisionMarker(
                marker_type=marker.marker_type,
                description=marker.description,
                position=map_brainvision_position(position, factor=factor),
                size=(marker.size + factor - 1) // factor,
                channel=marker.channel,
                date=marker.date,
                user_infos=marker.user_infos,
            )
        )
    return tuple(transformed)
```

Add `from .window import OutputWindow` to the imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_brainvision_io.py -v && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/brainvision_io.py tests/test_brainvision_io.py
git commit -m "feat: map markers through the output window"
```

---

### Task 4: Fuse the output-window slice into decimation

**Files:**
- Modify: `src/mri_correction/pipeline.py:329-341` (`_lowpass_and_decimate`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `OutputWindow` from Task 2.
- Produces: `_lowpass_and_decimate(data, *, sampling_rate, output_sampling_rate, lowpass_hz, window)` where `window` is required. It filters the **full** array, then returns `filtered[:, window.start:window.stop:ratio]`.

This is the step that can silently corrupt every output. Filtering the full array and slicing afterwards is correct; decimating from sample zero and slicing afterwards is not, because the decimation phase would shift by up to `ratio - 1` samples.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_lowpass_and_decimate_anchors_phase_to_the_window_start():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, 1000))
    coefficients = butter(2, 100.0, fs=5000.0)
    filtered = filtfilt(*coefficients, data, axis=1)

    actual = _lowpass_and_decimate(
        data,
        sampling_rate=5000.0,
        output_sampling_rate=1000.0,
        lowpass_hz=100.0,
        window=OutputWindow(start=13, stop=913),
    )
    assert actual.shape == (2, 180)
    # the whole array is filtered, then the window is sliced and decimated
    assert np.array_equal(actual, filtered[:, 13:913:5])
    # decimating first and slicing afterwards would start three samples late
    assert not np.array_equal(actual[:, 0], filtered[:, ::5][:, 3])


def test_lowpass_and_decimate_full_window_matches_legacy_stride():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 500))
    coefficients = butter(2, 100.0, fs=5000.0)
    expected = filtfilt(*coefficients, data, axis=1)[:, ::5]
    actual = _lowpass_and_decimate(
        data,
        sampling_rate=5000.0,
        output_sampling_rate=1000.0,
        lowpass_hz=100.0,
        window=OutputWindow(start=0, stop=500),
    )
    assert np.array_equal(actual, expected)
```

The second test is the byte-for-byte regression gate. Import `OutputWindow`, `butter`, `filtfilt` and `_lowpass_and_decimate` at the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k lowpass_and_decimate -v`
Expected: FAIL with `TypeError: _lowpass_and_decimate() got an unexpected keyword argument 'window'`.

- [ ] **Step 3: Implement**

```python
def _lowpass_and_decimate(
    data: np.ndarray,
    *,
    sampling_rate: float,
    output_sampling_rate: float,
    lowpass_hz: float,
    window: OutputWindow,
) -> np.ndarray:
    """Low-pass the whole array, then take the output window and decimate.

    Filtering before slicing keeps ``filtfilt``'s edge transient outside the
    emitted span. Slicing before decimating anchors the decimation phase to the
    window start, so the output sample grid does not shift.
    """
    ratio = round(sampling_rate / output_sampling_rate)
    coefficients = butter(2, lowpass_hz, fs=sampling_rate)
    filtered = filtfilt(*coefficients, data, axis=1)
    return filtered[:, window.start : window.stop : ratio]
```

Add `from .window import OutputWindow` to the pipeline imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/pipeline.py tests/test_pipeline.py
git commit -m "feat: anchor decimation phase to the output window"
```

---

### Task 5: Wire trimming through `run_correction`

**Files:**
- Modify: `src/mri_correction/pipeline.py:78-256` (`_run_correction`), `:431-438` (`_validate_marker_output_positions`), `_corrected_psd_window`, `_make_provenance`
- Modify: `examples/configuration.yml`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `TrimConfig` (Task 1), `resolve_output_window` (Task 2), `resample_markers(..., window=)` (Task 3), `_lowpass_and_decimate(..., window=)` (Task 4).
- Produces: sidecar gains a `"trim"` object: `{"mode": str, "window_start_sample": int, "window_stop_sample": int, "head_margin_samples": int, "tail_margin_samples": int, "required_head_margin_samples": int, "required_tail_margin_samples": int}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`. Reuse the synthetic-recording fixture the existing pipeline tests use; read them first and follow their construction.

```python
def test_trimmed_run_emits_the_first_to_last_volume_span(tmp_path):
    # a synthetic recording with 12 samples of head margin and 6000 of tail
    paths = _write_synthetic_untrimmed_recording(tmp_path, head=12, tail=6000)
    config = _correction_config(paths, trim_mode="first_to_last_volume")
    summary = run_correction(config)
    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["trim"]["mode"] == "first_to_last_volume"
    assert provenance["trim"]["window_start_sample"] == 12
    assert provenance["trim"]["head_margin_samples"] == 12
    window_length = provenance["trim"]["window_length"]
    assert summary.output_sample_count == (window_length - 1) // 5 + 1


def test_trimmed_run_corrects_the_first_volume(tmp_path):
    paths = _write_synthetic_untrimmed_recording(tmp_path, head=6000, tail=6000)
    config = _correction_config(paths, trim_mode="first_to_last_volume")
    summary = run_correction(config)
    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["markers"]["skipped_group_indices"] == []


def test_untrimmed_mode_none_is_unchanged(tmp_path):
    paths = _write_synthetic_untrimmed_recording(tmp_path, head=0, tail=0)
    summary = run_correction(_correction_config(paths, trim_mode="none"))
    assert summary.output_sample_count > 0
```

`_write_synthetic_untrimmed_recording(tmp_path, *, head, tail)` writes a BrainVision recording whose volume markers start at sample `head` and whose data continues `tail` samples past the last volume marker. `_correction_config(paths, *, trim_mode)` builds a `CorrectionConfig` pointing at it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k trim -v`
Expected: FAIL with `KeyError: 'trim'` from the provenance lookup.

- [ ] **Step 3: Implement**

In `_run_correction`, after `volume_starts` is computed and before the geometry is built:

```python
    window = resolve_output_window(
        volume_starts,
        mode=config.trim.mode,
        input_sample_count=int(raw.n_times),
    )
```

Leave the geometry construction over the full input unchanged — `prepare_fastr_geometry(..., sample_count=int(raw.n_times), allow_edges=True)` already covers the whole recording, which is exactly what gives boundary volumes their epochs.

Replace the output sizing:

```python
    output_sample_count = (window.length - 1) // decimation + 1
```

Pass the window into the decimation call inside the channel-batch loop:

```python
            corrected_output[start:stop] = _lowpass_and_decimate(
                correction.data,
                sampling_rate=input_rate,
                output_sampling_rate=output_rate,
                lowpass_hz=config.processing.lowpass_hz,
                window=window,
            )
```

and into the marker mapping:

```python
        transformed_markers = resample_markers(
            recording.markers,
            factor=decimation,
            window=window,
        )
```

Update `_corrected_psd_window` to return times relative to the window start; it currently derives them from `geometry` in input-sample coordinates, so subtract `window.start / input_rate` from both bounds and clamp to `[0, window.length / input_rate]`.

Add the margin report to `_make_provenance`. Compute the required margins from the geometry rather than hard-coding them:

```python
def _trim_provenance(
    window: OutputWindow,
    *,
    geometry: FastrGeometry,
    input_sample_count: int,
    mode: str,
) -> dict[str, int | str]:
    factor = geometry.interpolation_factor
    required_head = -(-(geometry.epoch.samples_before + geometry.search_radius) // factor)
    required_tail = -(-(geometry.epoch.samples_after + geometry.search_radius) // factor)
    return {
        "mode": mode,
        "window_start_sample": window.start,
        "window_stop_sample": window.stop,
        "window_length": window.length,
        "head_margin_samples": window.start,
        "tail_margin_samples": input_sample_count - window.stop,
        "required_head_margin_samples": int(required_head),
        "required_tail_margin_samples": int(required_tail),
    }
```

Add `"trim": _trim_provenance(...)` to the provenance dict.

Change `examples/configuration.yml` to add:

```yaml
trim:
  mode: first_to_last_volume
  minimum_epoch_coverage: 0.75
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check src tests`
Expected: PASS, including every pre-existing pipeline test.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/pipeline.py examples/configuration.yml tests/test_pipeline.py
git commit -m "feat: trim the emitted window inside the correction pipeline"
```

---

### Task 6: Partial-epoch boundary fit and `Bad_Gradient` annotations

**Files:**
- Modify: `src/mri_correction/fastr.py:1234-1258` (`_fit_channel_noise`), `:1140-1150` (`_extract_epochs`)
- Modify: `src/mri_correction/pipeline.py` (annotation writing)
- Test: `tests/test_fastr.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `TrimConfig.minimum_epoch_coverage`.
- Produces: `_fit_channel_noise(signal, fitted_triggers, window, epoch, *, coverage=None)` where `coverage` is a float array, one entry per trigger, giving the fraction of the epoch inside the signal. Groups below `minimum_epoch_coverage` are left uncorrected and reported in `FastrProvenance.skipped_group_indices`. The pipeline writes one `BrainVisionMarker(marker_type="Bad Interval", description="Bad_Gradient", ...)` per contiguous uncorrected span, in output samples.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fastr.py`:

```python
def test_partial_epoch_is_corrected_over_available_samples():
    rng = np.random.default_rng(0)
    interval = 250
    n_groups = 60
    n = interval * n_groups + 100  # final epoch is clipped
    shape = np.sin(2 * np.pi * np.arange(interval) / interval) * 1000.0
    artifact = np.tile(shape, n // interval + 1)[:n]
    eeg = rng.standard_normal(n) * 5.0
    triggers = np.arange(1, n_groups - 1) * interval
    result = slice_fastr(
        np.vstack([artifact + eeg]),
        triggers,
        neighbor_count=20,
        minimum_epoch_coverage=0.25,
    )
    tail = result.data[0, triggers[-1] : triggers[-1] + 100]
    assert np.std(tail) < 0.5 * np.std(artifact[triggers[-1] : triggers[-1] + 100])


def test_group_below_coverage_threshold_is_skipped_not_corrupted():
    rng = np.random.default_rng(0)
    interval = 250
    n_groups = 60
    n = interval * n_groups + 100  # final epoch is clipped to 40% coverage
    shape = np.sin(2 * np.pi * np.arange(interval) / interval) * 1000.0
    artifact = np.tile(shape, n // interval + 1)[:n]
    eeg = rng.standard_normal(n) * 5.0
    triggers = np.arange(1, n_groups - 1) * interval
    result = slice_fastr(
        np.vstack([artifact + eeg]),
        triggers,
        neighbor_count=20,
        minimum_epoch_coverage=0.99,
    )
    assert triggers.size - 1 in result.provenance.skipped_group_indices.tolist()
    tail = result.data[0, triggers[-1] :]
    original = (artifact + eeg)[triggers[-1] :]
    assert np.array_equal(tail, original)
```

Add to `tests/test_pipeline.py`:

```python
def test_uncorrected_span_is_annotated(tmp_path):
    paths = _write_synthetic_untrimmed_recording(tmp_path, head=0, tail=0)
    summary = run_correction(_correction_config(paths, trim_mode="none"))
    text = summary.output_vmrk.read_text(encoding="utf-8")
    assert "Bad Interval,Bad_Gradient" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fastr.py -k coverage tests/test_pipeline.py -k annotated -v`
Expected: FAIL with `TypeError: slice_fastr() got an unexpected keyword argument 'minimum_epoch_coverage'`.

- [ ] **Step 3: Implement**

In `fastr.py`, add a coverage-aware epoch extractor that pads short epochs with `NaN`, and have `_fit_channel_noise` mask them:

```python
def _extract_epochs_with_coverage(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    samples_before: int,
    samples_after: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract epochs, padding any that run past the recording with NaN."""
    offsets = np.arange(-samples_before, samples_after + 1)
    indices = fine_triggers[:, np.newaxis] + offsets
    inside = (indices >= 0) & (indices < signal.size)
    epochs = np.full(indices.shape, np.nan)
    epochs[inside] = signal[indices[inside]]
    return epochs, inside.mean(axis=1)
```

In `_fit_channel_noise`, compute `amplitudes` from the finite samples only:

```python
    valid = np.isfinite(epochs)
    products = np.where(valid, epochs * templates, 0.0)
    energies = np.sum(np.where(valid, templates**2, 0.0), axis=1)
    amplitudes = np.divide(
        np.sum(products, axis=1),
        energies,
        out=np.ones(fitted_triggers.size),
        where=energies > 0.0,
    )
```

and zero the fitted epoch outside the valid region before `_place_epochs`, so nothing is subtracted where there is no data:

```python
    fitted = np.where(valid, amplitudes[:, np.newaxis] * templates, 0.0)
```

Thread `minimum_epoch_coverage` from `slice_fastr` / `acquisition_group_fastr` / `prepare_fastr_geometry` down to the validity mask in `_run_fastr_with_edges`, replacing the current all-or-nothing `valid` computation for the tail only. Keep the whole-volume rule for the head, where a shortfall genuinely means the volume's own artifact was never recorded.

In `pipeline.py`, convert `geometry.skipped_group_indices` to contiguous output-sample spans and append one marker per span before writing:

```python
def _bad_gradient_markers(
    geometry: FastrGeometry,
    *,
    window: OutputWindow,
    decimation: int,
    output_sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    """Annotate every span the correction left untouched."""
    spans = _contiguous_skipped_spans(geometry)
    markers = []
    for first_sample, last_sample in spans:
        start = (max(first_sample - window.start, 0)) // decimation + 1
        stop = (min(last_sample - window.start, window.length - 1)) // decimation + 1
        if stop < 1 or start > output_sample_count:
            continue
        markers.append(
            BrainVisionMarker(
                marker_type="Bad Interval",
                description="Bad_Gradient",
                position=int(start),
                size=int(max(stop - start + 1, 1)),
                channel=0,
            )
        )
    return tuple(markers)
```

`_contiguous_skipped_spans` groups `geometry.skipped_group_indices` into runs and maps each run to its first and last input sample using `geometry.triggers` and the epoch geometry.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/fastr.py src/mri_correction/pipeline.py tests/
git commit -m "feat: correct partial boundary epochs and annotate untouched spans"
```

---

### Task 7: Residual QC in microvolts, with derived harmonics

**Files:**
- Create: `src/mri_correction/residual_qc.py`
- Test: `tests/test_residual_qc.py`

**Interfaces:**
- Produces:
  - `slice_harmonics(*, groups_per_volume, repetition_time_seconds, nyquist_hz, mains_hz=60.0, exclusion_hz=1.0) -> tuple[float, ...]`
  - `block_residual_uv(data, *, sampling_rate, harmonics, block_seconds=30.0) -> np.ndarray` of shape `(channels, blocks)`, each entry the RMS excess over local background at the harmonics, in microvolts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_residual_qc.py`:

```python
import numpy as np
import pytest

from mri_correction.residual_qc import block_residual_uv, slice_harmonics


def test_mains_harmonic_is_excluded():
    harmonics = slice_harmonics(
        groups_per_volume=18,
        repetition_time_seconds=0.9,
        nyquist_hz=500.0,
    )
    assert 20.0 in harmonics
    assert 40.0 in harmonics
    assert 60.0 not in harmonics
    assert 80.0 in harmonics


def test_non_colliding_slice_rate_keeps_every_harmonic():
    harmonics = slice_harmonics(
        groups_per_volume=14,
        repetition_time_seconds=0.8,
        nyquist_hz=500.0,
    )
    assert 17.5 in harmonics
    assert 52.5 in harmonics


def test_known_residual_amplitude_is_recovered():
    sampling_rate = 1000.0
    duration = 60.0
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    rng = np.random.default_rng(0)
    background = rng.standard_normal(t.size) * 3.0
    residual = 2.0 * np.sqrt(2) * np.sin(2 * np.pi * 20.0 * t)
    measured = block_residual_uv(
        np.vstack([background + residual]),
        sampling_rate=sampling_rate,
        harmonics=(20.0,),
        block_seconds=30.0,
    )
    assert measured.shape == (1, 2)
    assert np.allclose(measured, 2.0, rtol=0.05)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_residual_qc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mri_correction/residual_qc.py`. `slice_harmonics` multiplies the slice rate `groups_per_volume / repetition_time_seconds` by 1, 2, 3, ... up to `nyquist_hz`, dropping any harmonic within `exclusion_hz` of `mains_hz` or its multiples. `block_residual_uv` splits the data into `block_seconds` blocks, takes a Welch PSD per block with a `blackmanharris` window and `nperseg = min(8 * sampling_rate, block_samples)`, and for each harmonic integrates `max(psd - median(local_background), 0)` over a +/-0.15 Hz band, where the background is the two flanking bands from 0.3 to 0.9 Hz either side. The returned value is the square root of the summed excess.

Excluding the mains harmonic is load-bearing: with a 20 Hz slice rate the 3rd harmonic is 60 Hz exactly, and including it reported a 1.75 uV residual as "310x background".

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_residual_qc.py -v && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/residual_qc.py tests/test_residual_qc.py
git commit -m "feat: measure residual gradient artifact in microvolts"
```

---

### Task 8: Estimate the template on a high-passed signal (Niazy stage 2)

**Files:**
- Modify: `src/mri_correction/fastr.py:362-412` (`apply_fastr_batch`), `:1234` (`_fit_channel_noise` call site)
- Modify: `src/mri_correction/config.py` (`ProcessingConfig.template_high_pass_hz: float = 1.0`)
- Test: `tests/test_fastr_batch.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `apply_fastr_batch(data, geometry, alignment, *, template_high_pass_hz=1.0, sampling_rate)`. The template and the least-squares scalar are estimated from a high-passed copy of the channel; the resulting artifact estimate is subtracted from the **unfiltered** channel. `template_high_pass_hz=0.0` restores the current behaviour.

Niazy et al. (2005) stage 2 builds the moving-average template, and fits the
scalar, on `Y_h`, a 1 Hz high-passed copy of the interpolated signal, "to ensure
that the different artifact segments used in the average artifact estimation
have the same baseline", then subtracts the estimate from the original signal so
slow content survives. The current code estimates both from the unfiltered
signal, so baseline shifts leak into the template and bias the scalar.

Measured on nine channel-runs across subjects 0004 and 0012, current versus this
fix versus Analyzer: line residual falls from 1.1-20.9 uV to 0.00-0.43 uV
against Analyzer's 0.02-1.16 uV, broadband amplitude rises to within 0.2 % of
Analyzer in every motion block, and the scalar's spread falls by 10-50x. Clean
blocks are unchanged. Note that the released `fmrib_fastr.m` does not apply this
high-pass; the paper's description measurably outperforms the released code on
this data.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fastr_batch.py`:

```python
def test_baseline_drift_does_not_leak_into_the_template():
    rng = np.random.default_rng(0)
    sf = 1000.0
    interval = 50
    n_groups = 400
    n = interval * (n_groups + 2)
    phase = np.arange(interval) / interval
    shape = np.sin(2 * np.pi * phase) * 1000.0
    artifact = np.tile(shape, n // interval + 1)[:n]
    eeg = rng.standard_normal(n) * 5.0
    drift = np.zeros(n)
    drift[n // 3 : 2 * n // 3] = 400.0  # a step baseline shift, as in movement
    triggers = (np.arange(n_groups) + 1) * interval

    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=n,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )
    alignment = fit_fastr_alignment(artifact + eeg + drift, geometry)
    corrected = apply_fastr_batch(
        np.vstack([artifact + eeg + drift]),
        geometry,
        alignment,
        template_high_pass_hz=1.0,
        sampling_rate=sf,
    )
    span = slice(n // 3 + 5 * interval, 2 * n // 3 - 5 * interval)
    recovered = corrected.data[0, span]
    truth = (eeg + drift)[span]
    # the step survives: it is signal, not gradient artifact
    assert abs(recovered.mean() - truth.mean()) < 20.0
    # and the scalar is not dragged around by it
    assert corrected.provenance.amplitudes.std() < 0.02


def test_zero_high_pass_reproduces_the_unfiltered_estimate():
    rng = np.random.default_rng(1)
    sf = 1000.0
    interval = 50
    n_groups = 200
    n = interval * (n_groups + 2)
    phase = np.arange(interval) / interval
    signal = np.tile(np.sin(2 * np.pi * phase) * 1000.0, n // interval + 1)[:n]
    signal = signal + rng.standard_normal(n) * 5.0
    triggers = (np.arange(n_groups) + 1) * interval

    geometry = prepare_fastr_geometry(
        triggers,
        sample_count=n,
        interpolation_factor=10,
        neighbor_count=20,
        search_radius_samples=3,
    )
    alignment = fit_fastr_alignment(signal, geometry)
    corrected = apply_fastr_batch(
        np.vstack([signal]),
        geometry,
        alignment,
        template_high_pass_hz=0.0,
        sampling_rate=sf,
    )

    # the pre-change code path, computed here explicitly
    interpolated = _interpolate(
        signal,
        geometry.interpolation_taps,
        geometry.interpolation_factor,
    )
    noise, _ = _fit_channel_noise(
        interpolated,
        alignment.fitted_triggers,
        geometry.window,
        geometry.epoch,
    )
    expected = signal - noise[:: geometry.interpolation_factor]
    assert np.allclose(corrected.data[0], expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fastr_batch.py -k high_pass -v`
Expected: FAIL with `TypeError: apply_fastr_batch() got an unexpected keyword argument 'template_high_pass_hz'`.

- [ ] **Step 3: Implement**

In `apply_fastr_batch`, high-pass each channel before interpolation. Filtering at
the input rate rather than on the 10x interpolated grid keeps the filter
well-conditioned at 1 Hz:

```python
    if template_high_pass_hz > 0.0:
        sos = butter(2, template_high_pass_hz, btype="high", fs=sampling_rate,
                     output="sos")
    else:
        sos = None
    for index, channel in enumerate(recording):
        source = sosfiltfilt(sos, channel) if sos is not None else channel
        interpolated = _interpolate(
            source,
            geometry.interpolation_taps,
            geometry.interpolation_factor,
        )
        noise, amplitudes[index] = _fit_channel_noise(
            interpolated,
            alignment.fitted_triggers,
            geometry.window,
            geometry.epoch,
        )
        corrected[index] -= noise[:: geometry.interpolation_factor]
```

`corrected` is still initialised from the **unfiltered** recording, so the
subtraction removes only the artifact estimate and leaves slow content intact.

Add `from scipy.signal import butter, sosfiltfilt` to the fastr imports, add
`template_high_pass_hz: float = 1.0` to `ProcessingConfig` with validation that
it is finite and non-negative and below the input Nyquist, and pass it plus
`sampling_rate=input_rate` from `pipeline._run_correction`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check src tests`
Expected: PASS.

Several existing tests assert exact corrected values and will now differ. For
each one, decide deliberately: if it constructs drift-free synthetic data the
result should be unchanged and a failure is a real bug; if it includes drift,
update the expectation and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction/fastr.py src/mri_correction/config.py src/mri_correction/pipeline.py tests/
git commit -m "fix: estimate the artifact template on a high-passed signal"
```

---

### Task 9: Cohort re-run, validating trimming and the template fix

**Files:**
- Create: `scripts/validate_against_analyzer.py`
- Test: none (this is a validation harness, not shipped behaviour)

**Interfaces:**
- Consumes: the corrected outputs.
- Produces: a JSON report with, per run and channel, the off-harmonic transfer gain versus raw, the full-duration blockwise residual in uV, and agreement with Analyzer.

- [ ] **Step 1: Write the harness**

Port the four measurements used during the investigation: matched-file discovery across `fastr_python_validation_cohort`, `step1_scanner_artifact_pulse_marked` and the untrimmed source; off-harmonic transfer gain; 30 s blockwise residual; Pearson r and slope against Analyzer in 1-40 Hz. Use Task 7's `residual_qc` module for the residual so there is one implementation, not two.

- [ ] **Step 2: Re-run three subjects end to end**

Run the pipeline with `trim.mode: first_to_last_volume` on untrimmed input for `baseline sub0000`, `run4 sub0006` (smallest tail margin, 193 samples) and `run2 sub0012` (worst motion block).

- [ ] **Step 3: Verify the boundary defect is gone**

Expected: `skipped_group_indices` is empty or contains only the final volume's first group; the first 0.9 s of Fp1 has RMS comparable to mid-recording (about 20 uV) rather than 383 uV; output sample count matches the current cohort exactly.

- [ ] **Step 4: Re-run the remaining runs and compare**

Expected: no run regresses on residual or on agreement with Analyzer.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_against_analyzer.py
git commit -m "test: add Analyzer comparison harness"
```

---

## Phase 2: template quality and evidence


### Task 10: Residual QC in the sidecar and the `.vmrk`

**Files:**
- Modify: `src/mri_correction/pipeline.py` (`_make_provenance`, marker assembly)
- Modify: `src/mri_correction/config.py` (`ProcessingConfig` gains `residual_threshold_uv: float = 1.0`)
- Test: `tests/test_pipeline.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `slice_harmonics`, `block_residual_uv` (Task 7); `_bad_gradient_markers` (Task 6).
- Produces: sidecar `"residual_qc"`: `{"block_seconds": float, "harmonics_hz": [...], "threshold_uv": float, "channel_names": [...], "block_residual_uv": [[...]], "worst_block_index": [...], "worst_block_uv": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
def test_residual_qc_is_reported_and_annotated(tmp_path):
    paths = _write_synthetic_untrimmed_recording(tmp_path, head=6000, tail=6000)
    summary = run_correction(_correction_config(paths, trim_mode="first_to_last_volume"))
    provenance = json.loads(summary.provenance_json.read_text())
    qc = provenance["residual_qc"]
    assert 60.0 not in qc["harmonics_hz"]
    assert len(qc["block_residual_uv"]) == summary.channel_count
    assert qc["threshold_uv"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k residual_qc -v`
Expected: FAIL with `KeyError: 'residual_qc'`.

- [ ] **Step 3: Implement**

After the corrected file is written and re-read as `corrected_raw`, compute the QC on the corrected data, add it to the provenance, and extend the marker tuple with one `Bad Interval, Bad_Gradient` marker per block exceeding `residual_threshold_uv` on any channel. Because markers are written before the QC runs, either compute the QC from `corrected_output` before `write_brainvision_recording`, or move the write after the QC. Prefer the former: `corrected_output` is already in memory as a memmap.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mri_correction tests/
git commit -m "feat: report and annotate residual gradient artifact per block"
```

---

### Task 11: Neighbour-count sweep, then choose the default

**Files:**
- Create: `scripts/sweep_neighbor_count.py`
- Modify: `examples/configuration.yml`, `docs/algorithm.md` (only after the sweep)

**Interfaces:**
- Consumes: `residual_qc` (Task 7), the validation harness (Task 9).
- Produces: `docs/validation.md` gains a table of `neighbor_count` against transfer gain and residual.

- [ ] **Step 1: Write the sweep harness**

For each `neighbor_count` in `{10, 20, 30, 40, 60, 80}`, run `prepare_fastr_geometry` / `fit_fastr_alignment` / `apply_fastr_batch` directly on eight clean runs plus every run from subjects 0004, 0007 and 0012. Record per channel: off-harmonic transfer gain versus raw, median full-duration residual in uV, and worst 30 s block residual in uV.

- [ ] **Step 2: Run the sweep**

Expected: transfer gain falls roughly as `sqrt(1 + k/N)`; today's measured points are `N=20 -> 1.046`, `N=40 -> 1.021`, `N=80 -> 1.008`.

- [ ] **Step 3: Choose the default from the curve**

Pick the smallest `neighbor_count` whose transfer gain and worst-block residual both clear Analyzer's measured numbers (gain 1.000, residual 0.07-1.2 uV). If no setting clears both, record the trade-off honestly in `docs/validation.md` and pick the knee of the curve.

- [ ] **Step 4: Update the example config and docs**

Change `examples/configuration.yml` and the "template window is a scientific parameter" paragraph in `docs/algorithm.md` to cite the measured curve rather than a conservative guess.

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_neighbor_count.py examples/configuration.yml docs/
git commit -m "docs: choose neighbor_count from measured transfer and residual"
```

---

## Self-review notes

- Spec defect 1 is addressed by Task 11; defect 2 by Tasks 1-6; defect 3 by Task 8; QC by Tasks 7 and 10. Task 9 validates Phase 1 against the cohort. Defect 4 (missing OBS and ANC stages) is deliberately out of scope until Task 9 re-measures.
- The spec's byte-for-byte regression requirement is Task 4 Step 1, second test.
- The spec's derived-harmonics requirement is Task 7, `slice_harmonics`.
- `minimum_epoch_coverage` is introduced in Task 1 and consumed in Task 6.
- `residual_threshold_uv` is introduced and consumed in Task 10.
- `template_high_pass_hz` is introduced and consumed in Task 8.
- `OutputWindow` is defined in Task 2 and consumed in Tasks 3, 4, 5, 6.
