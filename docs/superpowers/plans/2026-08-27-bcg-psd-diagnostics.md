# BCG PSD Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add before/after PSD figures to the independent BCG correction pipeline using the same MNE-based diagnostic contract as FASTR.

**Architecture:** Extract FASTR's existing PSD renderer and montage preparation into `mri_correction.psd`, leaving compatibility wrappers in `pipeline.py` so FASTR behavior and its focused tests remain unchanged. BCG will derive the valid sample coverage from `BcgCorrectionResult.corrected_samples`, mark gaps as temporary bad annotations for PSD estimation, and render both conditions over the same time bounds.

**Tech Stack:** Python 3.12, MNE 1.12.1, Matplotlib 3.11.1, NumPy 2.4.6, pytest 9.1.1, Ruff.

---

### Task 1: Extract the shared PSD renderer

**Files:**
- Create: `src/mri_correction/psd.py`
- Modify: `src/mri_correction/pipeline.py:1-55,408-477`
- Test: `tests/test_pipeline.py:175-260`

- [ ] **Step 1: Write the shared module with the existing FASTR renderer.**

Create `src/mri_correction/psd.py` with `PSD_MAX_FREQUENCY_HZ = 100.0`, `save_psd_plot()`, `prepare_psd_raw()`, and the private positioned-channel helper. Move the existing function bodies without changing the MNE arguments, spatial-color behavior, figure title handling, or figure cleanup.

- [ ] **Step 2: Keep FASTR's current private test seam.**

Import the shared constant into `pipeline.py` and retain `_PSD_MAX_FREQUENCY_HZ = PSD_MAX_FREQUENCY_HZ`. Replace the existing renderer bodies with wrappers whose signatures remain:

```python
def _save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    *,
    fmax: float,
    title: str,
    tmin: float,
    tmax: float,
) -> None:
    save_psd_plot(
        raw,
        output_path,
        fmax=fmax,
        title=title,
        tmin=tmin,
        tmax=tmax,
    )


def _prepare_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    return prepare_psd_raw(raw)
```

Delete only the duplicated implementation and its private channel-position helper after the wrappers are in place. FASTR continues calling `_save_psd_plot()` and `_prepare_psd_raw()` from the same locations.

- [ ] **Step 3: Run the existing FASTR PSD tests.**

Run:

```bash
uv run pytest tests/test_pipeline.py::test_psd_diagnostics_use_only_corrected_time_window tests/test_pipeline.py::test_psd_plot_preparation_assigns_standard_channel_locations tests/test_pipeline.py::test_psd_plot_requests_spatial_colors -q
```

Expected: all selected tests pass, demonstrating that the extraction preserved the existing FASTR behavior.

### Task 2: Add failing BCG PSD tests

**Files:**
- Modify: `tests/test_bcg_pipeline.py:1-20,301-333`

- [ ] **Step 1: Add the output-contract assertions to the end-to-end test.**

Extend `test_run_bcg_correction_preserves_ecg_and_writes_pulse_markers()` with:

```python
    assert summary.psd_before == tmp_path / "corrected_psd_before.png"
    assert summary.psd_after == tmp_path / "corrected_psd_after.png"
    assert summary.psd_before.is_file()
    assert summary.psd_after.is_file()
    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert provenance["psd_before"] == str(summary.psd_before)
    assert provenance["psd_after"] == str(summary.psd_after)
    assert 0.0 <= provenance["psd_interval_seconds"]["start"]
    assert provenance["psd_interval_seconds"]["start"] < provenance[
        "psd_interval_seconds"
    ]["end"]
```

Add `import json` and `import mne` with the test module's other imports.

- [ ] **Step 2: Add a focused test for shared BCG coverage.**

Add this test, which captures both renderer calls and checks that BCG's disjoint corrected windows are represented by temporary bad gaps:

```python
def test_bcg_psd_diagnostics_share_the_corrected_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    calls: list[tuple[mne.io.BaseRaw, float, float, float]] = []

    def capture_plot(
        raw: mne.io.BaseRaw,
        output_path: Path,
        *,
        fmax: float,
        title: str,
        tmin: float,
        tmax: float,
    ) -> None:
        calls.append((raw, fmax, tmin, tmax))
        output_path.touch()

    monkeypatch.setattr(bcg_pipeline_module, "save_psd_plot", capture_plot)

    run_bcg_correction(config)

    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]
    assert calls[0][1] == 100.0
    assert calls[0][2] < calls[0][3]
    assert all(
        "bad_psd_gap" in raw.annotations.description
        for raw, _, _, _ in calls
    )
```

- [ ] **Step 3: Run the new tests to verify RED.**

Run:

```bash
uv run pytest tests/test_bcg_pipeline.py::test_run_bcg_correction_preserves_ecg_and_writes_pulse_markers tests/test_bcg_pipeline.py::test_bcg_psd_diagnostics_share_the_corrected_interval -q
```

Expected: both tests fail because `BcgCorrectionSummary` has no PSD fields and the BCG pipeline does not call `save_psd_plot` yet. Do not change production code before observing this failure.

### Task 3: Integrate BCG PSD output and provenance

**Files:**
- Modify: `src/mri_correction/bcg_pipeline.py:1-155,160-230`
- Test: `tests/test_bcg_pipeline.py:301-370`

- [ ] **Step 1: Add PSD paths to the BCG output contract.**

Extend `BcgCorrectionSummary` with:

```python
    psd_before: Path
    psd_after: Path
```

Add `_output_paths(output_vhdr)` returning the existing BrainVision paths, the `.bcg.json` path, and `<stem>_psd_before.png` / `<stem>_psd_after.png`. Pass the complete path mapping to `_ensure_outputs_are_absent()` and include every path in its existing-file check.

- [ ] **Step 2: Preserve the input Raw metadata for the before plot.**

While the preloaded MNE reader is open, copy it before closing:

```python
    raw = mne.io.read_raw_brainvision(
        config.input_vhdr, preload=True, verbose="ERROR"
    )
    try:
        names = tuple(raw.ch_names)
        sampling_rate_hz = float(raw.info["sfreq"])
        data = np.asarray(raw.get_data(), dtype=np.float64)
        before_raw = raw.copy()
    finally:
        raw.close()
```

Keep the existing data extraction and correction logic unchanged.

- [ ] **Step 3: Add a BCG valid-sample interval and gap annotation helper.**

Implement helpers with these exact responsibilities:

```python
def _bcg_psd_interval(
    corrected_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    sample_count: int,
) -> tuple[float, float, tuple[tuple[int, int], ...]]:
    values = np.asarray(corrected_samples)
    if values.ndim != 1 or values.size < 2:
        raise BcgInputError("BCG PSD requires at least two corrected samples")
    if not np.issubdtype(values.dtype, np.integer):
        raise BcgInputError("BCG corrected samples must be integer positions")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= sample_count):
        raise BcgInputError("BCG corrected samples exceed the recording")
    if np.any(np.diff(values) <= 0):
        raise BcgInputError("BCG corrected samples must be strictly increasing")
    gaps = tuple(
        (int(first + 1), int(second))
        for first, second in zip(values[:-1], values[1:], strict=True)
        if second > first + 1
    )
    return (
        float(values[0]) / sampling_rate_hz,
        float(values[-1] + 1) / sampling_rate_hz,
        gaps,
    )


def _mark_bcg_psd_gaps(
    raw: mne.io.BaseRaw,
    gaps: tuple[tuple[int, int], ...],
) -> mne.io.BaseRaw:
    marked = raw.copy()
    if gaps:
        annotations = mne.Annotations(
            onset=[start / raw.info["sfreq"] for start, _ in gaps],
            duration=[(stop - start) / raw.info["sfreq"] for start, stop in gaps],
            description=["bad_psd_gap"] * len(gaps),
        )
        marked.set_annotations(marked.annotations + annotations)
    return marked
```

The interval helper returns time bounds for the full corrected-sample bounding range and half-open gaps for all untouched spans inside it. It must not clamp or repair invalid arrays.

- [ ] **Step 4: Render both BCG figures after writing the corrected recording.**

Import `PSD_MAX_FREQUENCY_HZ` and `save_psd_plot` from `mri_correction.psd`. After `write_brainvision_recording()` succeeds, read the corrected output with MNE, derive the interval once, mark the same gaps on before and after copies, and call:

```python
    fmax = min(PSD_MAX_FREQUENCY_HZ, sampling_rate_hz / 2.0)
    before_psd_raw = _mark_bcg_psd_gaps(before_raw, gaps)
    after_psd_raw = _mark_bcg_psd_gaps(corrected_raw, gaps)
    save_psd_plot(
        before_psd_raw,
        output_paths["psd_before"],
        title="Before BCG correction (complete windows)",
        fmax=fmax,
        tmin=psd_tmin,
        tmax=psd_tmax,
    )
    save_psd_plot(
        after_psd_raw,
        output_paths["psd_after"],
        title="After BCG correction (complete windows)",
        fmax=fmax,
        tmin=psd_tmin,
        tmax=psd_tmax,
    )
```

Close both MNE readers/copies after plotting, preserving the existing exception behavior.

- [ ] **Step 5: Record the diagnostic paths and interval.**

Extend `_write_provenance()` with the output mapping and interval values, adding:

```python
        "psd_before": str(output_paths["psd_before"]),
        "psd_after": str(output_paths["psd_after"]),
        "psd_interval_seconds": {
            "start": psd_tmin,
            "end": psd_tmax,
        },
```

Return both paths in `BcgCorrectionSummary`.

- [ ] **Step 6: Run the focused BCG tests to verify GREEN.**

Run:

```bash
uv run pytest tests/test_bcg_pipeline.py -q
```

Expected: all BCG pipeline tests pass, including the new output and shared-interval checks.

### Task 4: Full verification and cleanup

**Files:**
- Modify: `README.md` or `docs/bcg_methods.md` only if the existing BCG output documentation omits the new figures.

- [ ] **Step 1: Review the complete diff for scope and style.**

Run:

```bash
git diff --check
git diff -- src/mri_correction/psd.py src/mri_correction/pipeline.py src/mri_correction/bcg_pipeline.py tests/test_bcg_pipeline.py tests/test_pipeline.py
```

Confirm that FASTR's plotted interval, titles, and renderer arguments are unchanged, and that only BCG adds new output files.

- [ ] **Step 2: Run the complete test suite.**

Run:

```bash
uv run pytest
```

Expected: exit code 0 with no failed tests.

- [ ] **Step 3: Run lint.**

Run:

```bash
uv run ruff check src tests
```

Expected: exit code 0 with no lint errors.

- [ ] **Step 4: Inspect the final working-tree status.**

Run:

```bash
git status --short
```

Report the BCG PSD implementation files separately from pre-existing user modifications; do not stage or overwrite unrelated work.
