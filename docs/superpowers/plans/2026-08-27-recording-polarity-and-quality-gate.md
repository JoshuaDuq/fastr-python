# Recording-Level ECG Polarity and BCG Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one ECG polarity per recording, expose deterministic RR degradation reasons, and prevent BCG output when the cardiac train is degraded.

**Architecture:** Keep Teager candidate generation polarity-independent, construct deterministic positive and negative seed arms before period estimation, and select one arm by the approved inversion-equivariant score. Pass that constant polarity through morphology refinement. Derive quality status from stable reason identifiers and gate the output pipeline before correction or file writes.

**Tech Stack:** Python 3.12, NumPy, SciPy, MNE-Python, pytest, Ruff

---

## Working-tree constraint

The relevant BCG implementation already exists as overlapping uncommitted work in the production and test files below. A worktree from `HEAD` would omit that implementation, while committing or stashing all existing edits would take ownership of user work. Execute in the current checkout, edit only named files, inspect scoped diffs, and do not commit overlapping production files. Preserve unrelated changes.

## File responsibilities

- `src/mri_correction/cardiac.py`: polarity selection, constant-polarity refinement, and quality reasons.
- `src/mri_correction/bcg_pipeline.py`: fail-fast gate before correction and output.
- `tests/test_cardiac.py`: detector and quality regressions.
- `tests/test_bcg_pipeline.py`: no-output orchestration regression.

### Task 1: Select one recording-level ECG polarity

**Files:**
- Modify: `src/mri_correction/cardiac.py`
- Test: `tests/test_cardiac.py`

- [ ] **Step 1: Stabilize the existing synthetic fixture**

Remove the isolated per-beat R inversion from `make_ecg()` and the amplitude-drift fixture. Global inversion remains an explicit test; an isolated opposite-polarity disturbance gets its own rejection test.

- [ ] **Step 2: Write failing polarity regressions**

Add a 10-second ECG fixture with one narrow positive QRS every 900 ms and a sizeable negative secondary deflection 320 ms later. Add tests that require one event per cardiac cycle, `selected_polarity == 1`, and global inversion with identical samples and opposite selected polarity.

```python
def test_detector_uses_one_recording_polarity(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_biphasic_ecg(1_000.0, 10.0)
    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert detection.quality.selected_polarity == 1
```

Strengthen the global-inversion assertion:

```python
positive = detect_r_peaks(ecg, 1_000.0, config=detector_config)
negative = detect_r_peaks(-ecg, 1_000.0, config=detector_config)
np.testing.assert_array_equal(negative.peak_samples, positive.peak_samples)
assert negative.quality.selected_polarity == -positive.quality.selected_polarity
```

- [ ] **Step 3: Write the isolated opposite-polarity disturbance test**

Invert one synthetic QRS event without adding another deflection. Assert that
the selected recording polarity does not change and that the disturbance does
not create an extra event. The detector may omit the inconsistent event and
surface RR degradation; it must not normalize that event into the template.

- [ ] **Step 4: Verify RED**

Run:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_cardiac.py::test_detector_uses_one_recording_polarity \
  tests/test_cardiac.py::test_detector_is_invariant_to_global_ecg_polarity \
  tests/test_cardiac.py::test_opposite_polarity_disturbance_is_not_normalized -v
```

Expected: FAIL because the field does not exist and per-event polarity admits secondary deflections.

- [ ] **Step 5: Add polarity-arm structures**

Add immutable `_PolarityArm` and `_PolaritySeed` structures. `_PolarityArm` owns `polarity`, raw arm candidate count, estimated period, consolidated `_CandidateSet`, and score tuple. `_PolaritySeed` owns the selected polarity, explicitly named `positive_candidate_count` and `negative_candidate_count`, period, and consolidated candidates. Add `selected_polarity: int` to `CardiacDetectionQuality`.

- [ ] **Step 6: Implement exact sign assignment and arm scoring**

Extract `_alignment_radius()`. Add `_signed_extremum()` that searches the approved radius and returns the maximum-absolute conditioned-ECG sample and sign. For each polarity arm:

1. retain candidates assigned that sign;
2. align with `_align_peak(..., polarity=polarity)`;
3. preserve corresponding prominence scores;
4. estimate period and consolidate;
5. require at least three consolidated candidates;
6. count intervals using the inclusive expression `abs(interval - period) <= 0.25 * period`; and
7. score lexicographically as `(coherent_interval_count, consolidated_count, median_prominence)`.

Only expected period-estimation insufficiency makes an arm ineligible; unexpected errors surface. Select the unique maximum. Raise exactly `CardiacInputError("ECG detector found no physiological polarity arm")` if neither arm is eligible and `CardiacInputError("ECG polarity is ambiguous")` on an exact score tie.

- [ ] **Step 7: Thread constant polarity through refinement**

Add required `polarity: int` parameters to `_build_template()`, `_select_events()`, `_align_events()`, `_align_peak()`, `_event_correlation()`, and `_recover_missing_events()`. Alignment uses `np.argmax(polarity * local_signal)`; template and correlation epochs multiply by the constant polarity. Delete `_event_polarity()`.

In `detect_r_peaks()`, replace the original period/initial-seed construction with `_select_polarity_seed()` and pass the selected polarity into every downstream stage and the quality summary.

- [ ] **Step 8: Verify GREEN and refactor**

Run the Step 3 command, then:

```bash
uv run pytest -p no:cacheprovider tests/test_cardiac.py -v
```

Expected: PASS. Keep helpers above callers and remove duplicated radius calculations without adding behavior.

### Task 2: Make polarity and RR quality auditable

**Files:**
- Modify: `src/mri_correction/cardiac.py`
- Test: `tests/test_cardiac.py`

- [ ] **Step 1: Write failing quality assertions**

Require the polarity regression to expose `positive_candidate_count`, `negative_candidate_count`, empty `degradation_reasons`, and `status == "ok"`.

Add `test_quality_summary_reports_rr_reasons()` using peaks `[0, 300, 1_200, 3_000]` at 1 kHz. Assert the fixed tuple:

```python
assert quality.degradation_reasons == (
    "rr_below_minimum",
    "rr_above_maximum",
)
assert quality.status == "degraded"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_cardiac.py::test_detector_uses_one_recording_polarity \
  tests/test_cardiac.py::test_quality_summary_reports_rr_reasons -v
```

Expected: FAIL because the audit fields and reason tuple do not exist.

- [ ] **Step 3: Implement quality fields and fixed-order reasons**

Add exactly:

```python
positive_candidate_count: int
negative_candidate_count: int
degradation_reasons: tuple[str, ...]
```

Pass arm counts through `_PolaritySeed` to `_quality_summary()`. Construct reasons in this order:

```python
reasons = []
if intervals.size and np.any(intervals < config.minimum_rr_seconds):
    reasons.append("rr_below_minimum")
if intervals.size and np.any(intervals > config.maximum_rr_seconds):
    reasons.append("rr_above_maximum")
degradation_reasons = tuple(reasons)
status = "ok" if not degradation_reasons else "degraded"
```

- [ ] **Step 4: Verify GREEN and serialization**

Run the Step 2 command, then:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_cardiac.py tests/test_cardiac_markers.py \
  tests/test_bcg_benchmark.py -v
```

Expected: PASS, including expanded `asdict()` payloads.

### Task 3: Refuse output from degraded detections

**Files:**
- Modify: `src/mri_correction/bcg_pipeline.py`
- Test: `tests/test_bcg_pipeline.py`

- [ ] **Step 1: Write failing gate tests**

Construct a real `CardiacDetection` whose quality contains `degradation_reasons=("rr_above_maximum",)` and assert `_require_usable_detection()` raises `CardiacInputError` matching `degraded ECG detection: rr_above_maximum`.

Add an orchestration test that monkeypatches only the detector boundary to return this result, calls `run_bcg_correction()`, and asserts that `.vhdr`, `.eeg`, `.vmrk`, and `.bcg.json` output paths do not exist after the error.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_bcg_pipeline.py::test_quality_gate_rejects_degraded_detection \
  tests/test_bcg_pipeline.py::test_degraded_detection_writes_no_output -v
```

Expected: FAIL because the gate does not exist and orchestration currently continues.

- [ ] **Step 3: Implement the minimal gate**

Add above `run_bcg_correction()`:

```python
def _require_usable_detection(detection: CardiacDetection) -> None:
    if detection.quality.status == "ok":
        return
    reasons = detection.quality.degradation_reasons
    detail = ": " + ", ".join(reasons) if reasons else ""
    raise CardiacInputError("degraded ECG detection" + detail)
```

Call it immediately after `detect_r_peaks()` and before `correct_bcg()`.

- [ ] **Step 4: Verify GREEN and orchestration**

Run the Step 2 command, then:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_bcg_pipeline.py tests/test_cli.py \
  tests/test_bcg_benchmark.py -v
```

Expected: PASS. Benchmark audit behavior remains unchanged; production correction refuses degraded output.

### Task 4: Validate stratified real recordings read-only

**Files:**
- Do not modify tracked files or cohort outputs.

- [ ] **Step 1: Lock the existing coherent-control train**

Read the current sidecar for `ThermalPainEEGFMRI_run4_sub0003_2026-03-23_11h40.06.701`. Use its stored samples as the pre-repair control; do not regenerate it.

- [ ] **Step 2: Run the repaired detector on ECG only**

Evaluate these fixed inputs:

- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0005/BaselineEEG_sub0005_2026-05-13_10h50.08.052_fastr.vhdr`;
- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0005/ThermalPainEEGFMRI_run1_sub0005_2026-05-13_11h01.07.608_fastr.vhdr`;
- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0017/BaselineEEG_sub0017_2026-08-05_10h50.57.549_fastr.vhdr`;
- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0017/ThermalPainEEGFMRI_run3_sub0017_2026-08-05_11h21.59.324_fastr.vhdr`;
- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0011/ThermalPainEEGFMRI_run1_sub0011_2026-06-26_10h33.54.874_fastr.vhdr`;
- `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0003/ThermalPainEEGFMRI_run4_sub0003_2026-03-23_11h40.06.701_fastr.vhdr`; and
- ordinary-rate control `/Volumes/KINGSTON/EEG_fMRI_data/source_data/fastr_python/sub-0001/BaselineEEG_sub0001_2026-03-02_10h43.51.248_fastr.vhdr`.

Print event count, selected polarity, both arm counts, median/minimum/maximum RR,
degradation reasons, and adjacent conditioned-ECG polarity alternation. Do not
call correction or write BrainVision files.

- [ ] **Step 3: Enforce affected-case acceptance**

For sub-0005 and sub-0017 accept either an `ok` train with alternation `<= 0.05`, or a degraded train that the production gate rejects. For sub-0011 accept either removal of over-maximum intervals or explicit `rr_above_maximum` degradation.

- [ ] **Step 4: Enforce coherent-control acceptance**

Match repaired sub-0003 events to stored events one-to-one within 20 samples. Require status `ok`, count difference `<= 2%`, match fraction `>= 98%`, and median absolute shift `<= 10` samples. If this fails, revisit polarity scoring; do not loosen criteria.

### Task 5: Full verification and code review

**Files:**
- Inspect only the four scoped production/test files.

- [ ] **Step 1: Run all automated checks**

```bash
uv run pytest -p no:cacheprovider
uv run ruff check src tests
git diff --check
```

Expected: all tests pass and both checks are clean.

- [ ] **Step 2: Inspect the scoped diff**

```bash
git diff -- \
  src/mri_correction/cardiac.py \
  src/mri_correction/bcg_pipeline.py \
  tests/test_cardiac.py \
  tests/test_bcg_pipeline.py
```

Confirm no AAS, PCA-OBS, correction-window, YAML, or cohort-output behavior changed.

- [ ] **Step 3: Request code review**

Use `superpowers:requesting-code-review` with the approved spec, this plan, scoped diff, automated output, and real-data validation summary. Address only approved-scope findings.

- [ ] **Step 4: Report completion**

Summarize evidence and any recordings correctly quarantined. Explicitly state that no existing BCG cohort file was changed or rerun.
