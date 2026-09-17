# Validation & Scientific Quality Assurance

Scientific validation of scanner-gradient correction requires a two-tiered approach:
1. **Deterministic Software Verification**: Establishing that code implementations conform to numerical and architectural specifications.
2. **Protocol-Specific Signal Validation**: Providing empirical evidence that gradient artifact is attenuated while preserving endogenous neurophysiological signals.

Passing the automated test suite guarantees software correctness; it does not substitute for validating signal preservation on your specific scanner, sequence, montage, and task.

---

## Software checks

The automated test suite verifies all software invariants, numerical contracts, and boundary conditions. Execute from the repository root:

```bash
# 1. Full unit, integration, and regression test suites
uv run pytest

# 2. Strict static analysis and style formatting
uv run ruff check src tests validation
uv run ruff format --check src tests validation

# 3. Static type analysis in strict mode
uv run mypy

# 4. Cleanliness and build integrity
git diff --check
uv build
```

### Coverage of Software Tests

- **Configuration Contracts**: Schema parsing, mutual exclusivity enforcement, and type validation.
- **Acquisition Geometry**: BIDS slice-timing synthesis, group offsets, multiband expansion, and trigger validation.
- **Numerical Invariance**: Batch-size invariance, sub-sample sinc alignment, moving template subtraction, and sectioned OBS.
- **Hardware Parity**: Deterministic MATLAB LMS ANC test fixtures verified to $1\times 10^{-13}$ numerical tolerance.
- **End-to-End Pipelines**: Execution of complete synthetic and real BrainVision recordings without memory leaks or state pollution.

---

## Protocol-Specific Signal Checks

Before initiating cohort studies or clinical trials, execute pilot corrections on representative data collected on the target scanner.

### Pre-Study Verification Matrix

| Check Item | Parameter / Condition | Acceptance Criterion | Verification Tool |
| --- | --- | --- | --- |
| **Marker Stream** | `timing.marker_type` & `description` | Monotonically increasing, exactly 1 trigger per volume (or slice group), 0 unexplained gaps | `fastr-python validate-timing` |
| **Timing Alignment** | BIDS $T_R$ and `SliceTiming` | Integer multiple of EEG sampling period ($T_R \cdot f_s \in \mathbb{Z}^+$) | `fastr-python validate-timing` |
| **Comb Harmonics** | $f_k = k / T_R$ | Residual power at $f_k$ attenuated by $> 15–20\text{ dB}$ relative to raw | `fastr-python compare` / PSD |
| **Broadband Transfer** | Off-comb bins (1–40 Hz) | Signal transfer ratio in $[0.90, 1.10]$ | Metric analysis |
| **Non-EEG Channels** | ECG / EMG channels | Morphological preservation; QRS complex Pearson correlation $\rho > 0.85$ | Metric analysis |
| **Filter Transition** | `processing.lowpass_hz` | Full stopband attenuation reached at or before output Nyquist ($f_{s,\text{out}}/2$) | Pre-flight configuration check |

---

## Run-level checks

For every individual production run, review the generated artifacts before releasing data to downstream analysis:

1. **Cryptographic Integrity**: Inspect `corrected.json` and verify SHA-256 hashes match raw input files.
2. **Timing & Boundary Geometry**: Confirm total volume count matches scanner log; verify that uncorrected boundary margins are annotated with `Bad_Gradient`.
3. **Multi-Channel PSD Review**: Inspect `corrected_psd_before.png` and `corrected_psd_after.png`. Verify sharp attenuation of gradient comb peaks across all scalp channels.
4. **Coherent Harmonic Blocks**: Inspect `residual_qc.volume_harmonic_qc` in the JSON sidecar. Ensure that no prolonged multi-channel residual bursts occur during active experimental blocks.
5. **Channel Failure Recommendations**: Review the `channel_recommendations` section of the provenance sidecar to identify potential bad electrodes requiring downstream repair.

---

## Quantitative Validation Metrics

FASTR-Python formalizes three scientific metrics for evaluating correction fidelity:

### 1. Scanner-Harmonic RMS Residual Excess

Quantifies residual artifact power across the discrete set of scanner volume harmonics $\mathcal{H} = \{f = k / T_R \le f_{\max}\}$ (excluding mains frequencies $\mathcal{M} = \{50, 60\text{ Hz}\} \pm 1\text{ Hz}$):

$$\mathrm{RMS}_{\text{harmonic}} = \sqrt{\frac{1}{|\mathcal{H}|} \sum_{f \in \mathcal{H}} P_{\text{corrected}}(f)}$$

where $P(f)$ is power spectral density in $\mu\mathrm{V}^2/\mathrm{Hz}$. High-quality corrections typically reduce harmonic RMS by one to two orders of magnitude.

### 2. Broadband Signal Transfer Ratio

Measures preservation of off-comb physiological background activity:

$$\text{Transfer}_{[f_1, f_2]} = \frac{\int_{f_1}^{f_2} P_{\text{corrected}}(f) \, df}{\int_{f_1}^{f_2} P_{\text{raw}}(f) \, df} \quad \text{for } f \notin (\mathcal{H} \cup \mathcal{M})$$

Values close to $1.0$ indicate that non-artifact neural activity is preserved without excessive attenuation or artificial amplification.

### 3. Non-EEG / ECG Waveform Correlation

Measures morphological preservation of reference physiological signals:

$$\rho = \frac{\sum_t (s_{\text{raw}}(t) - \bar{s}_{\text{raw}}) \, (s_{\text{corr}}(t) - \bar{s}_{\text{corr}})}{\sqrt{\sum_t (s_{\text{raw}}(t) - \bar{s}_{\text{raw}})^2 \, \sum_t (s_{\text{corr}}(t) - \bar{s}_{\text{corr}})^2}}$$

For ECG channels (which undergo unscaled template subtraction and bypass OBS/ANC), $\rho > 0.85$ confirms that cardiac R-peaks and QRS waveforms remain intact for downstream BCG correction.

---

## Comparison to Legacy Implementations

To benchmark FASTR-Python against legacy MATLAB FMRIB 2.1:

1. Consult the [FMRIB Parity Audit](fmrib-parity-validation.md) for audited capability matrices and MATLAB parity tests.
2. Consult the [Three-Way Benchmark Report](benchmark.md) for the 21-participant cohort evaluation against MATLAB FMRIB and FACETpy.
3. Execute reference runners:
   - `validation/run_python_reference.py`: Shared volume-stage reference path;
   - `validation/run_python_bids_reference.py`: Production BIDS geometry path;
   - `validation/compare_fmrib_reference.py`: Aggregate metrics calculator.

---

## Reproducibility record

When reporting gradient artifact correction in research publications, report:

1. **Software Version**: Output of `fastr-python --version` and Git commit SHA.
2. **Full Configuration**: Archive the exact YAML configuration file used.
3. **Provenance Sidecar**: Archive the generated `.json` provenance file containing SHA-256 hashes and environment details (`software_environment`).
4. **Acquisition Geometry**: State the declared $T_R$, multiband factor, slice timing sequence, and EEG sampling rate.
5. **Empirical Metrics**: Report residual harmonic suppression ($\mu\mathrm{V}$), off-comb broadband transfer ratio, and ECG correlation.

