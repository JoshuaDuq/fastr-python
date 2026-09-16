# FMRIB FASTR Parity Audit & Numerical Validation

This document presents a comprehensive audit of the legacy MATLAB [FMRIB EEGLAB plugin (FASTR 2.1)](https://github.com/sccn/fMRIb) alongside numerical and empirical validation of the FASTR-Python implementation.

---

## 1. Audited Reference Provenance

- **Repository**: [`sccn/fMRIb`](https://github.com/sccn/fMRIb)
- **Commit SHA**: `2aa522bc5ec4215f42b3ba8efdb2b84d2a312935` (2024-08-02)
- **Audited Source Files**:
  - `fmrib_fastr.m` (SHA-256: `0c193406735266e94000eb16aeeaf13d62e4e3f9b975f55e19f84e30c12dd4de`)
  - `trigcorrect.m`, `decimate2.m`, `pca_calc.m`
  - `fastranc.c` / `fastranc.m`, `prcorr2.c` / `prcorr2.m`
- **MATLAB Environment**: R2026a Update 3 (`26.1.0.3276743`) with EEGLAB 2024.0 and FMRIB 2.1 plug-in.

---

## 2. Feature & Capability Mapping

| FMRIB 2.1 Feature | FASTR-Python Implementation | Implementation Status | Design & Rigor Improvements |
| --- | --- | :---: | --- |
| **Slice-trigger template subtraction** | `timing.marker_kind: slice` and `slice_fastr` API | Complete | Reconstructed directly from measured trigger intervals; group count declared explicitly. |
| **Volume-trigger template subtraction** | `timing.marker_kind: volume` with BIDS slice timing | Complete | Expands multiband acquisition groups rather than treating whole volume as a single static waveform. |
| **Sub-sample alignment** | `search_radius_samples`, `interpolation_factor` | Complete | Sinc interpolation over band-limited grid; shared reference-channel delays. |
| **Moving template window** | `processing.neighbor_count` | Complete | Requires even count; invalid values fail fast rather than being silently rounded. |
| **Least-squares amplitude** | `apply_fastr_batch` | Complete | High-pass filtered estimate prevents slow DC drifts from biasing artifact amplitude fits. |
| **Missing trigger repair** | `missing_volume_markers: repair` | Complete | Replaces interior missing volume triggers when expected count is declared; ambiguous boundary losses raise errors. |
| **Non-EEG channels** | `processing.non_eeg_channels` | Complete | Excluded from template scaling, OBS PCA, ANC, and line regression; output decimation still applies. |
| **Residual OBS** | `residual_obs`, `residual_obs_rank` | Complete | Supports fixed integer rank or FMRIB automatic rank selection; refits per section. |
| **Sectioned OBS fitting** | `residual_obs_section_seconds` | Complete | Exposed as an explicit configuration parameter rather than hardcoded to 60 seconds. |
| **Adaptive Noise Cancellation** | `adaptive_noise_cancellation` | Complete | Normalized LMS implementation with strict divergence detection and opt-in warnings. |
| **Output low-pass filter** | `processing.lowpass_hz` | Complete | Linear-phase zero-phase FIR filter designed via MNE-Python; delay-compensated. |
| **Integer decimation** | `output_sampling_rate_hz` | Complete | Enforces exact integer ratio ($f_{s,\text{in}} / f_{s,\text{out}} \in \mathbb{Z}^+$) and mandatory anti-aliasing. |
| **UI & Metadata** | YAML configurations, CLI, JSON sidecars | Modernized | Eliminates mutable global EEGLAB structs; produces auditable JSON provenance with SHA-256 hashes. |

---

## 3. Algorithmic Audit & Exact Fixture Validation

The legacy FMRIB pipeline operates through seven canonical steps ([Niazy et al., 2005](references.md#niazy-et-al-2005)):
1. Validate or interpolate scanner triggers to delineate artifact epochs.
2. Interpolate, high-pass filter template-estimation signals, align epochs via cross-correlation, and construct target-excluding moving templates.
3. Fit a least-squares scalar amplitude per epoch and subtract the template.
4. High-pass filter residual epochs and project onto a fixed or automatically selected PCA basis (OBS).
5. Low-pass filter and decimate the residual signal and artifact estimate.
6. Adaptively subtract residual artifact via normalized LMS ANC.
7. Preserve excluded non-EEG channels from PCA and ANC.

### Deterministic LMS Fixture Parity

To verify the numerical correctness of the normalized LMS adaptive filter independently of file I/O:
- A deterministic MATLAB test fixture was synthesized using FMRIB's native MEX/C routine `fastranc.c`.
- Under identical input sequences, FASTR-Python's `fastr_anc` matches both the FMRIB error output vector and adaptation weight trajectory to an absolute and relative numerical tolerance of:

$$\| e_{\text{Python}} - e_{\text{MATLAB}} \|_{\infty} \le 1.0 \times 10^{-13}$$

---

## Representative real-recording comparison

### Evidence Scope & Provenance

The values below represent **project-generated** evidence obtained from a simultaneous EEG-fMRI dataset. They characterize empirical parity under the specific recording setup described and do not constitute a universal performance guarantee across all scanner sequences or electrode montages.

- **Data Characteristics**: Simultaneous EEG-fMRI recording (BrainVision format), channels Fp1, Cz, and ECG.
- **Epoch Selection**: 100 contiguous volume markers beginning at volume 20 (468,001 samples; 93.6002 seconds at 5,000 Hz).
- **Sequence Parameters**: $T_R = 0.900\text{ s}$, multiband acceleration factor = 3, 18 groups per volume ($T_{\text{group}} = 50\text{ ms}$).
- **FASTR Parameters**: Interpolation factor = 4, template window = 20, search radius = 3 samples, pre-trigger fraction = 0.03, 60-second OBS sections, ECG excluded from OBS and ANC.

**Input SHA-256 Hashes**:
- `.vhdr`: `8ea95066f5b2a012a05f48c8115a41117b66b891f9cca9ce6e6158c87d436614`
- `.vmrk`: `b46b3f9d73eede9e2fceba070abb7d33bc883a1a46265d100138c7fc96618d7c`
- `.eeg`: `b1f1ad5b541bd9ae993da199d720e726f8786e3f3d354d962a3b826f40fa2cc2`
- BIDS `.json`: `4a6b44fd7fdf0f35ed620e1e9446c968322cfc1b644be804905952215d82229e`

### Empirical Metric Comparison

| Processing Pipeline Configuration | Raw Uncorrected | MATLAB FMRIB 2.1 | Python BIDS Groups |
| --- | :---: | :---: | :---: |
| **Fixed OBS (Rank 4)**: Scanner-Harmonic RMS ($\mu\mathrm{V}$) | 77.197 | 8.891 | **9.810** |
| **Fixed OBS (Rank 4)**: Broadband Transfer Ratio (1–40 Hz) | — | 0.947 | **1.056** |
| **Fixed OBS (Rank 4)**: ECG Morphological Correlation ($\rho$) | — | 0.860 | **0.871** |
| **Fixed OBS (Rank 4)**: Sample-to-Sample RMSE vs MATLAB ($\mu\mathrm{V}$) | — | — | **8.525** |
| **Automatic OBS**: Scanner-Harmonic RMS ($\mu\mathrm{V}$) | 77.197 | 8.874 | **9.795** |
| **ANC + 100 Hz Lowpass**: Scanner-Harmonic RMS ($\mu\mathrm{V}$) | 77.197 | 8.902 | **9.678** |
| **ANC + 100 Hz Lowpass**: Broadband Transfer Ratio (1–40 Hz) | — | 0.918 | **1.085** |
| **ANC + 100 Hz Lowpass**: ECG Morphological Correlation ($\rho$) | — | 0.872 | **0.872** |

### Scientific Observations

1. **Comparable Residual Suppression**: Both implementations achieve approximately one order of magnitude reduction in scanner-harmonic residual RMS (from 77.2 $\mu\mathrm{V}$ down to 8.9–9.8 $\mu\mathrm{V}$).
2. **Signal Transfer Preservation**: FASTR-Python achieves a broadband transfer ratio of $1.056$ (vs $0.947$ in MATLAB), demonstrating minimal suppression of off-comb neural activity.
3. **ECG Fidelity**: ECG correlation is slightly higher in Python ($0.871$ vs $0.860$), confirming that cardiac complexes are preserved for subsequent BCG correction.
4. **Methodological Differences**: MATLAB models the entire 0.9-second TR as a single monolithic epoch with a fixed 70 Hz template high-pass and `firls`/`filtfilt`. FASTR-Python resolves the true physical multiband geometry (18 acquisition slots per TR) with a 1.0 Hz high-pass and linear-phase delay-compensated MNE FIR filtering.

---

## 4. Reproducing the Comparison

FASTR-Python provides dedicated validation scripts in `validation/`:

```bash
# 1. Run legacy volume-stage reference path
python validation/run_python_reference.py \
  --vhdr /path/to/raw.vhdr \
  --output /path/to/output_ref.vhdr

# 2. Run production BIDS multiband geometry path
python validation/run_python_bids_reference.py \
  --vhdr /path/to/raw.vhdr \
  --bids /path/to/bold.json \
  --output /path/to/output_bids.vhdr

# 3. Compute aggregate comparison metrics against MATLAB export
python validation/compare_fmrib_reference.py \
  --matlab /path/to/matlab_corrected.mat \
  --python /path/to/output_bids.vhdr \
  --output-json /path/to/parity_metrics.json
```

All comparison runners require explicit input and output filepaths and refuse to overwrite existing files.

