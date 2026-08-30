# FMRIB FASTR parity audit and validation

## Reference audited

The audit used `sccn/fMRIb` commit
`2aa522bc5ec4215f42b3ba8efdb2b84d2a312935` (2024-08-02). The complete
`fmrib_fastr.m`, its EEGLAB wrapper and GUI, `trigcorrect.m`, `decimate2.m`,
`pca_calc.m`, `fastranc.c`/`.m`, and `prcorr2.c`/`.m` were reviewed. The GitHub
and locally installed FMRIB 2.1 copies of `fmrib_fastr.m` both have SHA-256
`0c193406735266e94000eb16aeeaf13d62e4e3f9b975f55e19f84e30c12dd4de`.

MATLAB was R2026a Update 3 (`26.1.0.3276743`) with EEGLAB and FMRIB 2.1.

## Capability mapping

| FMRIB behavior | Python implementation | Status |
| --- | --- | --- |
| Slice-trigger template subtraction | `slice_fastr` low-level API | Complete |
| Volume-trigger template subtraction | BIDS-derived acquisition-group pipeline; a one-group timing gives classical volume mode | Complete, with a more precise production geometry |
| Interpolation factor | `processing.interpolation_factor` | Complete |
| Even moving-template window | `processing.neighbor_count` | Complete; invalid values fail instead of being silently changed |
| ±3-sample sub-sample alignment | `search_radius_samples` and shared alignment fit | Complete and configurable |
| Least-squares artifact amplitude | `apply_fastr_batch` | Complete |
| Relative trigger position (`pre_frac`) | `processing.pre_trigger_fraction` | Complete |
| Missing-trigger correction | Explicit interior volume repair followed by BIDS group synthesis | Complete for the production volume-marker interface; ambiguous boundary loss is rejected |
| Excluded/non-EEG channels | `processing.non_eeg_channels` | Complete across template scaling, OBS, and ANC |
| OBS disabled/fixed/automatic rank | `residual_obs`, `residual_obs_rank` (`int` or `auto`) | Complete |
| Sectioned OBS fitting | `residual_obs_section_seconds` | Complete and exposed rather than fixed internally |
| Adaptive noise cancellation | `adaptive_noise_cancellation` | Complete, opt-in, with strict divergence errors |
| Output low-pass or no low-pass | `lowpass_hz`, including zero at unchanged sampling rate | Complete |
| Decimation | `output_sampling_rate_hz` with an exact integer ratio | Complete with mandatory anti-alias filtering |
| Trigger timing correction from expected slices × volumes | Repair volume starts, then synthesize groups from BIDS `SliceTiming` and multiband factor | Equivalent production capability without guessing slice markers |
| EEGLAB GUI, `EEG.history`, and positional MATLAB call signature | YAML/CLI, BrainVision, and JSON provenance | Intentional interface difference |

The production filter remains MNE's delay-compensated FIR. It does not copy
MATLAB's twice-applied least-squares response: filtering is an implementation
choice, whereas selecting or disabling the low-pass is the user-facing FMRIB
capability. Python also fails on invalid or unstable inputs where the MATLAB
code changes a value, warns and skips, or enters `keyboard` from a broad catch.

## Algorithm audit

The original function performs these scientifically relevant operations:

1. validate or repair triggers and derive the artifact epoch around each event;
2. interpolate, high-pass the template-estimation signal, align artifacts by
   correlation, and form a target-excluding moving template;
3. fit a scalar template amplitude and subtract the template;
4. high-pass residual epochs and project them onto a fixed or automatically
   selected principal-component basis, optionally in 60-second sections;
5. optionally low-pass and decimate the clean signal and artifact estimate;
6. optionally scale the artifact reference and run normalized LMS ANC; and
7. preserve excluded channels from residual PCA and ANC.

Those operations map respectively to `fastr_timing.py`, `fastr_geometry.py`,
`fastr_processing.py`, `fastr_anc.py`, and `pipeline_io.py`. Python derives
actual multiband acquisition slots from BIDS rather than treating the entire
0.9-second volume as a single repeated waveform.

The automatic-rank criteria are the three FMRIB rules: four consecutive
eigenvalue slopes below 2 percentage points, cumulative explained variance
above 80%, and the first component below 5%. Python rejects spectra for which
the three rules do not identify a stable rank. It re-estimates rank with each
section; MATLAB selects it from the first section and reuses it.

The LMS update has a MATLAB-generated deterministic fixture. Python matches
both the FMRIB error and noise vectors to absolute and relative tolerance
`1e-13`.

## Representative real-recording comparison

No subject data or generated MAT files are tracked. The reproducible runners
used the run-1 BrainVision recording for one representative subject, channels
Fp1, Cz, and ECG, and 100 contiguous volume markers starting with marker 20.
The bounded array held 468,001 samples (93.6002 seconds) at 5 kHz. Parameters
were TR 0.9 seconds, multiband factor 3, 18 acquisition groups per volume,
interpolation factor 4, window 20, search radius 3, pre-trigger fraction 0.03,
60-second OBS sections, and ECG excluded from OBS and ANC.

The input hashes were:

- VHDR: `8ea95066f5b2a012a05f48c8115a41117b66b891f9cca9ce6e6158c87d436614`
- VMRK: `b46b3f9d73eede9e2fceba070abb7d33bc883a1a46265d100138c7fc96618d7c`
- EEG: `b1f1ad5b541bd9ae993da199d720e726f8786e3f3d354d962a3b826f40fa2cc2`
- BIDS JSON: `4a6b44fd7fdf0f35ed620e1e9446c968322cfc1b644be804905952215d82229e`

Metrics use volts internally and report microvolts. Scanner-harmonic RMS is
the median EEG-channel excess amplitude at volume harmonics up to 100 Hz,
excluding the 60 Hz collision. Broadband transfer is corrected/raw amplitude
from 1–40 Hz after excluding scanner and mains bins. ECG correlation uses the
0.5–40 Hz physiological band with those bins removed.

| Setting | Raw | MATLAB FMRIB | Python BIDS groups |
| --- | ---: | ---: | ---: |
| Fixed OBS rank 4: scanner-harmonic RMS (µV) | 77.197 | 8.891 | 9.810 |
| Fixed rank 4: broadband transfer | — | 0.947 | 1.056 |
| Fixed rank 4: ECG correlation | — | 0.860 | 0.871 |
| Fixed rank 4: median MATLAB/Python sample RMSE (µV) | — | — | 8.525 |
| Automatic OBS: scanner-harmonic RMS (µV) | 77.197 | 8.874 | 9.795 |
| ANC + 100 Hz low-pass: scanner-harmonic RMS (µV) | 77.197 | 8.902 | 9.678 |
| ANC + 100 Hz low-pass: broadband transfer | — | 0.918 | 1.085 |
| ANC + 100 Hz low-pass: ECG correlation | — | 0.872 | 0.872 |

For automatic OBS, MATLAB selected rank 13 for both EEG channels. Python
selected `[10, 11]` for Fp1 and `[7, 11]` for Cz across the two sections; ECG
remained rank zero. Aggregate suppression was effectively unchanged.

Exact samples are not an acceptance criterion for this whole-pipeline test.
MATLAB volume mode fits one 0.9-second artifact, uses a fixed 70 Hz template
high-pass, and applies its own `firls`/`filtfilt` output filter. Production
Python fits the 18 actual acquisition slots, uses the configured 1 Hz template
high-pass, and uses the MNE FIR. The comparable acceptance evidence is that
both reduce the measured scanner residual by about an order of magnitude,
retain near-unity non-harmonic broadband amplitude, and preserve the excluded
ECG band similarly. ANC gives only a small residual improvement here while
moving broadband transfer farther from unity, supporting its opt-in warning.

## Reproduction

`validation/fmrib_reference.m` runs the original implementation over an
explicit bounded input. `validation/run_python_reference.py` exercises the
shared classical volume-stage contract,
`validation/run_python_bids_reference.py` exercises production BIDS geometry,
and `validation/compare_fmrib_reference.py` emits the JSON metrics. Every runner
requires explicit input/output paths and refuses to overwrite an output.
