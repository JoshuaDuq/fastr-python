# Algorithmic & Mathematical Specifications

## Scope & Physical Principles

Simultaneous EEG-fMRI recordings suffer from severe scanner-gradient artifacts (GA) induced by the rapid switching of magnetic field gradients ($G_x, G_y, G_z$) and radiofrequency (RF) pulses. By Faraday's law of induction:

$$\mathcal{E} = -\frac{d\Phi_B}{dt} = -\frac{d}{dt} \iint_S \mathbf{B}(t) \cdot d\mathbf{A}$$

Because the gradient switching waveforms are precisely periodic and synchronized to scanner acquisition clocks, the induced artifact voltage (often reaching tens of millivolts, $10^3$ to $10^4$ times larger than physiological EEG) can be modeled as a deterministic, phase-locked periodic signal.

FASTR-Python removes scanner-gradient artifacts by combining sub-sample temporal alignment, target-excluding Averaged Artifact Subtraction (AAS), optional residual Optimal Basis Sets (OBS via PCA), optional normalized LMS Adaptive Noise Cancellation (ANC), delay-compensated FIR filtering, integer decimation, and stationary line-noise regression.

> [!NOTE]
> Cardioballistic (BCG) artifacts, head motion, electrode pops, and electromyographic (EMG) noise are outside the scope of GA correction and must be treated with dedicated downstream tools (e.g. FACETpy).

---

## Processing model

For each configured run, FASTR-Python executes the following sequential stages:

1. **Header & Recording Ingestion**: Reads BrainVision `.vhdr`, `.eeg`, and `.vmrk` files strictly without modifying the source files.
2. **Scanner Trigger Selection**: Locates all markers matching the declared `timing.marker_type` and `timing.marker_description`, optionally extracting a contiguous user-specified volume block.
3. **Acquisition Geometry Resolution**: Reconstructs exact temporal slots from declared BIDS fMRI metadata (`RepetitionTime`, `SliceTiming`, `MultibandAccelerationFactor`) or measured inter-trigger intervals (`marker_kind: slice`).
4. **Trigger & Interval Validation**: Verifies monotonic sample ordering, absence of anomalous gaps, and epoch completeness across the full recording.
5. **Sub-Sample Alignment**: Interpolates the EEG signal around each trigger using band-limited sinc resampling to estimate and correct sub-sample trigger jitter.
6. **Moving Average Artifact Estimation (AAS)**: Constructs an epoch template for each acquisition slot across $2N$ adjacent epochs (excluding the target epoch) and computes an optimal least-squares amplitude scale per channel.
7. **Adaptive & Gated Window Policies**: Optionally excludes residual outlier epochs from neighboring templates and selects local vs wide template windows.
8. **Residual Optimal Basis Sets (OBS)**: Projects out residual gradient variance across complete volume epochs using an orthogonal principal component basis.
9. **Normalized LMS Adaptive Noise Cancellation (ANC)**: Optionally removes residual high-frequency artifacts using the filtered template as an adaptive reference.
10. **Zero-Phase Filtering & Integer Decimation**: Low-pass filters the corrected signal using an MNE-designed delay-compensated FIR filter and decimates by an integer factor $D = f_{s,\text{in}} / f_{s,\text{out}}$.
11. **Stationary Line-Noise Regression**: Regresses configured power-line sinusoids (50/60 Hz) from EEG channels.
12. **Marker Resampling & Provenance Generation**: Resamples event markers to the output grid, annotates uncorrected boundary groups with `Bad_Gradient`, evaluates spectral QC metrics, and writes the BrainVision output and cryptographic JSON sidecar.

---

## Mathematical Formulation

### 1. Sub-sample Temporal Alignment

Due to unsynchronized scanner and EEG acquisition clocks, triggers may exhibit sub-sample jitter $\tau \in (-0.5, 0.5]$ samples. The signal $s(t)$ around each trigger is upsampled by integer factor $L$ (typically $L=10$) via band-limited sinc interpolation:

$$\tilde{s}(t) = \sum_{n} s[n] \, \mathrm{sinc}\left(\frac{t - n T_s}{T_s}\right)$$

Cross-correlation against the reference channel's high-pass template identifies the optimal fractional delay $\hat{\tau}_k$ within search radius $\pm R$ samples:

$$\hat{\tau}_k = \arg\max_{\tau} \sum_{t} \tilde{s}_{\text{ref}, k}(t - \tau) \, \tilde{A}_{\text{ref}}(t)$$

All channels within volume epoch $k$ are time-shifted by $\hat{\tau}_k$ to align gradient waveforms before averaging.

---

### 2. Averaged Artifact Subtraction (AAS)

For channel $c$ and acquisition slot $k$, the artifact template $A_{c,k}(t)$ is formed by averaging the $2N$ surrounding epochs, strictly excluding the target epoch $k$:

$$A_{c,k}(t) = \frac{1}{2N} \sum_{\substack{j=-N \\ j \neq 0}}^{N} s_{c, k+j}(t - \hat{\tau}_{k+j})$$

To account for minor electrode impedance changes or amplifier gain drift, an optimal scalar amplitude factor $\alpha_{c,k}$ is computed by ordinary least squares:

$$\alpha_{c,k} = \frac{\langle s_{c,k}, A_{c,k} \rangle}{\|A_{c,k}\|^2} = \frac{\sum_{t=0}^{T-1} s_{c,k}(t) \, A_{c,k}(t)}{\sum_{t=0}^{T-1} A_{c,k}(t)^2}$$

The template-subtracted residual epoch $r_{c,k}(t)$ is:

$$r_{c,k}(t) = s_{c,k}(t) - \alpha_{c,k} \, A_{c,k}(t)$$

A high-pass filtered copy of the signal (default $1.0\text{ Hz}$) is used to compute $\alpha_{c,k}$ and $\hat{\tau}_k$, ensuring that slow DC drifts and low-frequency neural rhythms do not distort template scaling.

---

### 3. Optimal Basis Sets (OBS)

Following [Niazy et al. (2005)](references.md#niazy-et-al-2005), residual artifact variance not captured by the moving average (such as subtle cardiac-induced head movements or respiratory modulation of the RF field) is removed using Principal Component Analysis (PCA).

For each channel, residual epochs are arranged into matrix $R \in \mathbb{R}^{M \times T}$, where $M$ is the number of volume epochs and $T$ is the number of samples per volume. Singular Value Decomposition yields:

$$R = U \Sigma V^T$$

The first $K$ right-singular vectors $v_1, v_2, \dots, v_K$ represent the dominant temporal artifact basis. The residual epoch $r_k$ is projected onto this orthogonal basis and subtracted:

$$r_k^{\text{clean}} = r_k - \sum_{i=1}^{K} \langle r_k, v_i \rangle \, v_i$$

#### Automatic Rank Selection

When `residual_obs_rank: auto`, $K$ is selected using the three FMRIB criteria:
1. **Eigenvalue Slope**: The slope of four consecutive normalized eigenvalues drops below 2 percentage points.
2. **Cumulative Variance**: Cumulative explained variance exceeds 80%.
3. **First Component Bound**: The first principal component accounts for less than 5% of total variance.

If these criteria are not met or yield an unstable rank, FASTR-Python raises an error rather than silently defaulting to an arbitrary rank.

---

### 4. Normalized LMS Adaptive Noise Cancellation (ANC)

Normalized Least Mean Squares (NLMS) adaptive filtering uses the estimated artifact as a reference input $x(n)$ to cancel residual artifact from desired signal $d(n)$:

$$e(n) = d(n) - \mathbf{w}^T(n) \, \mathbf{x}(n)$$

$$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{x}(n)\|^2 + \epsilon} \, e(n) \, \mathbf{x}(n)$$

where:
- $\mathbf{w}(n)$ is the $P$-tap adaptive FIR filter weight vector;
- $\mu$ is the normalized adaptation step size ($0 < \mu < 2$);
- $\epsilon$ is a small positive regularization constant preventing division by zero.

> [!WARNING]
> ANC is an opt-in stage. Because NLMS adapts rapidly, it can attenuate narrowband neural rhythms near scanner harmonics. Always verify broadband signal transfer when enabling ANC.

---

### 5. Zero-Phase Filtering & Integer Decimation

To eliminate high-frequency gradient harmonics and downsample to a typical EEG rate (e.g. 5 kHz $\rightarrow$ 500 Hz):
1. **Filter Design**: An linear-phase zero-phase FIR low-pass filter is designed via MNE-Python (`mne.filter.create_filter`). The passband edge $f_{\text{pass}}$ is set by `lowpass_hz`. The transition bandwidth $\Delta f$ is strictly constrained so that stopband attenuation is achieved at or before the output Nyquist frequency:
   $$f_{\text{stop}} = f_{\text{pass}} + \Delta f \le \frac{f_{s,\text{out}}}{2}$$
2. **Delay Compensation**: Filtering is applied with zero phase shift ($t_{\text{delay}} = 0$).
3. **Decimation**: Signal arrays are decimated by integer factor $D = f_{s,\text{in}} / f_{s,\text{out}}$:
   $$y[m] = x[m \cdot D]$$

---

### 6. Stationary Line-Noise Regression

Mains interference (50 Hz or 60 Hz and harmonics) is modeled as stationary sinusoidal regressors:

$$y_{\text{mains}}(t) = \sum_{m} \left( a_m \cos(2\pi f_m t) + b_m \sin(2\pi f_m t) \right)$$

Coefficients $a_m, b_m$ are estimated via multivariate linear regression on EEG channels and subtracted, leaving non-EEG channels unaffected.

---

## Quality Control & Spectral Diagnostics

### 1. Volume Harmonic Evaluation via ZoomFFT

Gradient artifacts concentrate energy at the fundamental volume repetition frequency and its integer harmonics:

$$f_k = \frac{k}{T_R}, \quad k \in \{1, 2, \dots, K_{\max}\}$$

FASTR-Python evaluates power at exact harmonic bins using SciPy's `ZoomFFT` with Welch spectral density normalization. Unlike standard discrete Fourier transforms with fixed frequency bins, ZoomFFT computes the continuous-time Fourier transform at exact harmonic frequencies without bin rounding errors.

### 2. Block-Level Coherent Harmonic RMS

Temporal blocks (default 30 seconds) are evaluated using a detrended rectangular window and Fourier-series normalization. By aligning window lengths to exact multiples of $T_R$, leakage between adjacent harmonic bins is eliminated. Frequencies within $\pm 1.0\text{ Hz}$ of mains harmonics are excluded from attribution.

---

## The 1/TR Limitation

Scanner gradient switching is strictly periodic with fundamental frequency $f_{\text{fund}} = 1 / T_R$. By Fourier series decomposition, all gradient artifact energy resides in the discrete comb:

$$\mathcal{H} = \left\{ f \in \mathbb{R}^+ \;\middle|\; f = \frac{k}{T_R}, \; k \in \mathbb{N} \right\}$$

### Physical Consequence

Any physiological or cognitive neural oscillation that coincides exactly with a comb frequency $f \in \mathcal{H}$ cannot be separated from gradient artifact by stationary spectral filtering.

**Concrete Example**:  
If $T_R = 0.900\text{ s}$, the 9th harmonic occurs at:

$$f_9 = \frac{9}{0.900} = 10.000\text{ Hz}$$

A continuous 10.000 Hz posterior alpha oscillation coincides identically with $f_9$ and will be partially subtracted by the template or OBS basis. However, an adjacent 10.500 Hz oscillation lies off-comb and is preserved with near-unity transfer.

Researchers must measure independent **broadband signal transfer** and evaluate task-related rhythms relative to scanner repetition harmonics.

---

## Known Limitations & Boundary Conditions

1. **Uncorrected Boundary Margins**: Epochs at the very beginning or end of a recording lacking complete moving-average neighborhoods are retained uncorrected and annotated with `Bad_Gradient` in the `.vmrk` file.
2. **Motion Disruption**: Rapid head movements alter the spatial geometry between electrodes and gradient fields, temporarily degrading template subtraction.
3. **Non-EEG Channels**: Channels listed in `processing.non_eeg_channels` (such as `ECG`) undergo unscaled template subtraction and bypass OBS, ANC, and line regression.

---

## External references

- [Niazy et al. (2005)](references.md#niazy-et-al-2005): Optimal basis sets (OBS) and moving template subtraction.
- [Allen et al. (2000)](references.md#allen-et-al-2000): Foundations of Averaged Artifact Subtraction (AAS).
- [BIDS MRI Specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html): Timing metadata definitions.
- [MNE-Python Filter Design](https://mne.tools/stable/generated/mne.filter.create_filter.html): Linear-phase FIR design.
- [FMRIB FASTR Implementation](references.md#fmrib-fastr-implementation): Legacy MATLAB EEGLAB plug-in reference.

