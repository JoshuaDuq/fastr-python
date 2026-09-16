# FASTR-Python Documentation

Welcome to the technical and scientific documentation for **FASTR-Python**, an open-source Python package for correcting scanner-gradient artifacts (GA) in simultaneous EEG-fMRI recordings.

This documentation provides complete guidance for researchers, clinical neuroscientists, and research software engineers, covering practical pipeline execution, mathematical and physical principles, full configuration parameters, algorithmic audits, and reproducibility standards.

---

## Documentation Navigation Map

```text
docs/
├── User & Operator Guides
│   ├── usage.md                     Operational recipes, CLI workflows, Python API, outputs
│   └── configuration.md             Complete YAML specification, validation rules, parameter matrix
├── Scientific & Algorithmic Foundation
│   ├── algorithm.md                 Signal processing theory, AAS, OBS, ANC, and 1/TR physics
│   ├── validation.md                Protocol validation checklist, signal transfer, residual metrics
│   └── references.md                Literature citations, BIDS standards, software foundations
├── Empirical Verification & Parity
│   ├── fmrib-parity-validation.md   MATLAB FMRIB 2.1 audit, capability mapping, benchmark data
│   └── benchmark.md                 Three-way comparison against MATLAB FMRIB and FACETpy
└── Software Architecture & Development
    ├── architecture.md              Module responsibilities, data flow, API contracts
    └── development.md               Quality gates, testing, typing, contribution standards
```

---

## Directory of Guides

### 1. User & Operator Guides

| Document | Primary Focus | Key Topics |
| --- | --- | --- |
| [**Usage Guide**](usage.md) | How to execute FASTR | Installation (`uv`), synthetic demo, pre-flight timing validation, production YAML runs, folder comparison, Python API recipes, output files. |
| [**Configuration Reference**](configuration.md) | YAML settings & constraints | Field-by-field definitions, types, defaults, physical units, mutual exclusivity rules, validation schemas, and example configs. |

### 2. Scientific & Algorithmic Foundation

| Document | Primary Focus | Key Topics |
| --- | --- | --- |
| [**Algorithmic Foundations**](algorithm.md) | Mathematical theory | Sub-sample sinc interpolation, Averaged Artifact Subtraction (AAS), Optimal Basis Sets (OBS via PCA), normalized LMS ANC, zero-phase delay-compensated FIR filtering, integer decimation, stationary line-noise regression, and the fundamental $1/T_R$ comb limitation. |
| [**Validation Checklist**](validation.md) | Quality assurance & metrics | Software verification, scanner protocol pre-flight checklist, run-level residual QC, volume harmonic RMS, broadband transfer ratio, and ECG waveform preservation. |
| [**Scientific References**](references.md) | Literature & standards | Primary methodology papers (Niazy et al., 2005; Allen et al., 2000), BIDS specifications, MNE-Python, SciPy, BrainVision Core Data Format, and BibTeX citations. |

### 3. Empirical Verification & Parity

| Document | Primary Focus | Key Topics |
| --- | --- | --- |
| [**FMRIB Parity Audit**](fmrib-parity-validation.md) | Comparison with MATLAB | Source-code audit of FMRIB 2.1 (`fmrib_fastr.m`), capability matrix, exact deterministic LMS test fixture ($1\times 10^{-13}$), and empirical real-recording comparisons across Fp1, Cz, and ECG. |

### 4. Software Architecture & Engineering

| Document | Primary Focus | Key Topics |
| --- | --- | --- |
| [**Architecture & Design**](architecture.md) | Codebase structure | Single-responsibility modules, data flow diagrams, separation of production pipeline from research test harnesses, and stable API boundaries. |
| [**Development Guide**](development.md) | Contributing & maintenance | Environment locking with `uv`, Ruff linting/formatting, MyPy strict typing, Pytest test suites, git cleanliness, and release procedures. |

---

## Which Guide Do I Need?

- **"I have a new EEG-fMRI dataset with BIDS metadata."**  
  $\rightarrow$ Start with [Usage](usage.md) and refer to [Configuration](configuration.md#acquisition).
- **"My scanner triggers are logged once per slice / multiband group rather than once per volume."**  
  $\rightarrow$ Read [Configuration: Slice Markers](configuration.md#timing) and [Usage: Validate Timing](usage.md#validate-timing-before-correction).
- **"I need to understand why gradient correction might attenuate 10 Hz alpha."**  
  $\rightarrow$ Read [Algorithm: The 1/TR Limitation](algorithm.md#the-1tr-limitation).
- **"I need to write a paper and justify our gradient correction parameters and citation."**  
  $\rightarrow$ See [References](references.md) and [Validation](validation.md#reproducibility-record).
- **"I want to verify if Python results match legacy FMRIB EEGLAB results."**  
  $\rightarrow$ Consult the [FMRIB Parity Audit](fmrib-parity-validation.md).
- **"I want to extend or contribute to the codebase."**  
  $\rightarrow$ Review [Architecture](architecture.md) and [Development](development.md).

---

## Standards & Physical Units

Across all documentation, configuration files, and software interfaces, FASTR-Python adheres strictly to:

- **Data format**: BrainVision Core Data Format 1.0 (`.vhdr`, `.eeg`, `.vmrk`).
- **Signal amplitudes**: Volts ($\mathrm{V}$) internally following MNE-Python norms; residual diagnostics and QC metrics are reported in microvolts ($\mu\mathrm{V}$).
- **Timing & Durations**: Seconds ($\mathrm{s}$).
- **Frequencies**: Hertz ($\mathrm{Hz}$).
- **Indexing**: 0-based in Python arrays; 1-based in raw BrainVision marker text files.
- **BIDS Alignment**: Repetition time, slice timing, and multiband factor strictly match BIDS MRI metadata specifications.

