# Scientific References & Standards

This document catalogs the primary scientific literature, neuroimaging data standards, and foundational software libraries underpinning **FASTR-Python**.

---

## Primary Scientific Literature

### Niazy et al. (2005)

- **Citation**: Niazy, R. K., Beckmann, C. F., Iannetti, G. D., Brady, J. M., & Smith, S. M. (2005). Removal of FMRI environment artifacts from EEG data using optimal basis sets. *NeuroImage*, 28(3), 720–737. [doi:10.1016/j.neuroimage.2005.06.067](https://doi.org/10.1016/j.neuroimage.2005.06.067)
- **Contribution**: Introduction of the Optimal Basis Set (OBS) principle using Principal Component Analysis to capture residual gradient artifact variance unaccounted for by moving-average subtraction.

```bibtex
@article{Niazy2005OBS,
  title     = {Removal of {FMRI} environment artifacts from {EEG} data using optimal basis sets},
  author    = {Niazy, Rami K. and Beckmann, Christian F. and Iannetti, Gian Domenico and Brady, J. Michael and Smith, Stephen M.},
  journal   = {NeuroImage},
  volume    = {28},
  number    = {3},
  pages     = {720--737},
  year      = {2005},
  publisher = {Elsevier},
  doi       = {10.1016/j.neuroimage.2005.06.067}
}
```

### Allen et al. (2000)

- **Citation**: Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from continuously recorded EEG during simultaneous functional MRI. *NeuroImage*, 12(2), 230–239. [doi:10.1006/nimg.2000.0599](https://doi.org/10.1006/nimg.2000.0599)
- **Contribution**: Foundational formulation of Averaged Artifact Subtraction (AAS) for simultaneous EEG-fMRI recordings.

```bibtex
@article{Allen2000AAS,
  title     = {A method for removing imaging artifact from continuously recorded {EEG} during simultaneous functional {MRI}},
  author    = {Allen, Philip J. and Josephs, Oliver and Turner, Robert},
  journal   = {NeuroImage},
  volume    = {12},
  number    = {2},
  pages     = {230--239},
  year      = {2000},
  publisher = {Elsevier},
  doi       = {10.1006/nimg.2000.0599}
}
```

---

## Neuroimaging Standards & Formats

### BIDS

- **Citation**: Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P., Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M., Keator, D., Li, X., Maumet, C., Nichols, T. E., Poline, J.-B., Reynolds, R. C., Sochat, V. V., Triplett, W., Turner, J. A., Varoquaux, G., & Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044. [doi:10.1038/sdata.2016.44](https://doi.org/10.1038/sdata.2016.44)
- **Specification**: [BIDS MRI Specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html), defining `RepetitionTime`, `SliceTiming`, and `MultibandAccelerationFactor`.

```bibtex
@article{Gorgolewski2016BIDS,
  title     = {The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments},
  author    = {Gorgolewski, Krzysztof J. and others},
  journal   = {Scientific Data},
  volume    = {3},
  pages     = {160044},
  year      = {2016},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/sdata.2016.44}
}
```

### BrainVision Core Data Format

- **Specification**: [BrainVision Core Data Format 1.0](https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/), defining the ASCII text header (`.vhdr`), binary multiplexed data file (`.eeg`), and marker file (`.vmrk`) formats read and produced by FASTR-Python via [pybv](https://pybv.readthedocs.io).

---

## Core Scientific Software Stack

### MNE-Python

- **Citation**: Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. S. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- **Relevant Documentation**:
  - [MNE Filter Design](https://mne.tools/stable/generated/mne.filter.create_filter.html): Linear-phase delay-compensated FIR filtering.
  - [MNE BrainVision Reader](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html): Standardized neurophysiological I/O.

```bibtex
@article{Gramfort2013MNE,
  title     = {{MEG} and {EEG} data analysis with {MNE-Python}},
  author    = {Gramfort, Alexandre and others},
  journal   = {Frontiers in Neuroscience},
  volume    = {7},
  pages     = {267},
  year      = {2013},
  publisher = {Frontiers},
  doi       = {10.3389/fnins.2013.00267}
}
```

### SciPy & NumPy

- **Citation**: Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, {\.I}., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., & SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
- **Relevant Tools**:
  - [SciPy ZoomFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ZoomFFT.html): High-resolution discrete evaluation at exact volume harmonic frequencies.
  - [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html): Spectral density estimation.

---

## FMRIB FASTR implementation

- **Repository**: [`sccn/fMRIb`](https://github.com/sccn/fMRIb) repository hosted by the Swartz Center for Computational Neuroscience (SCCN).
- **Commit Audited**: `2aa522bc5ec4215f42b3ba8efdb2b84d2a312935` (August 2, 2024).
- **Reference File**: [`fmrib_fastr.m`](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m) (SHA-256: `0c193406735266e94000eb16aeeaf13d62e4e3f9b975f55e19f84e30c12dd4de`).
- **Audit Details**: Detailed capability mapping, MEX fixtures, and empirical benchmark comparisons are reported in the [FMRIB Parity Audit](fmrib-parity-validation.md).

---

## Packaging and citation

- **Python Packaging Authority**: [Declaring Project Metadata (PEP 621)](https://packaging.python.org/specifications/declaring-project-metadata/) and [pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
- **Citation File Format (CFF)**: [Citation File Format 1.2.0](https://citation-file-format.github.io/), implemented in [`CITATION.cff`](../CITATION.cff) for software citation indexing on GitHub and Zenodo.

