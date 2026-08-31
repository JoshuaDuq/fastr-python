# References

## Niazy et al. (2005)

- Niazy, R. K., Beckmann, C. F., Iannetti, G. D., Brady, J. M., and Smith,
  S. M. (2005). “Removal of FMRI environment artifacts from EEG data using
  optimal basis sets.” *NeuroImage*, 28(3), 720–737.
  [doi:10.1016/j.neuroimage.2005.06.067](https://doi.org/10.1016/j.neuroimage.2005.06.067)

## BIDS

- Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S.,
  Duff, E. P., Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O.,
  Handwerker, D. A., Hanke, M., Keator, D., Li, X., Maumet, C., Nichols, T. E.,
  Poline, J.-B., Reynolds, R. C., Sochat, V. V., Triplett, W., Turner, J. A.,
  Varoquaux, G., and Poldrack, R. A. (2016). “The brain imaging data
  structure, a format for organizing and describing outputs of neuroimaging
  experiments.” *Scientific Data*, 3, 160044.
  [doi:10.1038/sdata.2016.44](https://doi.org/10.1038/sdata.2016.44)
- [BIDS MRI specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html),
  including `RepetitionTime`, `SliceTiming`, and
  `MultibandAccelerationFactor`.

## MNE-Python

- Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
  Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., and Hämäläinen,
  M. S. (2013). “MEG and EEG data analysis with MNE-Python.” *Frontiers in
  Neuroscience*, 7, 267.
  [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- [MNE documentation](https://mne.tools/stable/documentation/index.html),
  [BrainVision reader](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html),
  and [EEG reading tutorial](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html).
- [MNE examples and tutorials](https://mne.tools/stable/auto_examples/index.html)
  document the supported analysis patterns used by the project.

## File formats and signal processing

- [BrainVision Core Data Format](https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/),
  the header, binary data, and marker-file format read and written by FASTR.
- [SciPy signal-processing documentation](https://docs.scipy.org/doc/scipy/reference/signal.html),
  including [Welch spectral estimation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
  used by comparison diagnostics.

## FMRIB FASTR implementation

- [FMRIB `fMRIb` repository](https://github.com/sccn/fMRIb), including
  [`fmrib_fastr.m`](https://github.com/sccn/fMRIb/blob/master/fmrib_fastr.m).
  The audited commit and source hashes are recorded in the
  [parity validation](fmrib-parity-validation.md).

## Packaging and citation

- [PyPA: declaring project metadata](https://packaging.python.org/specifications/declaring-project-metadata/)
  and [writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
- [Citation File Format](https://citation-file-format.github.io/), used by
  [`CITATION.cff`](../CITATION.cff).
