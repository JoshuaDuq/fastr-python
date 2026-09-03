"""Read and assemble strict YAML correction configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from ..correction.timing import FmriAcquisitionTiming
from ..correction.types import FastrInputError
from .models import ConfigurationError, CorrectionConfig, InputConfig, OutputConfig
from .schema import _path_value, _reject_unknown_keys, _require_mapping, _section
from .sections import (
    _ACQUISITION_KEYS,
    _INPUT_KEYS,
    _OPTIONAL_INPUT_KEYS,
    _OPTIONAL_PROCESSING_KEYS,
    _OPTIONAL_TIMING_KEYS,
    _OUTPUT_KEYS,
    _PROCESSING_KEYS,
    _REQUIRED_TOP_LEVEL_KEYS,
    _TIMING_KEYS,
    _TOP_LEVEL_KEYS,
    _diagnostics_config,
    _processing_config,
    _quality_control_config,
    _timing_config,
    _trim_config,
)
from .validation import (
    _validate_channel_failure_policy,
    _validate_marker_selection_trim,
    _validate_timing_sources,
)


def load_config(path: str | Path) -> CorrectionConfig:
    """Load and validate one YAML configuration document.

    Relative paths are resolved against the directory containing ``path``. The
    loader validates the configuration structure and scalar values, but does not
    create outputs or require input files to exist.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"configuration file does not exist: {config_path}"
        raise ConfigurationError(message) from error
    except yaml.YAMLError as error:
        message = f"invalid YAML in configuration: {config_path}"
        raise ConfigurationError(message) from error

    root = _require_mapping(document, "configuration")
    _reject_unknown_keys(root, _TOP_LEVEL_KEYS, "configuration")
    for section in sorted(_REQUIRED_TOP_LEVEL_KEYS):
        if section not in root:
            raise ConfigurationError(f"missing required field: {section}")

    input_values = _section(
        root,
        "input",
        _INPUT_KEYS,
        optional_keys=_OPTIONAL_INPUT_KEYS,
    )
    output_values = _section(root, "output", _OUTPUT_KEYS)
    timing_values = _section(
        root,
        "timing",
        _TIMING_KEYS,
        optional_keys=_OPTIONAL_TIMING_KEYS,
    )
    processing_values = _section(
        root,
        "processing",
        _PROCESSING_KEYS,
        optional_keys=_OPTIONAL_PROCESSING_KEYS,
    )

    base_directory = config_path.parent
    timing = _timing_config(timing_values)
    fmri_metadata = (
        _path_value(input_values, "fmri_metadata", base_directory)
        if "fmri_metadata" in input_values
        else None
    )
    acquisition = _acquisition_config(root)
    _validate_timing_sources(
        timing,
        fmri_metadata=fmri_metadata,
        acquisition=acquisition,
    )
    trim = _trim_config(root)
    _validate_marker_selection_trim(timing, trim)
    processing = _processing_config(processing_values)
    quality_control = _quality_control_config(root)
    _validate_channel_failure_policy(processing, quality_control)
    return CorrectionConfig(
        input=InputConfig(
            raw_vhdr=_path_value(input_values, "raw_vhdr", base_directory),
            fmri_metadata=fmri_metadata,
        ),
        output=OutputConfig(
            vhdr=_path_value(output_values, "vhdr", base_directory),
        ),
        timing=timing,
        processing=processing,
        trim=trim,
        acquisition=acquisition,
        quality_control=quality_control,
        diagnostics=_diagnostics_config(root),
    )


def _acquisition_config(
    root: Mapping[str, object],
) -> FmriAcquisitionTiming | None:
    """Read slice timing declared inline instead of in a BIDS sidecar.

    The three fields are the BIDS ones, transcribed, so a recording whose
    sidecar omits ``SliceTiming`` or ``MultibandAccelerationFactor`` can be
    corrected without hand-editing a JSON file. They go through the same
    validation as the sidecar, so an inline declaration is not a weaker one.
    """
    if "acquisition" not in root:
        return None
    values = _require_mapping(root["acquisition"], "acquisition")
    _reject_unknown_keys(values, _ACQUISITION_KEYS, "acquisition")
    for name in sorted(_ACQUISITION_KEYS):
        if name not in values:
            raise ConfigurationError(f"missing required field: acquisition.{name}")
    slice_timing = values["slice_timing_seconds"]
    if not isinstance(slice_timing, list) or not slice_timing:
        raise ConfigurationError(
            "acquisition.slice_timing_seconds must be a nonempty list of "
            "offsets in seconds, one per slice"
        )
    try:
        return FmriAcquisitionTiming(
            repetition_time_seconds=cast(float, values["repetition_time_seconds"]),
            slice_timing_seconds=cast(tuple[float, ...], tuple(slice_timing)),
            multiband_acceleration_factor=cast(
                int,
                values["multiband_acceleration_factor"],
            ),
        )
    except FastrInputError as error:
        raise ConfigurationError(f"invalid acquisition section: {error}") from error
