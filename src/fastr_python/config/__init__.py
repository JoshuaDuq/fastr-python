"""Validated configuration for the correction pipeline."""

from .loading import load_config
from .models import (
    ConfigurationError,
    CorrectionConfig,
    DiagnosticsConfig,
    InputConfig,
    OutputConfig,
    ProcessingConfig,
    QualityControlConfig,
    TimingConfig,
    TrimConfig,
)
from .sections import _PROCESSING_KEYS as _PROCESSING_KEYS
from .sections import _QUALITY_CONTROL_KEYS as _QUALITY_CONTROL_KEYS
from .sections import MARKER_KINDS as MARKER_KINDS

__all__ = [
    "ConfigurationError",
    "CorrectionConfig",
    "DiagnosticsConfig",
    "InputConfig",
    "OutputConfig",
    "ProcessingConfig",
    "QualityControlConfig",
    "TimingConfig",
    "TrimConfig",
    "load_config",
]
