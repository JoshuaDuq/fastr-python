"""Stable high-level API for the configuration-driven FASTR pipeline."""

from .config import (
    ConfigurationError,
    CorrectionConfig,
    load_config,
)
from .pipeline import (
    CorrectionSummary,
    PipelineInputError,
    run_correction,
)

__all__ = [
    "ConfigurationError",
    "CorrectionConfig",
    "CorrectionSummary",
    "PipelineInputError",
    "load_config",
    "run_correction",
]
