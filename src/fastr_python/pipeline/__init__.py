"""Public configuration-driven correction pipeline."""

from .models import CorrectionSummary, PipelineInputError
from .runner import run_correction

__all__ = ["CorrectionSummary", "PipelineInputError", "run_correction"]
