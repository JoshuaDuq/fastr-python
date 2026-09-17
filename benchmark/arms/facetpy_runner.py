"""Correct one recording with FACETpy's volume-averaged AAS in FACETpy's environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from facet import (
    AASCorrection,
    BrainVisionExporter,
    DownSample,
    Loader,
    Pipeline,
    UpSample,
)


def main(request_path: str) -> int:
    """Correct one recording and describe the result beside it."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    triggers = np.asarray(request["triggers"], dtype=int)
    output_vhdr = Path(request["output_vhdr"])
    factor = request["interpolation_factor"]

    result = Pipeline(
        [
            Loader(path=request["raw_vhdr"], preload=True),
            _inject(triggers),
            UpSample(factor=factor),
            AASCorrection(window_size=request["window_size"]),
            DownSample(factor=factor),
            BrainVisionExporter(path=str(output_vhdr), overwrite=False),
        ],
        name=request["mode"],
    ).run(channel_sequential=True)

    if not result.success:
        raise RuntimeError(
            f"FACETpy failed in {result.failed_processor}: {result.error}"
        )
    if not output_vhdr.is_file():
        raise RuntimeError(f"FACETpy reported success but wrote no {output_vhdr}")

    Path(request["output_metadata"]).write_text(
        json.dumps(
            {
                "first_volume_sample": int(request["volume_starts"][0]),
                "volume_count": len(request["volume_starts"]) - 1,
                "trigger_count": int(triggers.size),
                "facetpy_version": _version(),
            }
        ),
        encoding="utf-8",
    )
    return 0


def _inject(triggers: np.ndarray) -> Any:
    """Hand FACETpy the trigger array every other arm corrects against."""

    def apply(context: Any) -> Any:
        return context.with_trigger_samples(triggers, samples_are_absolute=True)

    return apply


def _version() -> str:
    from importlib.metadata import version

    return version("facetpy")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
