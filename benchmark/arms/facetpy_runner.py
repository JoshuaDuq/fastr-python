"""Correct one recording with FACETpy, inside FACETpy's own environment.

This script is not imported by the benchmark. It is executed by a separate
interpreter, because FACETpy pins numpy 2.1.3 and MNE 1.10.2 against this
project's 2.4.6 and 1.12.1, and because FACETpy is GPL-3.0-only where this
project is GPL-2.0-only. It therefore imports nothing from `benchmark` or from
`fastr_python`, and speaks to them only through the JSON request it is handed
and the JSON summary it writes back.

It applies no low-pass and no decimation. The up- and downsampling around the
correction is FACETpy's alignment interpolation, matching the interpolation
factor the other arms use, not an output filter; the benchmark applies one
shared output filter to every arm afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from facet import (
    AASCorrection,
    BrainVisionExporter,
    CorrespondingSliceCorrection,
    DownSample,
    Loader,
    Pipeline,
    UpSample,
)
from facet.correction.deep_learning import (
    DeepLearningCorrection,
    DeepLearningOutputType,
    list_deep_learning_blueprints,
)

# Averaging each acquisition slot against the same slot in neighbouring volumes,
# which is what this project's own arm does and the only FACETpy mode that can
# represent a multiband sequence.
SLOT_MATCHED = "slot_matched"

# FACETpy's documented default shape: one artifact per volume, averaged against
# its neighbours. Its own recommended configuration rather than a match to
# anything else.
VOLUME_AVERAGED = "volume_averaged"

# One of the author's pretrained networks, run as published. Every one of them
# was trained on synthetic data, so on a real recording they are out of
# distribution and a failure to run at all is a result worth recording.
DEEP_LEARNING = "deep_learning"


def main(request_path: str) -> int:
    """Correct one recording and describe the result beside it."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    triggers = np.asarray(request["triggers"], dtype=int)
    output_vhdr = Path(request["output_vhdr"])

    result = Pipeline(
        [
            Loader(path=request["raw_vhdr"], preload=True),
            _inject(triggers),
            *_around(request, _correction(request)),
            BrainVisionExporter(path=str(output_vhdr), overwrite=False),
        ],
        name=request["mode"],
    ).run()
    # A FACETpy pipeline reports a failure in its result rather than raising,
    # so an unchecked run would write no recording and still exit zero.
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


def _around(request: dict[str, object], correction):
    """Wrap a correction in the interpolation it needs, and nothing more.

    Template subtraction is interpolated up and back down so that artifacts can
    be aligned to a fraction of a sample. A network has no template to align,
    and was trained at the recording's own rate, so resampling around it would
    feed it something it never saw.
    """
    if request["mode"] == DEEP_LEARNING:
        return (correction,)
    factor = request["interpolation_factor"]
    return (UpSample(factor=factor), correction, DownSample(factor=factor))


def _inject(triggers: np.ndarray):
    """Hand FACETpy the trigger array every other arm corrects against.

    Its own SliceTriggerGenerator would space the acquisition slots evenly
    inside the volume. This cohort's multiband slice timing alternates between
    237.5 and 250 samples, so even spacing would put most slots in the wrong
    place, and the arms would differ by their geometry rather than by their
    correction.
    """

    def apply(context):
        return context.with_trigger_samples(triggers, samples_are_absolute=True)

    return apply


def _correction(request: dict[str, object]):
    """Build the correction stage this run asked for."""
    mode = request["mode"]
    if mode == SLOT_MATCHED:
        return CorrespondingSliceCorrection(
            slices_per_volume=request["groups_per_volume"],
            window_size=request["neighbor_count"],
        )
    if mode == VOLUME_AVERAGED:
        return AASCorrection(window_size=request["window_size"])
    if mode == DEEP_LEARNING:
        return _deep_learning(request)
    raise ValueError(f"unknown correction mode: {mode!r}")


def _deep_learning(request: dict[str, object]):
    """Run one pretrained network under the settings it was trained with.

    The architecture blueprint describes the published paper; the chunk length
    and whether the network predicts artifact or clean signal come from this
    checkpoint's own recorded training run, which is the only description that
    matches the weights being loaded.
    """
    blueprints = list_deep_learning_blueprints()
    blueprint = blueprints.get(request["blueprint"])
    if blueprint is None:
        raise ValueError(
            f"no blueprint named {request['blueprint']!r}; "
            f"FACETpy declares {sorted(blueprints)}"
        )
    overrides = {
        "name": request["model_name"],
        "architecture": blueprint.architecture,
        "domain": blueprint.domain,
        "execution_granularity": blueprint.execution_granularity,
        "supports_multichannel": blueprint.supports_multichannel,
        "channel_group_size": blueprint.channel_group_size,
        "requires_channel_positions": blueprint.requires_channel_positions,
        "supports_chunking": True,
        "chunk_size_samples": request["chunk_size_samples"],
        "chunk_overlap_samples": blueprint.chunk_overlap_samples,
        "output_type": DeepLearningOutputType(request["output_type"]),
        "device_preference": "cpu",
    }
    return DeepLearningCorrection(
        model="pytorch_inference",
        model_kwargs={
            "checkpoint_path": request["checkpoint_path"],
            "spec_overrides": overrides,
        },
    )


def _version() -> str:
    from importlib.metadata import version

    return version("facetpy")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
