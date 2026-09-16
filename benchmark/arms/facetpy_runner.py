"""Correct one recording with FACETpy, inside FACETpy's own environment.

This script is not imported by the benchmark. It is executed by a separate
interpreter, because FACETpy pins numpy 2.1.3 and MNE 1.10.2 against this
project's 2.4.6 and 1.12.1, and because FACETpy is GPL-3.0-only where this
project is GPL-2.0-only. It therefore imports nothing from `benchmark` or from
`fastr_python`, and speaks to them only through the JSON request it is handed
and the JSON summary it writes back.

It applies no low-pass and no decimation. The up- and downsampling around a
template correction is alignment interpolation, matching the interpolation
factor the other arms use, not an output filter; the benchmark applies one
shared output filter to every arm afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    DeepLearningExecutionGranularity,
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
    contract = _probe_contract(request) if request["mode"] == DEEP_LEARNING else None
    # Averaging corrections build one template per channel, so processing the
    # channels one at a time gives the same result. It is the only way they fit:
    # upsampling 64 channels tenfold in one array asks for 12.4 GB, and the
    # slot-matched correction was killed by the OS at 64 channels without it.
    channel_wise = (
        contract[0] == 1 if contract is not None else request["mode"] != DEEP_LEARNING
    )

    result = Pipeline(
        [
            Loader(path=request["raw_vhdr"], preload=True),
            _inject(triggers),
            *_stages(request, contract=contract),
            BrainVisionExporter(path=str(output_vhdr), overwrite=False),
        ],
        name=request["mode"],
    ).run(channel_sequential=channel_wise)
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
                "channel_group": None if contract is None else contract[0],
                "chunk_size_samples": None if contract is None else contract[1],
                "facetpy_version": _version(),
            }
        ),
        encoding="utf-8",
    )
    return 0


def _stages(
    request: dict[str, Any], *, contract: tuple[int, int] | None
) -> tuple[Any, ...]:
    """Build the correction, wrapped in the interpolation it needs.

    Template subtraction is interpolated up and back down so that artifacts can
    be aligned to a fraction of a sample. A network has no template to align,
    and was trained at a fixed rate, so resampling around it would feed it
    something it never saw.
    """
    if request["mode"] == DEEP_LEARNING:
        if contract is None:
            raise ValueError(
                f"{request['model_name']} accepted none of the input shapes probed"
            )
        return (_deep_learning(request, contract=contract),)
    factor = request["interpolation_factor"]
    return (UpSample(factor=factor), _template(request), DownSample(factor=factor))


def _inject(triggers: np.ndarray) -> Any:
    """Hand FACETpy the trigger array every other arm corrects against.

    Its own SliceTriggerGenerator would space the acquisition slots evenly
    inside the volume. This cohort's multiband slice timing alternates between
    237.5 and 250 samples, so even spacing would put most slots in the wrong
    place, and the arms would differ by their geometry rather than by their
    correction.
    """

    def apply(context: Any) -> Any:
        return context.with_trigger_samples(triggers, samples_are_absolute=True)

    return apply


def _template(request: dict[str, Any]) -> Any:
    """Build the averaging correction this run asked for."""
    mode = request["mode"]
    if mode == SLOT_MATCHED:
        return CorrespondingSliceCorrection(
            slices_per_volume=request["groups_per_volume"],
            window_size=request["neighbor_count"],
        )
    if mode == VOLUME_AVERAGED:
        return AASCorrection(window_size=request["window_size"])
    raise ValueError(f"unknown correction mode: {mode!r}")


def _deep_learning(request: dict[str, Any], *, contract: tuple[int, int]) -> Any:
    """Run one pretrained network under the settings it was trained with.

    The architecture blueprint describes the published paper. The chunk length,
    whether the network predicts artifact or clean signal, and whether it takes
    one channel at a time all come from this checkpoint instead, because the
    exported weights and the paper do not always agree.
    """
    blueprints = list_deep_learning_blueprints()
    blueprint = blueprints.get(request["blueprint"])
    if blueprint is None:
        raise ValueError(
            f"no blueprint named {request['blueprint']!r}; "
            f"FACETpy declares {sorted(blueprints)}"
        )
    channels, chunk = contract
    granularity = DeepLearningExecutionGranularity.CHANNEL
    if channels == request["channel_count"]:
        granularity = DeepLearningExecutionGranularity.MULTICHANNEL
    elif channels > 1:
        granularity = DeepLearningExecutionGranularity.CHANNEL_GROUP
    return DeepLearningCorrection(
        model="pytorch_inference",
        model_kwargs={
            "checkpoint_path": request["checkpoint_path"],
            "spec_overrides": {
                "name": request["model_name"],
                "architecture": blueprint.architecture,
                "domain": blueprint.domain,
                "execution_granularity": granularity,
                "supports_multichannel": channels > 1,
                "channel_group_size": channels if channels > 1 else None,
                "requires_channel_positions": blueprint.requires_channel_positions,
                "supports_chunking": True,
                "chunk_size_samples": chunk,
                # The networks were trained on adjacent chunks, so they are run
                # on adjacent chunks. Several blueprint overlaps exceed these
                # checkpoints' chunk length outright.
                "chunk_overlap_samples": 0,
                "output_type": DeepLearningOutputType(request["output_type"]),
                "device_preference": "cpu",
            },
        },
    )


def _probe_contract(request: dict[str, Any]) -> tuple[int, int] | None:
    """Ask the checkpoint what shape of input it actually accepts.

    The blueprints describe the published papers and the training configuration
    describes a run, but the exported weights are the only thing being loaded,
    and they do not always agree with either: several take a fixed group of
    channels rather than one or all of them, and several were traced at a chunk
    length other than the one they were trained on.

    A shape counts only if the network returns as many samples as it was given.
    One that returns a different length is not correcting the recording, and
    silently resampling its output would invent data.
    """
    import torch

    model = torch.jit.load(request["checkpoint_path"], map_location="cpu")
    model.eval()
    trained = int(request["chunk_size_samples"])
    channels = _ordered(1, int(request["channel_count"]), 7, 30, 2, 16)
    chunks = _ordered(trained, 512, 1024, 2048, 3584, 4096)
    for channel_count in channels:
        for chunk in chunks:
            try:
                with torch.no_grad():
                    output = model(torch.zeros(1, channel_count, chunk))
            except Exception:
                continue
            if getattr(output, "ndim", 0) == 3 and output.shape[-1] == chunk:
                return channel_count, chunk
    return None


def _ordered(*values: int) -> tuple[int, ...]:
    """Keep the first appearance of each value, so preference survives dedup."""
    seen: dict[int, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return tuple(seen)


def _version() -> str:
    from importlib.metadata import version

    return version("facetpy")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
