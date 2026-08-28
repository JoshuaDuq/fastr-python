"""Command-line entry points for scanner-gradient correction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import mne

from .brainvision import BrainVisionMarkerError
from .brainvision_io import (
    BrainVisionInputError,
    read_brainvision_recording,
    select_marker_samples,
)
from .config import ConfigurationError, load_config
from .fastr import FastrInputError, load_bids_fmri_timing, make_group_trigger_samples
from .pipeline import PipelineInputError, run_correction


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            _run(arguments)
        elif arguments.command == "validate-timing":
            _validate_timing(arguments)
        elif arguments.command == "compare":
            _compare(arguments)
    except (
        BrainVisionInputError,
        BrainVisionMarkerError,
        ConfigurationError,
        FileExistsError,
        FastrInputError,
        PipelineInputError,
    ) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 1
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mri-correct",
        description=(
            "Validate and correct scanner-gradient artifact in EEG-fMRI recordings."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="run the correction described by a YAML configuration",
    )
    run.add_argument("--config", type=Path, required=True)

    timing = commands.add_parser(
        "validate-timing",
        help="validate BIDS timing and BrainVision volume-marker spacing",
    )
    timing.add_argument("--metadata", type=Path, required=True)
    timing.add_argument("--sampling-rate", type=float, required=True)
    timing.add_argument("--output", type=Path, required=True)
    timing.add_argument("--marker-type")
    timing.add_argument("--marker-description")
    starts = timing.add_mutually_exclusive_group(required=True)
    starts.add_argument("--volume-starts", type=int, nargs="+")
    starts.add_argument("--vhdr", type=Path)

    compare = commands.add_parser(
        "compare",
        help="plot uncorrected vs FASTR-corrected recordings from two folders",
    )
    compare.add_argument("--config", type=Path, required=True)
    return parser


def _run(arguments: argparse.Namespace) -> None:
    summary = run_correction(load_config(arguments.config))
    print(json.dumps(asdict(summary), indent=2, default=str))


def _compare(arguments: argparse.Namespace) -> None:
    from .compare.config import load_compare_config
    from .compare.pipeline import run_comparison

    rows = run_comparison(load_compare_config(arguments.config))
    print(f"COMPARE DONE recordings={len(rows)}")


def _validate_timing(arguments: argparse.Namespace) -> None:
    timing = load_bids_fmri_timing(arguments.metadata)
    if arguments.volume_starts is not None:
        volume_starts = arguments.volume_starts
    else:
        if arguments.marker_type is None or arguments.marker_description is None:
            raise BrainVisionInputError(
                "--marker-type and --marker-description are required with --vhdr"
            )
        volume_starts = _read_volume_starts(
            arguments.vhdr,
            marker_type=arguments.marker_type,
            marker_description=arguments.marker_description,
            sampling_rate=arguments.sampling_rate,
        )
    triggers = make_group_trigger_samples(
        volume_starts,
        sampling_rate=arguments.sampling_rate,
        timing=timing,
    )
    result = {
        "metadata": str(arguments.metadata.resolve()),
        "sampling_rate": arguments.sampling_rate,
        "volume_starts": list(volume_starts),
        "groups_per_volume": timing.groups_per_volume,
        "group_triggers": triggers.tolist(),
    }
    if arguments.output.exists():
        raise FileExistsError(
            f"validation output already exists: {arguments.output}"
        )
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


def _read_volume_starts(
    path: Path,
    *,
    marker_type: str,
    marker_description: str,
    sampling_rate: float,
) -> list[int]:
    recording = read_brainvision_recording(path)
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    if not math.isclose(
        raw.info["sfreq"],
        sampling_rate,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise BrainVisionInputError(
            "the declared sampling rate does not match the BrainVision header"
        )
    samples = select_marker_samples(
        recording.markers,
        marker_type=marker_type,
        marker_description=marker_description,
        sample_count=int(raw.n_times),
    )
    return [int(sample) for sample in samples]


if __name__ == "__main__":
    raise SystemExit(main())
