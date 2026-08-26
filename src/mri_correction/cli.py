"""Command-line entry points for strict acquisition validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne

from .fastr import (
    FastrInputError,
    load_bids_fmri_timing,
    make_group_trigger_samples,
)


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate-timing":
            _validate_timing(arguments)
    except FastrInputError as error:
        # Invalid acquisition timing is an expected answer from a validator, not
        # a crash: report what is wrong so the operator can act on it.
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 1
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mri-correct",
        description="Validate and benchmark simultaneous EEG-fMRI correction inputs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser(
        "validate-timing",
        help="validate BIDS timing and BrainVision volume-marker spacing",
    )
    timing.add_argument("--metadata", type=Path, required=True)
    timing.add_argument("--sampling-rate", type=float, required=True)
    timing.add_argument("--output", type=Path, required=True)
    starts = timing.add_mutually_exclusive_group(required=True)
    starts.add_argument("--volume-starts", type=int, nargs="+")
    starts.add_argument("--vhdr", type=Path)
    return parser


def _validate_timing(arguments: argparse.Namespace) -> None:
    timing = load_bids_fmri_timing(arguments.metadata)
    if arguments.volume_starts is not None:
        volume_starts = arguments.volume_starts
    else:
        volume_starts = _read_volume_starts(arguments.vhdr)
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
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


def _read_volume_starts(path: Path) -> list[int]:
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    descriptions = raw.annotations.description
    volume_indices = [
        index
        for index, description in enumerate(descriptions)
        if description == "Volume/V  1"
    ]
    if not volume_indices:
        raise FastrInputError("no exact Volume/V  1 markers found")
    return [
        round(float(raw.annotations.onset[index]) * raw.info["sfreq"])
        for index in volume_indices
    ]


if __name__ == "__main__":
    raise SystemExit(main())
