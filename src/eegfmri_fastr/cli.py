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
from .config import MARKER_KINDS, ConfigurationError, load_config
from .demo import write_demo_dataset
from .fastr import (
    AcquisitionGeometry,
    FastrInputError,
    load_bids_fmri_timing,
    slice_marker_geometry,
    volume_marker_geometry,
)
from .pipeline import PipelineInputError, run_correction


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return its exit status."""
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            _run(arguments)
        elif arguments.command == "validate-timing":
            _validate_timing(arguments)
        elif arguments.command == "compare":
            _compare(arguments)
        elif arguments.command == "demo":
            _demo(arguments)
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
        prog="eegfmri-fastr",
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
        help="validate acquisition timing and marker spacing before correcting",
    )
    timing.add_argument(
        "--marker-kind",
        choices=sorted(MARKER_KINDS),
        default="volume",
        help=(
            "volume: markers begin volumes and --metadata supplies the slice "
            "timing. slice: markers are acquisition groups, so --metadata is "
            "rejected and --groups-per-volume is required."
        ),
    )
    timing.add_argument(
        "--metadata",
        type=Path,
        help="BIDS sidecar carrying RepetitionTime, SliceTiming, and "
        "MultibandAccelerationFactor. Volume markers only.",
    )
    timing.add_argument(
        "--groups-per-volume",
        type=int,
        help="how many marked acquisition groups make one volume",
    )
    timing.add_argument(
        "--expected-repetition-time-seconds",
        type=float,
        help="checked against the repetition time measured from the markers",
    )
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

    demo = commands.add_parser(
        "demo",
        help="write a runnable synthetic dataset and configuration",
    )
    demo.add_argument("--output-dir", type=Path, required=True)
    return parser


def _demo(arguments: argparse.Namespace) -> None:
    paths = write_demo_dataset(arguments.output_dir)
    print(json.dumps(asdict(paths), indent=2, default=str))
    print(
        f"\nRun the correction with:\n  eegfmri-fastr run --config {paths.config}",
        file=sys.stderr,
    )


def _run(arguments: argparse.Namespace) -> None:
    summary = run_correction(load_config(arguments.config))
    print(json.dumps(asdict(summary), indent=2, default=str))


def _compare(arguments: argparse.Namespace) -> None:
    from .compare.config import load_compare_config
    from .compare.pipeline import run_comparison

    rows = run_comparison(load_compare_config(arguments.config))
    print(f"COMPARE DONE recordings={len(rows)}")


def _validate_timing(arguments: argparse.Namespace) -> None:
    """Resolve the acquisition geometry and write it out, correcting nothing.

    This runs the same validation the pipeline does, so a protocol can be
    checked before committing to a correction run.
    """
    _validate_timing_arguments(arguments)
    markers = _read_marker_samples(arguments)
    acquisition = _resolve_timing_geometry(arguments, markers)
    result = {
        "metadata": (
            None if arguments.metadata is None else str(arguments.metadata.resolve())
        ),
        "sampling_rate": arguments.sampling_rate,
        "marker_kind": arguments.marker_kind,
        "group_position_source": acquisition.source,
        "repetition_time_seconds": acquisition.repetition_time_seconds,
        "groups_per_volume": acquisition.groups_per_volume,
        "group_offsets_seconds": list(acquisition.group_offsets_seconds),
        "volume_starts": [int(sample) for sample in acquisition.volume_starts],
        "group_triggers": acquisition.group_triggers.tolist(),
    }
    if arguments.output.exists():
        raise FileExistsError(
            f"validation output already exists: {arguments.output}"
        )
    with arguments.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


def _validate_timing_arguments(arguments: argparse.Namespace) -> None:
    """Require exactly the inputs the chosen marker convention needs."""
    if arguments.marker_kind == "volume":
        if arguments.metadata is None:
            raise ConfigurationError(
                "--metadata is required for volume markers: they locate only "
                "the volume, so the slice timing has to be declared"
            )
        if arguments.groups_per_volume is not None:
            raise ConfigurationError(
                "--groups-per-volume applies to acquisition-group markers only"
            )
        if arguments.expected_repetition_time_seconds is not None:
            raise ConfigurationError(
                "--expected-repetition-time-seconds applies to "
                "acquisition-group markers only"
            )
        return
    if arguments.metadata is not None:
        raise ConfigurationError(
            "acquisition-group markers record their own timing; drop --metadata"
        )
    if arguments.groups_per_volume is None:
        raise ConfigurationError(
            "--groups-per-volume is required for acquisition-group markers"
        )


def _read_marker_samples(arguments: argparse.Namespace) -> list[int]:
    if arguments.volume_starts is not None:
        return arguments.volume_starts
    if arguments.marker_type is None or arguments.marker_description is None:
        raise BrainVisionInputError(
            "--marker-type and --marker-description are required with --vhdr"
        )
    return _read_vhdr_marker_samples(
        arguments.vhdr,
        marker_type=arguments.marker_type,
        marker_description=arguments.marker_description,
        sampling_rate=arguments.sampling_rate,
    )


def _resolve_timing_geometry(
    arguments: argparse.Namespace,
    markers: list[int],
) -> AcquisitionGeometry:
    if arguments.marker_kind == "slice":
        return slice_marker_geometry(
            markers,
            sampling_rate=arguments.sampling_rate,
            groups_per_volume=arguments.groups_per_volume,
            expected_repetition_time_seconds=(
                arguments.expected_repetition_time_seconds
            ),
        )
    return volume_marker_geometry(
        markers,
        sampling_rate=arguments.sampling_rate,
        timing=load_bids_fmri_timing(arguments.metadata),
    )


def _read_vhdr_marker_samples(
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
