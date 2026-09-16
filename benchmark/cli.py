"""Drive the benchmark: choose the recordings, then correct and measure them.

Selection is a separate command from correction so that the choice of
recordings is frozen, and visibly frozen, before anything is corrected against
it. Correction resumes from the manifest and skips the recordings already
measured, because a cohort-sized run does not fit in one sitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.arms import Arm
from benchmark.arms.facetpy import SLOT_MATCHED, VOLUME_AVERAGED, FacetpyArm
from benchmark.arms.facetpy_deep import FacetpyDeepArm
from benchmark.arms.fastr_python import FastrPythonArm
from benchmark.arms.matlab_fmrib import MatlabFmribArm
from benchmark.arms.model_zoo import discover_exports, fetch
from benchmark.cohort import CohortPaths, load_manifest, select_cohort, write_manifest
from benchmark.config import BenchmarkConfig, BenchmarkPaths
from benchmark.orchestrator import (
    correct_run,
    measure_run,
    plan_run_tones,
    release_native_outputs,
    write_outcomes,
)
from benchmark.report import write_rows

OUTCOMES = "outcomes.jsonl"
MEASUREMENTS = "measurements.csv"


def main(argv: list[str] | None = None) -> int:
    """Run one benchmark command."""
    arguments = _parse_args(argv)
    config = _config(arguments)
    if arguments.command == "select":
        return _select(config, arguments.participant)
    return _run(config, arguments)


def _select(config: BenchmarkConfig, participants: list[str] | None) -> int:
    """Choose the motion-stratified recordings and freeze them to a manifest.

    Stratification always ranks a participant against their own runs, so
    narrowing to one participant selects the same recordings it would have
    selected for them inside the whole cohort.
    """
    runs = select_cohort(
        CohortPaths(
            source_root=config.paths.source_root,
            confounds_root=config.paths.confounds_root,
            protocol_root=config.paths.protocol_root,
        ),
        runs_per_participant=config.runs_per_participant,
    )
    if participants:
        chosen = set(participants)
        runs = tuple(run for run in runs if run.participant in chosen)
        if not runs:
            raise SystemExit(f"no selected recording belongs to {sorted(chosen)}")
    write_manifest(runs, config.paths.manifest)
    selected = len({run.participant for run in runs})
    print(f"selected {len(runs)} recordings from {selected} participants")
    print(f"manifest {config.paths.manifest}")
    return 0


def _run(config: BenchmarkConfig, arguments: argparse.Namespace) -> int:
    """Correct and measure every selected recording with every arm."""
    runs = load_manifest(config.paths.manifest)
    arms = _arms(config, arguments)
    print(f"{len(arms)} arms over {len(runs)} recordings")
    outcomes = config.paths.output_root / OUTCOMES
    measurements = config.paths.output_root / MEASUREMENTS
    for index, run in enumerate(runs, start=1):
        label = f"{run.participant} run {run.run_index} ({run.motion_stratum})"
        print(f"[{index}/{len(runs)}] {label}", flush=True)
        tones = plan_run_tones(run, config=config)
        attempts = correct_run(run, arms, config=config, tones=tones)
        measured = measure_run(run, attempts, config=config, tones=tones)
        write_outcomes(attempts, run, outcomes)
        write_rows(measured, run, measurements)
        release_native_outputs(attempts)
        failed = [attempt for attempt in attempts if not attempt.succeeded]
        print(f"    {len(measured)} arms measured, {len(failed)} attempts failed")
    print(f"outcomes {outcomes}")
    print(f"measurements {measurements}")
    return 0


def _arms(config: BenchmarkConfig, arguments: argparse.Namespace) -> list[Arm]:
    """Build every arm this run was asked for, refusing any it cannot build."""
    arms: list[Arm] = [FastrPythonArm(config.matched)]
    if arguments.matlab is not None:
        arms.append(
            MatlabFmribArm(
                config.matched, matlab=arguments.matlab, eeglab_root=arguments.eeglab
            )
        )
    if arguments.facetpy is not None:
        for mode in (SLOT_MATCHED, VOLUME_AVERAGED):
            arms.append(FacetpyArm(config.matched, arguments.facetpy, mode))
        if arguments.facetpy_source is not None:
            cache = config.paths.output_root / "models"
            arms.extend(
                FacetpyDeepArm(
                    config.matched,
                    interpreter=arguments.facetpy,
                    export=export,
                    checkpoint=fetch(export, cache),
                )
                for export in discover_exports(arguments.facetpy_source)
            )
    return arms


def _config(arguments: argparse.Namespace) -> BenchmarkConfig:
    """Read the paths and settings this benchmark runs under."""
    declared = json.loads(arguments.config.read_text(encoding="utf-8"))
    return BenchmarkConfig(
        paths=BenchmarkPaths(
            source_root=Path(declared["source_root"]),
            confounds_root=Path(declared["confounds_root"]),
            protocol_root=Path(declared["protocol_root"]),
            output_root=Path(declared["output_root"]),
        ),
        runs_per_participant=declared.get("runs_per_participant", 3),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matlab", type=Path, help="path to the MATLAB executable")
    parser.add_argument("--eeglab", type=Path, help="path to the EEGLAB root")
    parser.add_argument(
        "--facetpy", type=Path, help="python interpreter of the FACETpy environment"
    )
    parser.add_argument(
        "--facetpy-source", type=Path, help="FACETpy clone holding the model exports"
    )
    parser.add_argument(
        "--participant",
        action="append",
        help="select only this participant; repeatable, for a trial run",
    )
    arguments = parser.parse_args(argv)
    if (arguments.matlab is None) != (arguments.eeglab is None):
        parser.error("--matlab and --eeglab are given together or not at all")
    return arguments


if __name__ == "__main__":
    raise SystemExit(main())
