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

from benchmark.analysis import (
    cost,
    load_measurements,
    load_outcomes,
    motion_sensitivity,
    robustness,
    suppression_and_transfer,
)
from benchmark.arms import Arm
from benchmark.arms.facetpy import FacetpyArm
from benchmark.arms.fastr_python import FastrPythonArm
from benchmark.arms.matlab_fmrib import MatlabFmribArm
from benchmark.cohort import CohortPaths, load_manifest, select_cohort, write_manifest
from benchmark.config import BenchmarkConfig, BenchmarkPaths
from benchmark.figures import write_figures
from benchmark.harmonise import HarmoniseError
from benchmark.metrics import MetricError
from benchmark.orchestrator import (
    completed_runs,
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
    if arguments.command == "report":
        return _report(config)
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
    outcomes = config.paths.output_root / OUTCOMES
    measurements = config.paths.output_root / MEASUREMENTS
    done = completed_runs(outcomes)
    remaining = [run for run in runs if (run.participant, run.run_index) not in done]
    print(f"{len(arms)} arms over {len(remaining)} recordings", end="")
    print(f" ({len(done)} already measured, skipped)" if done else "")
    for index, run in enumerate(remaining, start=1):
        label = f"{run.participant} run {run.run_index} ({run.motion_stratum})"
        print(f"[{index}/{len(remaining)}] {label}", flush=True)
        tones = plan_run_tones(run, config=config)
        attempts = correct_run(run, arms, config=config, tones=tones)
        try:
            measured = measure_run(run, attempts, config=config, tones=tones)
        except (HarmoniseError, MetricError) as error:
            # One recording that cannot be measured must not end the run. Its
            # outcomes stay unwritten, so resuming comes back to it rather than
            # counting it as done.
            release_native_outputs(attempts)
            print(f"    NOT MEASURED: {type(error).__name__}: {error}", flush=True)
            continue
        write_outcomes(attempts, run, outcomes)
        write_rows(measured, run, measurements)
        release_native_outputs(attempts)
        failed = [attempt for attempt in attempts if not attempt.succeeded]
        print(f"    {len(measured)} arms measured, {len(failed)} attempts failed")
    print(f"outcomes {outcomes}")
    print(f"measurements {measurements}")
    return 0


def _report(config: BenchmarkConfig) -> int:
    """Summarise a finished run and draw its figures."""
    measurements = load_measurements(config.paths.output_root / MEASUREMENTS)
    outcomes = load_outcomes(config.paths.output_root / OUTCOMES)
    motion = motion_sensitivity(measurements)
    print("\nSuppression and what it cost\n")
    print(suppression_and_transfer(measurements).to_string(float_format="%.4f"))
    print("\nProcessing cost\n")
    print(cost(measurements).to_string(float_format="%.1f"))
    print("\nRobustness\n")
    print(robustness(outcomes).to_string(float_format="%.3f"))
    if motion:
        print("\nSensitivity to movement\n")
        for arm in motion:
            print(
                f"  {arm.arm:<26}{arm.slope_uv_per_mm:>8.3f} "
                f"+/- {arm.standard_error:.3f} uV/mm  "
                f"({arm.blocks} blocks, {arm.participants} participants)"
            )
    else:
        print("\nSensitivity to movement: not enough motion range to fit a slope")
    figures = write_figures(measurements, motion, config.paths.output_root / "figures")
    print(f"\n{len(figures)} figures in {figures[0].parent}")
    return 0


def _arms(config: BenchmarkConfig, arguments: argparse.Namespace) -> list[Arm]:
    """Build every arm this run was asked for, refusing any it cannot build.

    Naming arms explicitly is how a run reuses corrections it already has. An
    arm left out has no rows at all rather than partial ones, so whatever is
    reported for it has to come from the build it was actually measured on.
    """
    arms = _available_arms(config, arguments)
    if not arguments.arm:
        return arms
    wanted = list(dict.fromkeys(arguments.arm))
    known = {arm.name: arm for arm in arms}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(
            f"no such arm: {', '.join(unknown)}; this run offers {sorted(known)}"
        )
    return [known[name] for name in wanted]


def _available_arms(
    config: BenchmarkConfig, arguments: argparse.Namespace
) -> list[Arm]:
    """Build every arm the given tools and environments can support."""
    arms: list[Arm] = [FastrPythonArm(config.matched)]
    if arguments.matlab is not None:
        arms.append(
            MatlabFmribArm(
                config.matched, matlab=arguments.matlab, eeglab_root=arguments.eeglab
            )
        )
    if arguments.facetpy is not None:
        arms.append(FacetpyArm(config.matched, arguments.facetpy))
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
    parser.add_argument("command", choices=("select", "run", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matlab", type=Path, help="path to the MATLAB executable")
    parser.add_argument("--eeglab", type=Path, help="path to the EEGLAB root")
    parser.add_argument(
        "--facetpy", type=Path, help="python interpreter of the FACETpy environment"
    )
    parser.add_argument(
        "--participant",
        action="append",
        help="select only this participant; repeatable, for a trial run",
    )
    parser.add_argument(
        "--arm",
        action="append",
        help="run only this arm; repeatable, to reuse corrections already made",
    )
    arguments = parser.parse_args(argv)
    if (arguments.matlab is None) != (arguments.eeglab is None):
        parser.error("--matlab and --eeglab are given together or not at all")
    return arguments


if __name__ == "__main__":
    raise SystemExit(main())
