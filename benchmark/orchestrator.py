"""Run every arm over every recording, and record what each one did.

One recording at a time, every arm twice: once on the recording and once on a
copy carrying the injected tones. Both passes are needed because signal
transfer is measured by difference, and both must come from the same arm at the
same settings or the difference would carry more than the tones.

An arm that fails is recorded as having failed and the run continues. That is
not a fallback: how often an arm fails, and on which recordings, is the
robustness measurement this whole benchmark exists to make. Nothing is retried
and nothing is substituted.

The arms correct at the recorded rate, which costs about 650 MB per output, so
the native-rate recordings are harmonised onto the shared grid and then
deleted. Only the measured span survives, which is what every metric reads.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from benchmark.arms import Arm, ArmError, ArmOutput, MeasuredRun, resolve_geometry
from benchmark.cohort import RunSpec
from benchmark.config import BenchmarkConfig
from benchmark.harmonise import (
    CommonWindow,
    common_window,
    crop,
    guard_volumes,
    to_output_rate,
)
from benchmark.metrics import (
    ToneResult,
    coherent_residual_microvolts,
    harmonic_orders,
    measure_tone_transfer,
)
from benchmark.motion import block_peak_displacement
from benchmark.probes import (
    Tone,
    plan_tones,
    synthesize_tones,
    write_tone_injected_recording,
)

CLEAN = "clean"
TONES = "tones"
PASSES = (CLEAN, TONES)


@dataclass(frozen=True, slots=True)
class ArmAttempt:
    """One arm's attempt at one recording, in one of the two passes."""

    arm: str
    probe: str
    output: ArmOutput | None
    cost: MeasuredRun | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        """Report whether this attempt produced a recording."""
        return self.output is not None


@dataclass(frozen=True, slots=True)
class MeasuredArm:
    """One arm's corrected recording, on the shared grid, with its measurements."""

    arm: str
    channel_names: tuple[str, ...]
    block_residual_microvolts: np.ndarray
    block_peak_displacement: np.ndarray
    tone_transfer: tuple[ToneResult, ...]
    wall_clock_seconds: float
    peak_memory_bytes: int


def correct_run(
    run: RunSpec,
    arms: Sequence[Arm],
    *,
    config: BenchmarkConfig,
    tones: tuple[Tone, ...],
) -> tuple[ArmAttempt, ...]:
    """Correct one recording with every arm, twice, recording every failure."""
    probes = {CLEAN: run, TONES: _tone_injected(run, config=config, tones=tones)}
    attempts: list[ArmAttempt] = []
    for arm in arms:
        for name, probe_run in probes.items():
            directory = config.paths.arm_directory(arm.name, run.participant) / name
            try:
                output = arm.correct(probe_run, output_directory=directory)
            except (ArmError, OSError, ValueError) as error:
                attempts.append(
                    ArmAttempt(
                        arm.name,
                        name,
                        None,
                        getattr(error, "cost", None),
                        f"{type(error).__name__}: {error}",
                    )
                )
                continue
            attempts.append(ArmAttempt(arm.name, name, output, output.cost, None))
    return tuple(attempts)


def measure_run(
    run: RunSpec,
    attempts: Sequence[ArmAttempt],
    *,
    config: BenchmarkConfig,
    tones: tuple[Tone, ...],
) -> tuple[MeasuredArm, ...]:
    """Put every complete arm on one grid and measure it there.

    An arm counts only when both of its passes succeeded, because one pass
    alone cannot say what happened to the injected tones.
    """
    complete = _complete_arms(attempts)
    if not complete:
        return ()
    geometry = resolve_geometry(run, settings=config.matched)
    window = _window(complete, geometry.repetition_time_seconds, config)
    probe = _probe_on_grid(run, window, config=config, tones=tones)
    displacement = block_peak_displacement(
        _confounds(run, config),
        repetition_time_seconds=geometry.repetition_time_seconds,
        block_seconds=config.matched.block_seconds,
        block_count=_block_count(window, config.matched.block_seconds),
    )
    orders = harmonic_orders(
        repetition_time_seconds=geometry.repetition_time_seconds,
        max_hz=config.output.lowpass_hz,
        mains_hz=config.matched.mains_frequency_hz,
        mains_guard_hz=config.tones.mains_guard_hz,
    )
    measured: list[MeasuredArm] = []
    for arm, passes in complete.items():
        clean, names = _on_grid(passes[CLEAN], window, config)
        injected, _ = _on_grid(passes[TONES], window, config)
        eeg = [
            index
            for index, name in enumerate(names)
            if name not in config.matched.non_eeg_channels
        ]
        measured.append(
            MeasuredArm(
                arm=arm,
                channel_names=tuple(names[index] for index in eeg),
                block_residual_microvolts=coherent_residual_microvolts(
                    clean[eeg],
                    sampling_rate=window.sampling_rate,
                    repetition_time_seconds=geometry.repetition_time_seconds,
                    orders=orders,
                    block_seconds=config.matched.block_seconds,
                ),
                block_peak_displacement=displacement,
                tone_transfer=measure_tone_transfer(
                    clean[eeg],
                    injected[eeg],
                    probe,
                    channel_names=[names[index] for index in eeg],
                    tones=tones,
                    sampling_rate=window.sampling_rate,
                ),
                wall_clock_seconds=passes[CLEAN].cost.wall_clock_seconds,
                peak_memory_bytes=passes[CLEAN].cost.peak_memory_bytes,
            )
        )
    return tuple(measured)


def plan_run_tones(run: RunSpec, *, config: BenchmarkConfig) -> tuple[Tone, ...]:
    """Place the probe tones for one recording's own acquisition timing."""
    geometry = resolve_geometry(run, settings=config.matched)
    return plan_tones(
        repetition_time_seconds=geometry.repetition_time_seconds,
        amplitude_volts=config.tones.amplitude_volts,
        mains_hz=config.matched.mains_frequency_hz,
        mains_guard_hz=config.tones.mains_guard_hz,
    )


def release_native_outputs(attempts: Sequence[ArmAttempt]) -> None:
    """Delete the recorded-rate corrections once they have been measured."""
    for attempt in attempts:
        if attempt.output is None:
            continue
        for suffix in (".eeg", ".vhdr", ".vmrk"):
            attempt.output.corrected_vhdr.with_suffix(suffix).unlink(missing_ok=True)


def _tone_injected(
    run: RunSpec, *, config: BenchmarkConfig, tones: tuple[Tone, ...]
) -> RunSpec:
    """Stage the copy of this recording that carries the injected tones."""
    directory = config.paths.probes_root / run.participant
    directory.mkdir(parents=True, exist_ok=True)
    injected = directory / f"{run.raw_vhdr.stem}_tones.vhdr"
    if not injected.is_file():
        write_tone_injected_recording(
            run.raw_vhdr,
            injected,
            tones=tones,
            excluded_channels=config.matched.non_eeg_channels,
        )
    return RunSpec(
        participant=run.participant,
        run_index=run.run_index,
        raw_vhdr=injected,
        protocol_json=run.protocol_json,
        mean_framewise_displacement=run.mean_framewise_displacement,
        motion_stratum=run.motion_stratum,
        marker_block=run.marker_block,
    )


def _complete_arms(
    attempts: Sequence[ArmAttempt],
) -> dict[str, dict[str, ArmOutput]]:
    """Keep the arms whose two passes both produced a recording."""
    passes: dict[str, dict[str, ArmOutput]] = {}
    for attempt in attempts:
        if attempt.output is not None:
            passes.setdefault(attempt.arm, {})[attempt.probe] = attempt.output
    return {arm: found for arm, found in passes.items() if set(found) == set(PASSES)}


def _window(
    complete: dict[str, dict[str, ArmOutput]],
    repetition_time_seconds: float,
    config: BenchmarkConfig,
) -> CommonWindow:
    """Take the span every complete arm can be measured over."""
    native_rate = next(iter(complete.values()))[CLEAN].sampling_rate
    return common_window(
        {
            f"{arm}/{name}": output.volume_count
            for arm, passes in complete.items()
            for name, output in passes.items()
        },
        repetition_time_seconds=repetition_time_seconds,
        sampling_rate=config.output.sampling_rate_hz,
        guard=guard_volumes(
            sampling_rate=native_rate,
            output_sampling_rate=config.output.sampling_rate_hz,
            lowpass_hz=config.output.lowpass_hz,
            repetition_time_seconds=repetition_time_seconds,
        ),
    )


def _on_grid(
    output: ArmOutput, window: CommonWindow, config: BenchmarkConfig
) -> tuple[np.ndarray, list[str]]:
    """Filter, decimate, and crop one arm's output onto the shared grid.

    The recording is cut to its own first volume before it is decimated, so
    that every arm's decimated grid begins on the same acquisition rather than
    on whatever sample its file happens to start at. Decimating first would
    leave each arm on the phase of its own file origin: this project's arm
    trims its output to the first volume and the others emit the whole
    recording, so their grids would sit up to four recorded samples apart, and
    a first volume that is not a whole number of decimated samples from the
    origin could not be indexed onto the grid at all.

    Cutting before filtering puts the filter's edge on the cut, which the
    measured span already drops a whole guard volume to avoid.
    """
    raw = mne.io.read_raw_brainvision(
        output.corrected_vhdr, preload=True, verbose="error"
    )
    from_first_volume = raw.get_data()[:, output.first_volume_sample :]
    decimated = to_output_rate(
        from_first_volume,
        sampling_rate=output.sampling_rate,
        output_sampling_rate=config.output.sampling_rate_hz,
        lowpass_hz=config.output.lowpass_hz,
    )
    return crop(decimated, first_volume_sample=0, window=window), raw.ch_names


def _probe_on_grid(
    run: RunSpec,
    window: CommonWindow,
    *,
    config: BenchmarkConfig,
    tones: tuple[Tone, ...],
) -> np.ndarray:
    """Put the injected waveform through the same filter and crop as the arms.

    Synthesised on the recorded grid and harmonised, rather than rebuilt at the
    output rate, so that its phase lands where the arms' outputs put it without
    anyone having to track the offset by hand.
    """
    raw = mne.io.read_raw_brainvision(run.raw_vhdr, preload=False, verbose="error")
    native_rate = float(raw.info["sfreq"])
    waveform = synthesize_tones(
        tones, sample_count=raw.n_times, sampling_rate=native_rate
    )
    decimated = to_output_rate(
        waveform[np.newaxis, :],
        sampling_rate=native_rate,
        output_sampling_rate=config.output.sampling_rate_hz,
        lowpass_hz=config.output.lowpass_hz,
    )
    return crop(decimated, first_volume_sample=0, window=window)[0]


def _block_count(window: CommonWindow, block_seconds: float) -> int:
    """Count the complete residual blocks the measured span holds."""
    return int(window.sample_count // round(block_seconds * window.sampling_rate))


def _confounds(run: RunSpec, config: BenchmarkConfig) -> Path:
    """Locate the motion estimates recorded against this recording's BOLD run."""
    stem = f"{run.participant}_task-thermalactive_run-{run.run_index:02d}"
    return (
        config.paths.confounds_root
        / run.participant
        / "func"
        / f"{stem}_desc-confounds_timeseries.tsv"
    )


def completed_runs(path: Path) -> frozenset[tuple[str, int]]:
    """Name the recordings already written to an outcome log.

    A cohort-sized run does not fit in one sitting, and the correction pipeline
    refuses to write over an existing output, so resuming means skipping whole
    recordings rather than stepping back into one.
    """
    if not path.is_file():
        return frozenset()
    done: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                done.add((row["participant"], row["run_index"]))
    return frozenset(done)


def write_outcomes(attempts: Sequence[ArmAttempt], run: RunSpec, path: Path) -> None:
    """Append one line per attempt, so a failure is as recorded as a success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for attempt in attempts:
            handle.write(
                json.dumps(
                    {
                        "participant": run.participant,
                        "run_index": run.run_index,
                        "motion_stratum": run.motion_stratum,
                        "arm": attempt.arm,
                        "probe": attempt.probe,
                        "status": "ok" if attempt.succeeded else "failed",
                        "error": attempt.error,
                        "wall_clock_seconds": (
                            attempt.cost.wall_clock_seconds if attempt.cost else None
                        ),
                        "peak_memory_bytes": (
                            attempt.cost.peak_memory_bytes if attempt.cost else None
                        ),
                        "exit_code": attempt.cost.exit_code if attempt.cost else None,
                    }
                )
                + "\n"
            )
