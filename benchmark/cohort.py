"""Choose the recordings every arm corrects, and freeze that choice.

Selection happens once, before any correction runs, and is written to a
manifest carrying a hash per recording. Every arm then reads the manifest, so
that all of them demonstrably saw the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.motion import mean_framewise_displacement

TRIMMED_DIR_NAME = "original_trimmed_5khz"
UNTRIMMED_DIR_NAME = "original_untrimmed_5khz"
TRIM_SUFFIX = "_first_to_last_volume"
TASK_PREFIX = "ThermalPainEEGFMRI"
RUN_PATTERN = re.compile(r"run(\d+)", re.IGNORECASE)

# Stratum names by how many runs each participant contributes. Anything else
# raises: a benchmark that silently renamed its strata would report motion
# sensitivity against labels nobody chose.
STRATUM_NAMES: dict[int, tuple[str, ...]] = {
    1: ("median",),
    2: ("low", "high"),
    3: ("low", "middle", "high"),
}

# Its filename says run 1, but it was recorded at 11h30 between run 2 (11h20)
# and run 4 (11h40), it carries 514 volume markers like its siblings, and
# sub-0003 has a run-03 BOLD with no EEG delivered against it.
MISNAMED_RUNS = {"ThermalPainEEGFMRI_run1_sub0003_2026-03-23_11h30.23.962": 3}

# Recordings holding more than one acquisition, and the marker block that is
# the BOLD run. sub-0000 run 1 is 570 volumes, a 63.5 s scanner break, then an
# aborted 118-volume block; 570 matches its NIfTI.
MARKER_BLOCKS: dict[str, tuple[int, int]] = {
    "ThermalPainEEGFMRI_run1_sub0000_2026-02-09_11h11.44.706": (0, 570),
}


class CohortError(ValueError):
    """Raised when the cohort on disk cannot supply the requested selection."""


@dataclass(frozen=True, slots=True)
class CohortPaths:
    """Where the recordings, motion estimates, and BOLD sidecars live."""

    source_root: Path
    confounds_root: Path
    protocol_root: Path


@dataclass(frozen=True, slots=True, order=True)
class RunSpec:
    """One recording every arm corrects, and the motion it was recorded with."""

    participant: str
    run_index: int
    raw_vhdr: Path
    protocol_json: Path
    mean_framewise_displacement: float
    motion_stratum: str
    marker_block: tuple[int, int] | None


def select_cohort(
    paths: CohortPaths, *, runs_per_participant: int
) -> tuple[RunSpec, ...]:
    """Take evenly spaced motion strata from every participant's task runs.

    Ranking within a participant holds montage, session, and head constant, so
    the spread that remains is motion rather than everything that differs
    between people.
    """
    names = STRATUM_NAMES.get(runs_per_participant)
    if names is None:
        raise CohortError(
            f"runs_per_participant must be one of {sorted(STRATUM_NAMES)}, "
            f"not {runs_per_participant}"
        )
    selected: list[RunSpec] = []
    for participant_dir in sorted(paths.source_root.glob("sub-0*")):
        ranked = _rank_task_runs_by_motion(paths, participant_dir.name)
        if len(ranked) < runs_per_participant:
            raise CohortError(
                f"{participant_dir.name} has {len(ranked)} task runs, "
                f"fewer than the {runs_per_participant} requested"
            )
        selected.extend(_take_strata(ranked, names))
    return tuple(selected)


def write_manifest(runs: tuple[RunSpec, ...], path: Path) -> None:
    """Freeze the selection, with a hash per recording, before anything runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"runs": [_manifest_row(run) for run in runs]}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> tuple[RunSpec, ...]:
    """Read back a frozen selection without re-reading the cohort."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_run_from_row(row) for row in document["runs"])


def verify_manifest(path: Path) -> None:
    """Confirm every recording still hashes to what the manifest recorded."""
    document = json.loads(path.read_text(encoding="utf-8"))
    for row in document["runs"]:
        raw_vhdr = Path(row["raw_vhdr"])
        current = _manifest_row(_run_from_row(row))
        for field in ("vhdr_sha256", "vmrk_sha256", "eeg_bytes"):
            if current[field] != row[field]:
                raise CohortError(f"{raw_vhdr} changed since the manifest was written")


def _rank_task_runs_by_motion(
    paths: CohortPaths, participant: str
) -> list[tuple[float, int, Path, Path]]:
    """Order a participant's task runs from least to most head motion."""
    trimmed_dir = paths.source_root / participant / "eeg" / TRIMMED_DIR_NAME
    untrimmed_dir = paths.source_root / participant / "eeg" / UNTRIMMED_DIR_NAME
    ranked: list[tuple[float, int, Path, Path]] = []
    for trimmed in sorted(trimmed_dir.glob(f"{TASK_PREFIX}*.vhdr")):
        if trimmed.name.startswith("._"):
            continue
        raw_vhdr = untrimmed_dir / f"{trimmed.stem.removesuffix(TRIM_SUFFIX)}.vhdr"
        if not raw_vhdr.is_file():
            raise CohortError(f"no untrimmed source for {trimmed}")
        run_index = _run_index(raw_vhdr)
        stem = f"{participant}_task-thermalactive_run-{run_index:02d}"
        confounds = (
            paths.confounds_root
            / participant
            / "func"
            / f"{stem}_desc-confounds_timeseries.tsv"
        )
        protocol = paths.protocol_root / participant / "func" / f"{stem}_bold.json"
        if not protocol.is_file():
            raise CohortError(f"no BOLD sidecar for {raw_vhdr.name}: {protocol}")
        ranked.append(
            (mean_framewise_displacement(confounds), run_index, raw_vhdr, protocol)
        )
    ranked.sort()
    return ranked


def _take_strata(
    ranked: list[tuple[float, int, Path, Path]], names: tuple[str, ...]
) -> list[RunSpec]:
    """Pick evenly spaced ranks, so the strata span the observed motion range."""
    last = len(ranked) - 1
    divisor = max(len(names) - 1, 1)
    return [
        _run_spec(ranked[last * position // divisor], stratum=name)
        for position, name in enumerate(names)
    ]


def _run_spec(ranked_run: tuple[float, int, Path, Path], *, stratum: str) -> RunSpec:
    displacement, run_index, raw_vhdr, protocol = ranked_run
    return RunSpec(
        participant=raw_vhdr.parents[2].name,
        run_index=run_index,
        raw_vhdr=raw_vhdr,
        protocol_json=protocol,
        mean_framewise_displacement=displacement,
        motion_stratum=stratum,
        marker_block=MARKER_BLOCKS.get(raw_vhdr.stem),
    )


def _run_index(raw_vhdr: Path) -> int:
    """Return the run this recording belongs to, correcting known mislabels."""
    renamed = MISNAMED_RUNS.get(raw_vhdr.stem)
    if renamed is not None:
        return renamed
    match = RUN_PATTERN.search(raw_vhdr.stem)
    if match is None:
        raise CohortError(f"cannot parse a run number from {raw_vhdr.name}")
    return int(match.group(1))


def _manifest_row(run: RunSpec) -> dict[str, Any]:
    """Describe one selected recording, hashing what identifies it.

    The header and marker streams are hashed because they carry the timing
    every arm depends on. The sample file is recorded by size instead: it runs
    to hundreds of megabytes, and the pipeline already hashes it per run.
    """
    return {
        "participant": run.participant,
        "run_index": run.run_index,
        "raw_vhdr": str(run.raw_vhdr),
        "protocol_json": str(run.protocol_json),
        "mean_framewise_displacement": run.mean_framewise_displacement,
        "motion_stratum": run.motion_stratum,
        "marker_block": list(run.marker_block) if run.marker_block else None,
        "vhdr_sha256": _sha256(run.raw_vhdr),
        "vmrk_sha256": _sha256(run.raw_vhdr.with_suffix(".vmrk")),
        "eeg_bytes": run.raw_vhdr.with_suffix(".eeg").stat().st_size,
    }


def _run_from_row(row: dict[str, Any]) -> RunSpec:
    block = row["marker_block"]
    return RunSpec(
        participant=row["participant"],
        run_index=row["run_index"],
        raw_vhdr=Path(row["raw_vhdr"]),
        protocol_json=Path(row["protocol_json"]),
        mean_framewise_displacement=row["mean_framewise_displacement"],
        motion_stratum=row["motion_stratum"],
        marker_block=None if block is None else (block[0], block[1]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
