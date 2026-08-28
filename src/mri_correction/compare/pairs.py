"""Pair uncorrected pulse-marked files with FASTR-corrected recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CompareConfig

_UNCORRECTED_SUFFIXES = (
    "_first_to_last_volume_scanner_artifact_with_pulse_markers",
    "_scanner_artifact_with_pulse_markers",
)
_FASTR_SUFFIX = "_fastr"


@dataclass(frozen=True, slots=True)
class RecordingPair:
    bids_id: str
    idx_run: int
    key: str
    uncorrected_vhdr: Path
    fastr_vhdr: Path


def recording_key(stem: str) -> str:
    """Strip FASTR/step1 suffixes so both sides share one identifier."""
    if stem.endswith(_FASTR_SUFFIX):
        stem = stem[: -len(_FASTR_SUFFIX)]
    for suffix in _UNCORRECTED_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _list_vhdrs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.vhdr")
        if not path.name.startswith("._") and path.stat().st_size > 100
    )


def _run_sort_key(path: Path):
    name = path.name
    if name.startswith("BaselineEEG"):
        return (0, 0, name)
    marker = "run"
    lower = name.lower()
    index = lower.find(marker)
    if index >= 0:
        digits = []
        for char in name[index + len(marker) :]:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            return (1, int("".join(digits)), name)
    return (1, 99, name)


def index_uncorrected(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in _list_vhdrs(root):
        indexed[recording_key(path.stem)] = path
    if indexed:
        return indexed
    # Nested subject folders, if the uncorrected root is not flat.
    for sub_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for path in _list_vhdrs(sub_dir):
            indexed[recording_key(path.stem)] = path
    return indexed


def pair_recordings(config: CompareConfig) -> list[RecordingPair]:
    if not config.paths.fastr_root.is_dir():
        raise FileNotFoundError(
            f"fastr_root is not a directory: {config.paths.fastr_root}"
        )
    uncorrected = index_uncorrected(config.paths.uncorrected_root)
    pairs: list[RecordingPair] = []
    for sub_dir in sorted(config.paths.fastr_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        if sub_dir.name.startswith("._"):
            continue
        bids_id = sub_dir.name
        if (
            config.include
            and bids_id not in config.include
            and bids_id.replace("-", "") not in config.include
        ):
            continue
        if bids_id in config.exclude:
            continue
        vhdrs = sorted(_list_vhdrs(sub_dir), key=_run_sort_key)
        for idx, fastr_vhdr in enumerate(vhdrs, start=1):
            key = recording_key(fastr_vhdr.stem)
            uncorrected_vhdr = uncorrected.get(key)
            if uncorrected_vhdr is None:
                print(f"skip {bids_id} {fastr_vhdr.name}: no uncorrected match")
                continue
            pairs.append(
                RecordingPair(
                    bids_id=bids_id,
                    idx_run=idx,
                    key=key,
                    uncorrected_vhdr=uncorrected_vhdr,
                    fastr_vhdr=fastr_vhdr,
                )
            )
    return pairs
