"""Pair uncorrected recordings with their corrected counterparts by name."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import CompareConfig, NamingConfig

# A BrainVision header shorter than this is a placeholder or a truncated
# export, not a recording worth pairing.
_MINIMUM_HEADER_BYTES = 100
# Recordings whose name carries no run number sort after every numbered run.
_UNNUMBERED_RUN = sys.maxsize


@dataclass(frozen=True, slots=True)
class RecordingPair:
    """Store the two files belonging to one logical recording."""

    bids_id: str
    idx_run: int
    key: str
    uncorrected_vhdr: Path
    fastr_vhdr: Path


def recording_key(stem: str, naming: NamingConfig) -> str:
    """Strip the configured suffixes so both sides share one identifier.

    Corrected and uncorrected exports of one recording differ only by the
    suffix each stage appends, so removing both leaves the recording itself.
    """
    stem = _strip_longest_suffix(stem, naming.corrected_suffixes)
    return _strip_longest_suffix(stem, naming.uncorrected_suffixes)


def index_uncorrected(root: Path, naming: NamingConfig) -> dict[str, Path]:
    """Index uncorrected recordings by key, flat or in subject directories."""
    indexed = {
        recording_key(path.stem, naming): path for path in _list_vhdrs(root)
    }
    if indexed:
        return indexed
    for subject_dir in _subject_directories(root, naming):
        for path in _list_vhdrs(subject_dir):
            indexed[recording_key(path.stem, naming)] = path
    return indexed


def pair_recordings(config: CompareConfig) -> list[RecordingPair]:
    """Match every corrected recording to the uncorrected one it came from."""
    if not config.paths.fastr_root.is_dir():
        raise FileNotFoundError(
            f"fastr_root is not a directory: {config.paths.fastr_root}"
        )
    naming = config.naming
    uncorrected = index_uncorrected(config.paths.uncorrected_root, naming)
    pairs: list[RecordingPair] = []
    for subject_dir in _subject_directories(config.paths.fastr_root, naming):
        bids_id = subject_dir.name
        if not _is_selected(bids_id, config):
            continue
        corrected = sorted(
            _list_vhdrs(subject_dir),
            key=lambda path: _run_sort_key(path, naming),
        )
        for idx_run, fastr_vhdr in enumerate(corrected, start=1):
            key = recording_key(fastr_vhdr.stem, naming)
            uncorrected_vhdr = uncorrected.get(key)
            if uncorrected_vhdr is None:
                print(f"skip {bids_id} {fastr_vhdr.name}: no uncorrected match")
                continue
            pairs.append(
                RecordingPair(
                    bids_id=bids_id,
                    idx_run=idx_run,
                    key=key,
                    uncorrected_vhdr=uncorrected_vhdr,
                    fastr_vhdr=fastr_vhdr,
                )
            )
    return pairs


def _strip_longest_suffix(stem: str, suffixes: Sequence[str]) -> str:
    """Remove the longest matching suffix, so a more specific one wins."""
    matches = [suffix for suffix in suffixes if stem.endswith(suffix)]
    if not matches:
        return stem
    return stem[: -len(max(matches, key=len))]


def _subject_directories(root: Path, naming: NamingConfig) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith("._")
        and path.name.startswith(naming.subject_directory_prefix)
    )


def _list_vhdrs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.glob("*.vhdr")
        if not path.name.startswith("._")
        and path.stat().st_size > _MINIMUM_HEADER_BYTES
    )


def _is_selected(bids_id: str, config: CompareConfig) -> bool:
    if bids_id in config.exclude:
        return False
    if not config.include:
        return True
    return bids_id in config.include or bids_id.replace("-", "") in config.include


def _run_sort_key(path: Path, naming: NamingConfig) -> tuple[int, int, str]:
    """Order one subject's recordings: declared leading runs, then by number."""
    name = path.name
    if naming.first_run_prefixes and name.startswith(naming.first_run_prefixes):
        return (0, 0, name)
    return (1, _run_index(name, naming.run_index_token), name)


def _run_index(name: str, token: str) -> int:
    """Read the run number written after ``token``, case-insensitively."""
    if not token:
        return _UNNUMBERED_RUN
    position = name.lower().find(token.lower())
    if position < 0:
        return _UNNUMBERED_RUN
    digits = ""
    for character in name[position + len(token) :]:
        if not character.isdigit():
            break
        digits += character
    return int(digits) if digits else _UNNUMBERED_RUN
