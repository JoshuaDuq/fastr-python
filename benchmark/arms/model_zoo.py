"""Find, fetch, and describe FACETpy's pretrained correction models.

The models are not in the published package. They live in the FACETpy
repository as Git LFS objects, and the code that runs them is only on its main
branch, so both come from a clone rather than from an install. Each export is
downloaded over plain HTTPS and checked against the SHA-256 its pointer
declares, which makes the fetch verifiable without a git-lfs installation.

Every model here was trained by its author on synthetic data, not on this
cohort. They are out of distribution on a real recording from a different
scanner, montage, and sampling rate, and the screening result is a measurement
of how far that generalises rather than of how well the architectures work.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RAW_URL = "https://github.com/H0mire/FACETpy/raw/main/{path}"
EXPORT_ROOT = "artifacts/exports/masterthesis/phase_1"
PROVENANCE_ROOT = "masterthesis_guide/experiments/phase_1"
POINTER_PATTERN = re.compile(r"^oid sha256:([0-9a-f]{64})$\n^size (\d+)$", re.M)


class ModelZooError(RuntimeError):
    """Raised when a model cannot be found, fetched, or described."""


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """What one checkpoint was actually trained with."""

    chunk_size_samples: int
    output_type: str
    trigger_aligned: bool


@dataclass(frozen=True, slots=True)
class ModelExport:
    """One pretrained model, and the settings it was trained under."""

    name: str
    repository_path: str
    sha256: str
    size_bytes: int
    chunk_size_samples: int
    output_type: str
    trigger_aligned: bool


def discover_exports(source_root: Path) -> tuple[ModelExport, ...]:
    """Describe every phase-one model a FACETpy clone declares.

    A CPU export is preferred where the author published one: the others are
    traced for CUDA and will not load on a machine without it.
    """
    export_root = source_root / EXPORT_ROOT
    if not export_root.is_dir():
        raise ModelZooError(f"no FACETpy model exports under {export_root}")
    exports: list[ModelExport] = []
    for model_directory in sorted(export_root.iterdir()):
        pointer = _preferred_export(model_directory)
        if pointer is None:
            continue
        digest, size = _read_pointer(pointer)
        if size == 0:
            continue
        training = _training_settings(source_root, model_directory.name)
        exports.append(
            ModelExport(
                name=model_directory.name,
                repository_path=str(pointer.relative_to(source_root)),
                sha256=digest,
                size_bytes=size,
                chunk_size_samples=training.chunk_size_samples,
                output_type=training.output_type,
                trigger_aligned=training.trigger_aligned,
            )
        )
    if not exports:
        raise ModelZooError(f"no usable model exports under {export_root}")
    return tuple(exports)


def fetch(export: ModelExport, cache: Path) -> Path:
    """Download one model, or return the copy already verified on disk."""
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{export.name}.ts"
    if destination.is_file() and _sha256(destination) == export.sha256:
        return destination
    url = RAW_URL.format(path=export.repository_path)
    partial = destination.with_suffix(".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    digest = _sha256(partial)
    if digest != export.sha256:
        partial.unlink()
        raise ModelZooError(
            f"{export.name} downloaded as {digest}, not the declared {export.sha256}"
        )
    partial.rename(destination)
    return destination


def _preferred_export(model_directory: Path) -> Path | None:
    """Return the export to use for this model, preferring a CPU trace."""
    if not model_directory.is_dir():
        return None
    candidates = sorted(model_directory.glob("*/*.ts"))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.stem.endswith("_cpu"):
            return candidate
    return candidates[0]


def _read_pointer(pointer: Path) -> tuple[str, int]:
    """Read the hash and size a Git LFS pointer declares for its content."""
    match = POINTER_PATTERN.search(pointer.read_text(encoding="utf-8"))
    if match is None:
        raise ModelZooError(f"{pointer} is not a Git LFS pointer")
    return match.group(1), int(match.group(2))


def _training_settings(source_root: Path, model: str) -> TrainingSettings:
    """Read the chunk length and target a model was actually trained with.

    Taken from the run's own resolved configuration rather than from the
    architecture blueprints, which describe the published papers and not these
    checkpoints.
    """
    resolved = (
        source_root
        / PROVENANCE_ROOT
        / f"holdout_{model}"
        / "provenance"
        / "facet_train_config.resolved.json"
    )
    if not resolved.is_file():
        raise ModelZooError(f"{model} has no resolved training configuration")
    training = json.loads(resolved.read_text(encoding="utf-8"))["training"]
    return TrainingSettings(
        chunk_size_samples=int(training["chunk_size"]),
        output_type=str(training["target_type"]),
        trigger_aligned=bool(training["trigger_aligned"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
