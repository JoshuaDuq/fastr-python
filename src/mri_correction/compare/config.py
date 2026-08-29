"""YAML for comparing uncorrected EEG with FASTR-corrected recordings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import ConfigurationError


@dataclass(frozen=True, slots=True)
class ComparePaths:
    uncorrected_root: Path
    fastr_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class PlotSettings:
    psd_max_hz: float


@dataclass(frozen=True, slots=True)
class CompareConfig:
    paths: ComparePaths
    plot: PlotSettings
    include: tuple[str, ...]
    exclude: tuple[str, ...]


_TOP = frozenset({"paths", "plot", "subjects"})
_PATH_KEYS = frozenset({"uncorrected_root", "fastr_root", "output_root"})
_PLOT_KEYS = frozenset({"psd_max_hz"})
_SUBJECT_KEYS = frozenset({"include", "exclude"})


def load_compare_config(path: str | Path) -> CompareConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"configuration file does not exist: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML in configuration: {config_path}"
        ) from error
    if not isinstance(document, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    unknown = sorted(str(key) for key in document if key not in _TOP)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in configuration: {', '.join(unknown)}"
        )
    for key in sorted(_TOP):
        if key not in document:
            raise ConfigurationError(f"missing required field: {key}")

    base = config_path.parent
    paths = _mapping(document["paths"], "paths")
    _require_keys(paths, _PATH_KEYS, "paths")
    plot = _mapping(document["plot"], "plot")
    _require_keys(plot, _PLOT_KEYS, "plot")
    subjects = _mapping(document["subjects"], "subjects")
    _require_keys(subjects, _SUBJECT_KEYS, "subjects")
    return CompareConfig(
        paths=ComparePaths(
            uncorrected_root=_path(paths, "uncorrected_root", base),
            fastr_root=_path(paths, "fastr_root", base),
            output_root=_path(paths, "output_root", base),
        ),
        plot=PlotSettings(
            psd_max_hz=float(plot["psd_max_hz"]),
        ),
        include=_string_list(subjects, "include"),
        exclude=_string_list(subjects, "exclude"),
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _require_keys(
    values: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    unknown = sorted(str(key) for key in values if key not in expected)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field}: {', '.join(unknown)}"
        )
    for key in sorted(expected):
        if key not in values:
            raise ConfigurationError(f"missing required field: {field}.{key}")


def _string(values: Mapping[str, object], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value


def _path(values: Mapping[str, object], name: str, base: Path) -> Path:
    path = Path(_string(values, name)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _string_list(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = values[name]
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a list of strings")
    return tuple(str(item) for item in value)
