"""YAML for comparing uncorrected EEG with FASTR-corrected recordings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from ..config import ConfigurationError


@dataclass(frozen=True, slots=True)
class ComparePaths:
    """Store the input and output roots used for a comparison."""

    uncorrected_root: Path
    fastr_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class PlotSettings:
    """Store PSD plotting limits for a comparison."""

    psd_max_hz: float


@dataclass(frozen=True, slots=True)
class NamingConfig:
    """How file and folder names identify a recording, a subject, and a run.

    Every export convention names its stages differently, so the suffixes that
    identify them are declared rather than assumed. The defaults recognise only
    a ``_fastr`` corrected suffix and BIDS ``sub-`` directories; an uncorrected
    suffix has to be named before anything will pair.
    """

    corrected_suffixes: tuple[str, ...] = ("_fastr",)
    uncorrected_suffixes: tuple[str, ...] = ()
    subject_directory_prefix: str = "sub-"
    first_run_prefixes: tuple[str, ...] = ()
    run_index_token: str = "run"


@dataclass(frozen=True, slots=True)
class CompareConfig:
    """Store validated paths, subject selection, and naming rules."""

    paths: ComparePaths
    plot: PlotSettings
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    naming: NamingConfig = field(default_factory=NamingConfig)


_TOP = frozenset({"paths", "plot", "subjects", "naming"})
_REQUIRED_TOP = frozenset({"paths", "plot", "subjects"})
_PATH_KEYS = frozenset({"uncorrected_root", "fastr_root", "output_root"})
_PLOT_KEYS = frozenset({"psd_max_hz"})
_SUBJECT_KEYS = frozenset({"include", "exclude"})
_NAMING_KEYS = frozenset(
    {
        "corrected_suffixes",
        "uncorrected_suffixes",
        "subject_directory_prefix",
        "first_run_prefixes",
        "run_index_token",
    }
)


def load_compare_config(path: str | Path) -> CompareConfig:
    """Load and validate a comparison configuration from YAML."""
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
    for key in sorted(_REQUIRED_TOP):
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
            psd_max_hz=float(cast(float, plot["psd_max_hz"])),
        ),
        include=_string_list(subjects, "include"),
        exclude=_string_list(subjects, "exclude"),
        naming=_naming_config(document),
    )


def _naming_config(document: Mapping[str, object]) -> NamingConfig:
    """Read the optional naming section, defaulting each absent field."""
    if "naming" not in document:
        return NamingConfig()
    values = _mapping(document["naming"], "naming")
    unknown = sorted(str(key) for key in values if key not in _NAMING_KEYS)
    if unknown:
        raise ConfigurationError(f"unknown field(s) in naming: {', '.join(unknown)}")
    defaults = NamingConfig()
    return NamingConfig(
        corrected_suffixes=_optional_string_list(
            values,
            "corrected_suffixes",
            defaults.corrected_suffixes,
        ),
        uncorrected_suffixes=_optional_string_list(
            values,
            "uncorrected_suffixes",
            defaults.uncorrected_suffixes,
        ),
        subject_directory_prefix=_optional_string(
            values,
            "subject_directory_prefix",
            defaults.subject_directory_prefix,
        ),
        first_run_prefixes=_optional_string_list(
            values,
            "first_run_prefixes",
            defaults.first_run_prefixes,
        ),
        run_index_token=_optional_string(
            values,
            "run_index_token",
            defaults.run_index_token,
        ),
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
        raise ConfigurationError(f"unknown field(s) in {field}: {', '.join(unknown)}")
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


def _optional_string(
    values: Mapping[str, object],
    name: str,
    default: str,
) -> str:
    if name not in values:
        return default
    value = values[name]
    if not isinstance(value, str):
        raise ConfigurationError(f"naming.{name} must be a string")
    return value


def _optional_string_list(
    values: Mapping[str, object],
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if name not in values:
        return default
    entries = _string_list(values, name)
    if any(not entry for entry in entries):
        raise ConfigurationError(f"naming.{name} must not contain empty strings")
    return entries
