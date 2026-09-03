"""Strict scalar and mapping readers for YAML configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Literal

from .models import ConfigurationError


def _section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
    *,
    optional_keys: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    values = _require_mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    for field_name in sorted(expected_keys - optional_keys):
        if field_name not in values:
            raise ConfigurationError(f"missing required field: {name}.{field_name}")
    return values


def _optional_section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, object]:
    if name not in root:
        return {}
    values = _require_mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    return values


def _optional_finite_number(
    values: Mapping[str, object],
    name: str,
    *,
    default: float,
    minimum: float,
    inclusive: bool = False,
    maximum: float | None = None,
) -> float:
    if name not in values:
        return default
    return _finite_number(
        values,
        name,
        minimum=minimum,
        inclusive=inclusive,
        maximum=maximum,
    )


def _optional_positive_integer(
    values: Mapping[str, object],
    name: str,
    *,
    default: int,
) -> int:
    if name not in values or values[name] is None:
        return default
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _optional_obs_rank(
    values: Mapping[str, object],
    name: str,
    *,
    default: int | Literal["auto"],
) -> int | Literal["auto"]:
    if name not in values:
        return default
    value = values[name]
    if value == "auto":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer or 'auto'")
    return value


def _optional_positive_number_or_none(
    values: Mapping[str, object],
    name: str,
) -> float | None:
    if name not in values or values[name] is None:
        return None
    return _finite_number(values, name, minimum=0.0)


def _optional_integer_or_none(
    values: Mapping[str, object],
    name: str,
) -> int | None:
    if name not in values or values[name] is None:
        return None
    return _integer_value(values, name, minimum=1)


def _require_mapping(value: object, field_path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_path} must be a mapping")
    return value


def _reject_unknown_keys(
    values: Mapping[str, object],
    expected_keys: frozenset[str],
    field_path: str,
) -> None:
    unknown = sorted(set(values) - expected_keys)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field_path}: {', '.join(unknown)}"
        )


def _required_value(values: Mapping[str, object], name: str) -> object:
    if name not in values:
        raise ConfigurationError(f"missing required field: {name}")
    return values[name]


def _string_value(values: Mapping[str, object], name: str) -> str:
    value = _required_value(values, name)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
    if any(character in value for character in ("\0", "\n", "\r")):
        raise ConfigurationError(f"{name} contains an invalid character")
    return value


def _path_value(
    values: Mapping[str, object],
    name: str,
    base_directory: Path,
) -> Path:
    value = _string_value(values, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _boolean_value(values: Mapping[str, object], name: str) -> bool:
    value = _required_value(values, name)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _integer_value(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = _required_value(values, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_number(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    inclusive: bool = False,
    maximum: float | None = None,
) -> float:
    value = _required_value(values, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    number = float(value)
    too_small = number < minimum if inclusive else number <= minimum
    too_large = maximum is not None and number > maximum
    if not math.isfinite(number) or too_small or too_large:
        if too_large:
            raise ConfigurationError(f"{name} must be less than or equal to {maximum}")
        bound = "greater than or equal to" if inclusive else "greater than"
        raise ConfigurationError(f"{name} must be {bound} {minimum}")
    return number
