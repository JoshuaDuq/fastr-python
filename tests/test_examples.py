"""The shipped examples are the documentation, so they have to stay loadable.

A configuration example that no longer parses is worse than no example: it is
the first thing a new user copies. These tests fail the moment a field is
renamed or removed without the example following.
"""

from pathlib import Path

import pytest
import yaml

from eegfmri_fastr.compare.config import load_compare_config
from eegfmri_fastr.config import load_config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_the_correction_example_loads() -> None:
    config = load_config(EXAMPLES / "configuration.yml")

    assert config.processing.method == "acquisition_group_fastr"
    assert config.input.fmri_metadata is not None
    assert config.timing.marker_kind == "volume"
    assert config.trim.mode == "first_to_last_volume"


def test_the_compare_example_loads() -> None:
    config = load_compare_config(EXAMPLES / "compare.yaml")

    assert config.naming.corrected_suffixes == ("_fastr",)
    assert config.naming.uncorrected_suffixes
    assert config.naming.subject_directory_prefix == "sub-"


def test_the_correction_example_documents_every_processing_field() -> None:
    """A setting absent from the example is a setting nobody will discover."""
    document = yaml.safe_load(
        (EXAMPLES / "configuration.yml").read_text(encoding="utf-8")
    )
    from eegfmri_fastr.config import _PROCESSING_KEYS

    assert set(document["processing"]) == set(_PROCESSING_KEYS)


def test_the_correction_example_documents_every_quality_control_field() -> None:
    document = yaml.safe_load(
        (EXAMPLES / "configuration.yml").read_text(encoding="utf-8")
    )
    from eegfmri_fastr.config import _QUALITY_CONTROL_KEYS

    assert set(document["quality_control"]) == set(_QUALITY_CONTROL_KEYS)


def test_the_compare_example_documents_every_naming_field() -> None:
    document = yaml.safe_load(
        (EXAMPLES / "compare.yaml").read_text(encoding="utf-8")
    )
    from eegfmri_fastr.compare.config import _NAMING_KEYS

    assert set(document["naming"]) == set(_NAMING_KEYS)


@pytest.mark.parametrize(
    "commented_block",
    [
        "acquisition:",
        "marker_kind: slice",
        "groups_per_volume: 18",
        "expected_repetition_time_seconds: 0.9",
    ],
)
def test_the_correction_example_shows_the_alternatives(
    commented_block: str,
) -> None:
    """Both timing routes have to be visible in the file a user copies."""
    text = (EXAMPLES / "configuration.yml").read_text(encoding="utf-8")

    assert commented_block in text
