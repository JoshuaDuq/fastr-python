import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_project_metadata_identifies_the_research_software() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert metadata["authors"] == [{"name": "Joshua Duquette"}]
    assert "Intended Audience :: Science/Research" in metadata["classifiers"]
    assert metadata["urls"]["Repository"] == (
        "https://github.com/JoshuaDuq/eeg-fmri-fastr"
    )


def test_citation_file_contains_version_license_and_required_references() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["title"] == "EEG-fMRI FASTR"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "GPL-2.0-only"
    reference_titles = {
        reference["title"] for reference in citation["references"]
    }
    assert (
        "Removal of FMRI environment artifacts from EEG data using optimal basis sets"
        in reference_titles
    )
    assert "MEG and EEG data analysis with MNE-Python" in reference_titles
    assert any(
        title.startswith("The brain imaging data structure")
        for title in reference_titles
    )


def test_quality_workflow_uses_the_documented_read_only_quality_gates() -> None:
    workflow = (ROOT / ".github/workflows/quality.yml").read_text()

    for required_text in (
        "push:",
        "pull_request:",
        "python-version-file: .python-version",
        "uv sync --locked",
        "uv run pytest",
        "uv run ruff check src tests validation",
        "uv build",
        "permissions:\n  contents: read",
    ):
        assert required_text in workflow


def test_shipped_configuration_examples_are_loadable() -> None:
    from eegfmri_fastr.config import load_config

    volume = load_config(ROOT / "examples/configuration.yml")
    sliced = load_config(ROOT / "examples/configuration-slice.yml")
    assert volume.timing.marker_kind == "volume"
    assert sliced.timing.marker_kind == "slice"
