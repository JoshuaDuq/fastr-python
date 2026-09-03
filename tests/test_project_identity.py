import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_public_package_uses_fastr_python_import_name() -> None:
    import fastr_python

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert metadata["name"] == "fastr-python"
    assert fastr_python.__version__ == metadata["version"]


def test_readme_introduces_the_python_fmrib_fastr_implementation() -> None:
    readme = (ROOT / "README.md").read_text()

    assert readme.startswith("# FASTR-Python\n\nFASTR-Python is the Python version")
    assert "https://github.com/sccn/fMRIb" in readme


def test_quality_and_validation_utilities_have_distinct_packages() -> None:
    from fastr_python.quality import residuals
    from fastr_python.validation import diagnostics, metrics, simulation

    assert callable(residuals.block_residual_uv)
    assert callable(diagnostics.estimate_slice_period_candidates)
    assert callable(metrics.tone_transfer)
    assert callable(simulation.simulate_gradient_artifact)
