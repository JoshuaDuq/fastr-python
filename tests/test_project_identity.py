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
