# Development & Contribution Guide

This guide details the development workflows, static analysis standards, testing protocols, and engineering principles governing the **FASTR-Python** codebase.

---

## 1. Environment Setup

FASTR-Python requires **Python 3.12**. Development environments are strictly managed and locked using [`uv`](https://docs.astral.sh/uv/).

To initialize the locked development environment:

```bash
uv sync --locked
```

This installs runtime dependencies along with developer tools (`pytest`, `ruff`, `mypy`, `types-pyyaml`).

---

## 2. Repository Architecture

```text
fastr-python/
├── src/fastr_python/            Production package source code
│   ├── api.py                   Stable high-level API entrypoint
│   ├── fastr.py                 Stable low-level array API entrypoint
│   ├── cli.py                   Command-line interface (`fastr-python`)
│   ├── config/                  Configuration schemas, validation, and loading
│   ├── correction/              Pure numerical signal processing algorithms
│   ├── io/                      BrainVision reader, writer, and marker management
│   ├── pipeline/                Pipeline orchestration, windowing, and provenance
│   ├── quality/                 Harmonic and residual quality-control analytics
│   ├── validation/              Simulation models, metrics, and reference runners
│   └── compare/                 Cohort folder-level comparison tooling
├── tests/                       Comprehensive test suites matching source structure
├── examples/                    Loadable YAML configuration examples
├── validation/                  Standalone reference runners and comparison scripts
├── docs/                        Technical and scientific documentation
├── .github/workflows/           Continuous integration (CI) quality gates
├── pyproject.toml               PEP 621 package and dependency declarations
├── uv.lock                      Cryptographically locked dependency graph
└── CITATION.cff                 Machine-readable citation metadata
```

---

## 3. Read-Only Quality Gates

Before committing changes or opening a pull request, run the complete local quality gate:

```bash
# 1. Verify dependencies are locked and synced
uv sync --locked

# 2. Fast linting and style checking
uv run ruff check src tests validation

# 3. Code formatting compliance
uv run ruff format --check src tests validation

# 4. Strict static type analysis
uv run mypy

# 5. Execute full test suite
uv run pytest

# 6. Check for whitespace issues or merge conflict markers
git diff --check

# 7. Verify source distribution and wheel packaging
uv build
```

These exact commands are executed by the GitHub Actions CI workflow (`.github/workflows/quality.yml`) on every push and pull request.

---

## 4. Engineering & Scientific Principles

Contributors must adhere to the project's core engineering principles:

1. **Clarity & Scientific Rigor over Cleverness**: Algorithms should clearly reflect their mathematical formulations. Avoid opaque optimizations that hinder peer audit.
2. **Single Responsibility**: Each module, class, and function must do exactly one thing. Functions should be small, focused, and free of flag arguments.
3. **Fail-Fast Error Handling**: Validate assumptions (types, shapes, ranges, mutual exclusivity) at public boundaries. Raise specific exceptions (`ConfigurationError`, `ValueError`) early; never swallow exceptions or implement hidden silent fallbacks.
4. **Behavioral Invariance**: Refactorings must preserve numerical outputs and side effects bit-for-bit.
5. **No Hidden State**: Avoid global singletons, mutable module state, or hidden side effects. Prefer pure functions operating on explicit immutable arguments.
6. **Domain-Aware Naming**: Functions are verbs; variables are nouns. Embed physical units in variable names where ambiguity can arise (`*_hz`, `*_seconds`, `*_uv`).

---

## 5. Modifying Configuration

When adding, renaming, or modifying a configuration parameter:

1. **Co-update Schema**: Update dataclass models and validation logic in `fastr_python.config`.
2. **Synchronize Examples**: Update both [`examples/configuration.yml`](../examples/configuration.yml) and [`examples/configuration-slice.yml`](../examples/configuration-slice.yml).
3. **Update Documentation**: Update the parameter specification tables and mutual exclusivity rules in [`docs/configuration.md`](configuration.md).
4. **Write Contract Tests**: Add unit tests in `tests/pipeline/test_config.py` asserting both valid parsing and expected error rejection for invalid inputs.

---

## 6. Data Hygiene & Confidentiality

- **Never commit human subject EEG or fMRI data**: Real patient or volunteer recordings, private BIDS metadata, and scan notes must never be committed to Git.
- **Use Synthetic Generators for Tests**: Unit tests must use programmatic synthetic signals (`fastr_python.validation.simulation`) or existing minimal fixtures.
- **Keep Artifacts Local**: Store scratch logs, temporary recordings, and generated MAT files in temporary directories excluded by `.gitignore`.

---

## 7. Review & Release Guidelines

- Commits should be focused, atomic, and descriptive.
- All 720+ automated unit and regression tests must pass without warnings.
- Documentation links must be verified (`uv run pytest tests/contracts/test_documentation.py`).
- Update [`CITATION.cff`](../CITATION.cff) and version declarations in `pyproject.toml` when preparing releases.
