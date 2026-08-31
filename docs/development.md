# Development

## Supported environment

The supported and tested interpreter is Python 3.12. Runtime dependencies are
declared as compatible ranges in `pyproject.toml`; `uv.lock` records the exact
environment used for reproducible checks.

Set up the environment:

```text
uv sync
```

## Repository layout

- `src/fastr_python/`: production package and low-level numerical modules;
- `tests/`: deterministic unit, integration, compatibility, and contract tests;
- `examples/`: loadable configuration examples;
- `validation/`: explicit reference and comparison runners;
- `docs/`: user, scientific, architecture, development, and reference guides;
- `CITATION.cff`: machine-readable software citation; and
- `.github/workflows/quality.yml`: pull-request and push quality gate.

## Quality checks

Run these commands before opening a review:

```text
uv sync
uv run pytest
uv run ruff check src tests validation
git diff --check
uv build
```

`.github/workflows/quality.yml` runs the same checks in a locked Python
environment for pushes and pull requests. The local commands and `uv.lock`
remain the source of truth for contributors outside GitHub.

## Adding or changing configuration

Treat `config.py`, both shipped examples, the tests, and
[`configuration.md`](configuration.md) as one change. Add a failing contract
test first, state units and defaults, preserve fail-fast validation, and update
the interaction table. Do not add a fallback or silently reinterpret an
ambiguous input.

## Adding validation evidence

Keep validation evidence reproducible and scoped. Record the input description,
configuration, software version, hashes, metrics, and comparison conditions.
Label project-generated measurements with their dataset scope. Keep private
recordings and generated outputs outside the tracked repository.

## Data and generated files

Do not commit subject recordings, BIDS sidecars containing private metadata,
generated BrainVision outputs, plots, provenance from private runs, or local
virtual environments. Use temporary or ignored directories for these files.

## Commit and review expectations

Keep commits focused and use descriptive names. Preserve valid-input behavior,
output schemas, CLI formats, and expected errors during refactors. Review
numerical changes with tests and, where relevant, a reference or signal-transfer
comparison. Update [references](references.md) when a scientific or software
claim is introduced.
