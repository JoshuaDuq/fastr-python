# Development

## Supported environment

The supported interpreter is Python 3.12. Runtime ranges are in
`pyproject.toml`; `uv.lock` records the tested environment.

Set up the environment:

```text
uv sync
```

## Repository layout

- `src/fastr_python/`: domain-organized production and validation packages;
- `tests/`: matching correction, I/O, pipeline, quality, validation, comparison,
  and public-contract suites;
- `examples/`: loadable configuration examples;
- `validation/`: explicit reference and comparison runners;
- `docs/`: user, scientific, architecture, development, and reference guides;
- `CITATION.cff`: machine-readable software citation; and
- `.github/workflows/quality.yml`: pull-request and push quality gate.

## Quality checks

Run the quality checks:

```text
uv sync
uv run ruff check src tests validation
uv run ruff format --check src tests validation
uv run mypy
uv run pytest
git diff --check
uv build
```

`.github/workflows/quality.yml` runs the same checks in a locked environment for
pushes and pull requests.

## Adding or changing configuration

Update the relevant module in `fastr_python.config`, both examples, the tests, and
[`configuration.md`](configuration.md) together. Add a failing contract test
first, document units and defaults, preserve fail-fast validation, and update
the interaction table. Do not add a fallback or silently reinterpret ambiguous
input.

## Adding validation evidence

Keep validation evidence reproducible and scoped. Record the inputs,
configuration, software version, hashes, metrics, and comparison conditions.
Label project-generated measurements and keep private recordings and outputs
outside the repository.

## Data and generated files

Do not commit subject recordings, private BIDS metadata, generated BrainVision
outputs, plots, private provenance, or virtual environments. Use temporary or
ignored directories.

## Commit and review expectations

Keep commits focused and descriptive. Preserve valid-input behavior, output
schemas, CLI formats, and expected errors. Test numerical changes and compare
with a reference or signal-transfer measure when relevant. Update
[references](references.md) for new scientific or software claims.
