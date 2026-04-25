# AGENTS.md

> A concise cheat‑sheet to avoid common pitfalls and speed up development.

## Environment
- Python ≥ 3.10 (project uses 3.10‑3.12; 3.10 is the minimal version).
- **Poetry** is used for dependency and virtual‑environment management.

## Installation
```bash
# Install deps and create the virtual env
poetry install
```
Use `poetry shell` if you prefer to work interactively.

## Running tests
The repository ships a small Makefile.
```bash
make test  # runs pytest with coverage built into pyproject.toml
make cov   # runs tests and outputs an XML coverage report
```
Alternatively, run directly:
```bash
poetry run pytest tests/
```
Coverage is enforced at 90 % (`--cov-fail-under=90`). If it falls below, CI will fail.

## Linting & type checking
```bash
poetry run ruff check .      # lint + format
poetry run mypy src           # strict type check
```
`ruff` also applies isort; it knows that `model_track` is a first‑party package.

## Security audit
```bash
poetry run pip-audit
```
The CI skips `CVE‑2026‑4539`.

## Common pitfalls
- The lock file (`poetry.lock`) may be missing. Run `poetry lock` after a fresh install.
- Tests live in `tests/`; running from the repo root is fine because `pythonpath` is set in `pyproject.toml` to include `src`.

## CI details
- Triggers on pushes or PRs to `main`/`develop` and on new tags `v*`.
- Tests run on Python 3.10‑3.13, lint on the default test job.
- The `publish` job is gated on a `v*` tag and expects the package to build and pass all checks.

## Documentation & workflow
The README contains the full library overview and usage examples.
For contribution guidelines and the mandatory testing checklist, see [CONTRIBUTING.md](CONTRIBUTING.md).
