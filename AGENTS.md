# Repository Guidelines

## Project Structure & Module Organization
Core logic sits in `pdf2zh/`: `pdf2zh/pdf2zh.py` drives the CLI, `converter.py` extracts layouts, `translator.py` wires service adapters, and GUI/backends share the package. Docker recipes and helper scripts live in `script/` plus the top-level Docker files. Tests sit in `test/` with fixture PDFs in `test/file/`, and docs with images stay in `docs/`. Runtime settings flow through `pdf2zh/config.py`, persisting to `~/.config/PDFMathTranslate/config.json`; leave that user-owned state untracked.

## Build, Test, and Development Commands
- `uv pip install --group dev` pulls the dev toolchain (`pytest`, `black`, `flake8`, `pre-commit`).
- `uv pip install --group local` installs optional local-model dependencies (e.g., `nvidia-riva-client`).
- `pdf2zh sample.pdf -o out/` exercises the CLI and writes translated assets to `out/`.
- `pytest test/` runs the automated test suite; add `-k <keyword>` for targeted runs.
- `python -m build` validates packaging before releasing, and `docker compose up` spins up the demo stack.

## Local Environment Setup
Provision a venv via `UV_CACHE_DIR=.uv-cache uv venv .venv`, install runtime deps with `UV_PROJECT_ENVIRONMENT=.venv uv pip install --editable .`, then pull tooling through `UV_PROJECT_ENVIRONMENT=.venv uv pip install --group dev`. Smoke-check with `UV_PROJECT_ENVIRONMENT=.venv uv run pytest test`.

## Coding Style & Naming Conventions
Use 4-space indentation, format with `black` (88-character target), and lint with `flake8` (120-character ceiling; ignores E203/E261/E501/W503/E741). Prefer snake_case for functions and variables, PascalCase for classes, and module names that match their responsibility (`config`, `converter`, `gui`). When adding translators, mirror existing adapter patterns and document new environment keys inline.

## Testing Guidelines
Tests run under `pytest`, wrapping `unittest` suites. Add coverage to `test/test_<module>.py`, name classes `Test<Subject>`, and mock translators or network clients so runs stay offline. Extend `test/file/` only when fixtures are essential. Execute `pytest test/` (or a targeted `-k` run) before pushing, and record CLI smoke results when workflow-critical behavior changes.

## Commit & Pull Request Guidelines
Keep commits short, present-tense statements with optional scope prefixes (e.g., `doc: clarify install guide`, `fix: guard config lookup`). Group related changes and reference issues via `#<id>` when it helps. Pull requests need a concise summary, key-change bullets, proof of tests or CLI runs, and refreshed docs or screenshots when user flows change. Highlight new configuration requirements so reviewers can reproduce the setup.

## Configuration & Secrets
Never commit API keys or generated configs. Populate translation credentials through environment variables; `ConfigManager` picks them up and persists sanitized copies in `~/.config/PDFMathTranslate/config.json`. If you must reference that file while debugging, share redacted snippets or a schema and describe cleanup steps for reviewers.
