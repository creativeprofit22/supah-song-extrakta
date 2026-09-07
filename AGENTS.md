# Coding agent guide

- Python 3.12 CLI; run commands from the repository root. See README.md for setup and audio constraints.
- Install for CI/build work: `uv sync --locked`. On this existing workstation, first set `UV_PROJECT_ENVIRONMENT=.cache/ci-venv` (PowerShell: `$env:UV_PROJECT_ENVIRONMENT = ".cache/ci-venv"`) to preserve the working CUDA `.venv`.
- Build: `uv build` using the setuptools backend in `pyproject.toml`.
- Dependencies: edit `requirements.txt`, then run `uv lock`; preserve the historical Windows CUDA snapshot.
- Local verification, only when the ignored harness and original fixtures exist: `.venv/Scripts/python.exe -B .cache/firered-aed/verify_cleanup_workflow.py` and `.venv/Scripts/python.exe -B -m songtool cleanup-preview --help`.
- No portable test suite or lint command is configured. CI builds only; do not claim audio verification from a green build.
- CI lives in `.github/workflows/ci.yml` and must stay green.
- Never commit with `--no-verify`.
- Never commit credentials, environments, caches, models, or generated media. Keep exclusive output creation and preserve existing user files.
- Keep audio thresholds, the fixed cleanup recipe, and manual listening approval unchanged. Do not rerun the already-completed full-song trial or replace the preferred master without explicit approval.
