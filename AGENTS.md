# Coding agent guide

- Python 3.12 CLI; run commands from the repository root. See README.md for setup and audio constraints.
- Install for CI/build work: `uv sync --locked`. On this existing workstation, first set `UV_PROJECT_ENVIRONMENT=.cache/ci-venv` (PowerShell: `$env:UV_PROJECT_ENVIRONMENT = ".cache/ci-venv"`) to preserve the working CUDA `.venv`.
- Build: `uv build` using the setuptools backend in `pyproject.toml`.
- Dependencies: edit `requirements.txt`, then run `uv lock`; preserve the historical Windows CUDA snapshot.
- Portable verification: `.venv/Scripts/python.exe -B scripts/verify_audio_jobs.py` (synthetic audio, installed dependencies and FFmpeg/ffprobe; no model/GPU/private fixtures). On other platforms use the configured Python 3.12 interpreter.
- Safe CLI checks: `python -B -m songtool job run --help` and `python -B -m songtool cleanup-preview --help`. Only when the ignored harness and original fixtures exist, also run `.venv/Scripts/python.exe -B .cache/firered-aed/verify_cleanup_workflow.py`.
- No lint command is configured. CI wiring lives in the workflow below; the verification contract is the portable harness before build. Synthetic checks and packaging do not verify perceptual audio quality or confer listening approval.
- CI lives in `.github/workflows/ci.yml` and must stay green.
- Never commit with `--no-verify`.
- Never commit credentials, environments, caches, models, or generated media. Keep exclusive output creation and preserve existing user files.
- Keep audio thresholds, the fixed cleanup recipe, and manual listening approval unchanged. Do not rerun the already-completed full-song trial or replace the preferred master without explicit approval.

## Audio-job agent procedure

- Read `docs/audio-workflow.md`. This is an explicit private-local job workflow, not an autonomous optimizer or hostile-upload service. Require an explicit directory; never infer a song from newest files.
- Read `job status JOB --json` first; identify exact current and parent IDs/hashes, feedback, protected maps and failed receipts. Raw working baseline, canonical source conversion, candidate and older preferred master are distinct. State intended operation/device/budget before acting.
- Default to CPU, two numerical threads and one owned operation; CUDA is opt-in per job run. The GPU lease controls only this tool in the same workspace, not other applications/workspaces. Preserve the CUDA environment; do not install or fetch models without authorization.
- Run one approved operation once. Whole-song attempts do not require endless microclip review; clips are optional uncertain hints. Keep original feedback wording and exact half-open 48 kHz frame scope. Acceptance protects mapped samples, never implies whole-song approval.
- Respect fingerprints and failures: reuse completed results, never evade a failed/rejected recipe by renaming outputs. Only operational failures allow a recorded reasoned retry; thresholds stay fixed. Wanted-vocal loss/reverse-like artifacts need new capability, not speculative denoise or speech-stem reinsertion.
- Report actual outcome (`opened_existing`, `analyzed_only`, `rendered_new`, `reused_result`, or failure), execution, technical checks and scoped listening judgment separately. Say `no_supported_repair` when appropriate; partial files are not successful candidates.
- Do not auto-adopt legacy media. With explicit permission use `workflow.adopt_failed_evidence` and the guide's validated mapping procedure for immutable retry-blocking evidence. Historical trailer fixed recipes stay separately labelled unsupported operations, never generic-denoise equivalents. Missing/mismatched evidence remains a blocker and the identical failed recipe stays manually barred; never fabricate measurements, native receipts or whole-song approval.
- Treat snapshots as bounded immutable local history. Never edit/prune committed state or overwrite media. Copy/restore to fresh destinations and verify status/hashes; same-disk copies are not disaster backups. No off-device backup is configured.
