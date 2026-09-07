# Trailer song extraction

Local, modular tools to download one YouTube video, decode its audio, separate its music from dialogue/effects, and create a listening master. No cloud audio upload, paid service, or account cookies.

## Latest listening copy — offset excerpt approved

**Start here:** `E:\Projects\stranger\outputs\final\clarity-offset\song-offset.mp3`

Lossless copy: `E:\Projects\stranger\outputs\final\clarity-offset\song-offset.wav`

You preferred the offset version, **“but not perfect either.”** Only **3:13–3:38**
was replaced, with 100 ms crossfades inside that interval. The rest of each WAV
is sample-identical to its baseline; the MP3 is a new lossy encoding. No claim
that the remaining muffling or leakage is fixed. The final joined track has not
yet received a listening review.

The 4:56.333 timeline remains intact: the reported 58–60-second song onset was
not confirmed, so no opening or interior music was cut. Original files below
are untouched. `music-offset.wav` in the same folder is the patched raw stem;
`finish.json` records the joins, gain and measurements. The finished listening
WAV measures **−15.96 LUFS / −1.50 dBTP**; MP3 **−15.96 LUFS / −1.44 dBTP**.

## Original files — preserved

Source: https://www.youtube.com/watch?v=avpTgTNadh4

| File | Purpose |
| --- | --- |
| `outputs/final/song-master.mp3` | Convenient listening copy, 320 kbps |
| `outputs/final/song-master.wav` | 24-bit / 48 kHz stereo listening master |
| `outputs/final/song-master.json` | Before/after loudness and true-peak measurements |
| `outputs/stems/music.wav` | Unmastered music estimate, 32-bit float |
| `outputs/stems/speech.wav` | Separated speech estimate; useful for checking misplaced singing |
| `outputs/stems/sfx.wav` | Separated effects estimate |
| `outputs/audio/trailer.wav` | Original decoded soundtrack, before separation |
| `outputs/source/avpTgTNadh4.mkv` | Downloaded 1080p video with its original compressed audio |

The music export retains the full **4:56.33** timeline. It has not been manually trimmed to a particular song section or identified by title. That avoids guessing which musical passage you meant or discarding a quiet intro.

**Quality limits:** this is an AI estimate, not the original studio recording. Dialogue/effects may bleed through; singing can leak into the speech stem. No automated listening assessment was available. Your later preference for the offset excerpt is limited listening evidence, not verification of perfect isolation or vocal preservation. WAV conversion does not restore information lost in YouTube compression. The raw soundtrack and all three stems remain available.

## Setup — Windows

Already installed in this project. For a fresh setup, run these commands from the project root in PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 --link-mode copy
uv pip install --python .venv/Scripts/python.exe -r requirements-windows.lock.txt --link-mode copy
.venv/Scripts/python.exe -m songtool setup-model
```

Prerequisites on `PATH`: **uv, Git, FFmpeg, ffprobe, and Node.js**. FFmpeg must include `loudnorm` and `libmp3lame`. This environment uses an NVIDIA GTX 1080; CUDA inference was exercised successfully. The CUDA 11.8 wheels are deliberately retained for this older GPU. Downloads include roughly 2.6 GiB for PyTorch and 149 MB for the model; allow several additional GiB for installation and media.

`requirements.txt` lists direct dependencies. `requirements-windows.lock.txt` records all 56 installed versions; it is a version snapshot, not a hash-verified supply-chain audit. Other platforms/GPU generations are not tested. `--device cpu` is available, but will be substantially slower.

## CI and package build

GitHub CI uses Python 3.12 on Ubuntu, `uv sync --locked`, then `uv build` through the
setuptools backend in `pyproject.toml`. Direct dependencies come from `requirements.txt`;
`uv.lock` records the portable resolution. Run `uv lock` after changing those requirements.
The build produces a wheel and source archive locally; CI does not upload artifacts.
No portable test or lint script is configured, so CI is **build-only**, not audio validation.
The ignored local cleanup harness requires the original project's private audio/receipts and FFmpeg.
Checkout v7.0.1 and setup-uv v10.0.1 were verified against their official READMEs/releases
on 2026-09-07 and pinned by commit in `.github/workflows/ci.yml`.

**Keep the working CUDA environment intact:** do not run `uv sync` against this project's existing
`.venv`. For a separate build environment in PowerShell, use:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".cache/ci-venv"
uv sync --locked
uv build
```

The Windows CUDA setup above remains the listening workflow; CI never downloads media/models or
runs a full-song trial. Weekly Dependabot updates cover actions, uv locking and pip requirements;
the historical Windows version snapshot is excluded from pip updates.

## Independent stages

```powershell
# 1. Download (preserves existing downloads)
.venv/Scripts/python.exe -m songtool download "https://www.youtube.com/watch?v=avpTgTNadh4"

# 2. Extract the soundtrack to a new WAV
.venv/Scripts/python.exe -m songtool extract outputs/source/avpTgTNadh4.mkv outputs/audio/trailer.wav

# 3. Separate music, speech and effects into a NEW directory
.venv/Scripts/python.exe -m songtool separate outputs/audio/trailer.wav outputs/stems

# 4. Master the music; also creates matching MP3 and JSON files
.venv/Scripts/python.exe -m songtool master outputs/stems/music.wav outputs/final/song-master.wav
```

These outputs already exist here. **Choose new output names when rerunning**; extraction, separation and mastering refuse to overwrite existing files. A failed FFmpeg export may leave a partial file; do not treat it as successful. A successful master prints its paths and writes the JSON report last. Original media is never edited in place.

To choose a section after listening, reuse the extraction stage on the music stem. Times are **seconds**, with `--end` relative to the input, not relative to `--start`:

```powershell
# Example only: 1:00 through 2:00; not an identified song boundary
.venv/Scripts/python.exe -m songtool extract outputs/stems/music.wav outputs/audio/selected-section.wav --start 60 --end 120
.venv/Scripts/python.exe -m songtool master outputs/audio/selected-section.wav outputs/final/selected-section.wav
```

Separation accepts stereo 48 kHz WAVs up to ten minutes. Longer inputs need trimming first. This deliberate ceiling bounds RAM usage; it is not a film-length streaming engine.

## What mastering does

1. Removes subsonic rumble below 25 Hz.
2. Adds a 50 ms opening fade and 500 ms closing fade.
3. Measures the cleaned audio, then performs two-pass loudness normalization targeting **−16 LUFS / −1.5 dBTP**.
4. Uses linear gain where the peak ceiling allows it; otherwise FFmpeg applies dynamic normalization.
5. Exports 24-bit WAV and 320 kbps MP3, then independently measures both.

No speculative aggressive EQ, stereo widening, generative reconstruction, or repeated denoising. Those can damage already-separated audio without careful listening.

For this run, measured WAV loudness is **−15.96 LUFS**, with **−1.50 dBTP** true peak. MP3 measured **−15.96 LUFS / −1.44 dBTP**. Loudness range changed from 13.0 to 8.9 LU: the master is more even, not dynamically identical to the raw stem. In the JSON, `wav.input_*` and `mp3.input_*` describe the actual finished files; `output_*` is the analysis filter's hypothetical further processing, not another saved master.

## Reproduce the clarity comparison

No additional packages, training, checkpoint downloads, EQ boost, or extra
denoising pass. Run from the project root. **These directories already exist;
choose fresh names to rerun**, updating subsequent commands to match.

```powershell
# Original / raw / master plus speech and effects references
.venv/Scripts/python.exe -m songtool compare outputs/clarity_affected_baseline --start 193 --end 218
.venv/Scripts/python.exe -m songtool compare outputs/clarity_control_baseline --start 220 --end 245

# Opt-in experiment: two source-aligned grids, one loaded model per excerpt
.venv/Scripts/python.exe -m songtool separate outputs/audio/trailer.wav outputs/clarity_affected_experiment --window-offset --start 193 --end 218
.venv/Scripts/python.exe -m songtool separate outputs/audio/trailer.wav outputs/clarity_control_experiment --window-offset --start 220 --end 245

# Include the baseline-grid, offset and averaged candidates at matched loudness
.venv/Scripts/python.exe -m songtool compare outputs/clarity_affected_candidates --start 193 --end 218 --experiment outputs/clarity_affected_experiment
.venv/Scripts/python.exe -m songtool compare outputs/clarity_control_candidates --start 220 --end 245 --experiment outputs/clarity_control_experiment

# Only after listening preference: patch the approved offset, not the whole song
.venv/Scripts/python.exe -m songtool finish-offset outputs/clarity_affected_candidates outputs/final/clarity-offset --approved
```

Comparison folders open automatically on Windows; `compare --no-open` disables
that. Each contains an absolute-path `LISTEN.md`, `manifest.json`, and untouched
excerpt copies under `unadjusted`. Folder without its completion JSON means an
incomplete export. No existing folder or output is replaced.

- Affected comparison: `E:\Projects\stranger\outputs\clarity_affected_candidates`
- Provisional control: `E:\Projects\stranger\outputs\clarity_control_candidates`
- `raw_music.wav`: original isolation; `offset_music.wav`: preferred experiment;
  `average_music.wav`: equal-weight blend, **not selected**.
- `grid_music.wav`: contextual baseline, verified sample-identical to raw music.
- `original.wav` / `mastered_music.wav`: original soundtrack / existing master.
- Speech/effects are leakage references at documented shared gain, **not** matched
  to the music loudness. Do not mix them back into the song.

Keep player volume fixed and disable player normalization/enhancements. Judge
music detail, singing, percussion attacks, and unwanted dialogue/fighting.
The control was chosen automatically for low estimated speech/effects energy,
not certified clean by listening. Its candidate is **not** applied to the final.

`compare` requires 1–60-second ranges and aligned full-length stereo/48 kHz
inputs; override defaults with `--source`, `--stems`, and `--master`. Start/end
are source seconds rounded to sample frames. Only constant gain is applied:
shared attainable target at most −16 LUFS, −2 dBTP ceiling, at most +6 dB boost.
Silent/very quiet clips (RMS below −60 dBFS or unmeasurable integrated loudness)
are reported and never boosted; they are excluded from loudness matching.
All main clips in these packs measured within 0.01 LU of −16 LUFS.

`--window-offset` leaves normal `separate` behavior unchanged. It uses a
one-second grid offset, at least eight seconds of real surrounding context
where available, and crops back to the requested frames. It saves both estimates
and their average separately; no endpoint wrapping or assumption averaging wins.
Both 25-second GPU experiments completed in about a minute of inference each,
plus model initialization; no CPU experiment was run.

`finish-offset --approved` requires the completed matching comparison, a 24-bit
baseline master, and sources up to ten minutes. It checks source provenance and
that the candidate still matches the auditioned fixed-gain file. The raw patch
uses unity gain; the listening patch uses peak-safe constant gain to match the
old master excerpt (+0.77 dB here). No full-track remastering is performed.
Outside the approved interval the WAVs are checked sample-for-sample; the
100 ms crossfades, final peaks and MP3 decode integrity were checked too.
Numeric checks do not establish that the final joins are inaudible.

## Guarded full-song cleanup preview (not a replacement master)

The preferred baseline remains `outputs/final/clarity-offset/song-offset.wav`.
Preference for the cleaned/brightened **193–218s excerpt** is not approval of a full-song treatment.
Run from this project using the existing interpreter, with a destination that does not exist:

```powershell
.venv/Scripts/python.exe -B -m songtool cleanup-preview outputs/full-song-cleanup-trial
```

Approval covers **one full-song trial only**. Do not rerun under a different name after a guard failure
without reviewing the failure and obtaining approval. This command never plays audio or promotes it.
It accepts only an output folder, not a different source or user-configured filters.

- One continuous `afftdn=nr=3:nf=-50:tn=0:tr=0:gs=5` pass starts from the preferred master.
  Installed-executable impulse checks gate 1,200-sample padding/delay removal, including the last frame.
- +1.5dB peaking EQ at 3.5kHz/Q=0.7 is blended only at original **203.5–204.5,
  208.5–209, 211.5–212s**, using shared 100ms raised-cosine transitions wholly inside each window.
  Outside these windows PCM24 samples must match the denoised intermediate encoded identically.
- All 14,223,987 stereo/48kHz frames remain, opening included. No normalization, compression,
  separation, model loading, downloads, dependencies, extra denoising, or GPU analysis.
  The brief sound at 197–198s is not specifically targeted. Full-song filter context can differ
  from the isolated listened excerpt near its edges; equality to that preview is not promised.

### Outputs and stop conditions

| Output | Meaning |
| --- | --- |
| `candidate.wav` | Single full-song PCM24 candidate; eligible for listening only after verification |
| `metrics.csv` | Half-second timeline RMS, peaks and spectral ratios; original/stems are context only |
| `report.json` | Authoritative completion commit: final status, hashes, versions, timings, guards and comparisons |
| `report.md` | Provisional human-readable summary; never proof of completion by itself |
| `report.pending.json` | Retained staging diagnostics, possibly partial; never a completion receipt |
| `review-reel.wav`, `review-reel.json` | Up to four five-second A/B pairs, mapping every frame, at most 44s |
| `technical/` | Two continuous float64 intermediates and small delay fixtures; not alternate candidates |
| `incomplete.json`, `failed.json` | Initial marker and failure marker; precedence rules below |

Completion is fail-closed:

- Any `failed.json` takes precedence over all reports, even if truncated or a final JSON says success.
  A valid failure receipt supplies the reason; a partial failure marker still disqualifies the run.
- Otherwise, only a complete, valid `report.json` with `status: verified_candidate`,
  `preservation_verified: true` and `listening_approved: false` establishes technical completion.
  A final `failed` status is failure. Missing, partial, malformed or unexpected final JSON is not success.
- Without a qualifying final JSON, the run is incomplete/unverified, regardless of `report.md` or
  `report.pending.json` contents. `incomplete.json` is retained and superseded only by a valid final
  receipt or a failure marker. Even a missing/partial initial marker cannot establish success.

Publication writes and closes Markdown after the required artifacts and preservation checks, then
writes and closes the staging JSON. An exclusive hard link publishes that complete JSON as
`report.json` last; existing files are never replaced, removed or retried. The staging link is retained
as diagnostics and must not be edited (it shares the final file's bytes). Filesystems without hard-link
support fail closed. Report errors print no candidate-success message. Failure reporting is best-effort:
if disk failure or interruption also prevents `failed.json`, absent final JSON still means incomplete.
An interruption after the commit may leave a complete receipt; any failure marker still overrides it.
This is a publication-order guarantee, not a power-loss durability guarantee.

The reel uses baseline **A**, then candidate **B**, unity gain, identical 10ms edge fades,
and 0.5s separators. It includes 197–198s and an approved brightness region; up to two additional
nonoverlapping windows use greatest measured change with stable tie-breaking. No independent
loudness matching, narration, or automatic playback. Keep player volume unchanged.

Preflight validates receipts, listened-preview hashes, full timelines, installed filters and free disk
(about 0.8GB reserved). Streaming hashes and installed-package metadata are checked before/after.
Fresh exclusive outputs reject occupied paths and symlink destinations. Owned subprocesses have
timeouts and are killed/reaped on failure or interruption; originals and partial outputs are retained.
Inputs exceeding the ten-minute ceiling or differing from this recorded baseline are rejected.

Verification requires no full-scale samples and **≤−1dBTP**, overall RMS change **≤0.5dB**, active
half-second changes **≤1dB**, and active five-second difference RMS **below −20dB relative**.
Quiet windows are labelled and checked against an absolute −60dBFS ceiling instead of silence ratios.
EQ bounds, exact untouched regions, finite decoded samples, frames, loudness and bounded real-data
alignment checks must pass. Low-energy correlation is explicitly inconclusive. If the baseline itself
violates headroom, stop before rendering rather than normalize. Any failure stops completion and attempts
an explicit **failed** receipt; partial-write precedence above applies if reporting itself fails.
There is no stronger retry or automatic repair.

`verified_candidate` means technical checks passed, **not better sound or listening approval**.
Residual speech, tonal dullness, transient damage and vocal preservation still require human judgment.
FireRed previously missed the reported sound and is neither a gate nor a full-length scan here.
The one-shot local harness is `.cache/firered-aed/verify_cleanup_workflow.py` (ignored, no test framework).

## Module map

| Module | Responsibility |
| --- | --- |
| `songtool/media.py` | URL validation, downloading, probing, extraction and trims |
| `songtool/resources.py` | Pinned inference checkout, bounded model download, SHA-256 validation |
| `songtool/separation.py` | Bandit model inference and overlap-add reconstruction |
| `songtool/mastering.py` | Two-pass mastering, encoding and measurements |
| `songtool/comparison.py` | Aligned fixed-gain comparisons and approved offset joins |
| `songtool/cleanup.py` | Fixed-recipe, fail-closed full-song preview and bounded diagnostics |
| `songtool/__main__.py` | Thin command-line routing; heavy ML imports only for separation |

To change separation models, replace the separation adapter and its resource definition; downloading and mastering remain independent. There is intentionally no web UI, database, custom neural network, or training system.

## Troubleshooting and boundaries

- **CUDA unavailable:** try `separate ... --device cpu`, or check your NVIDIA driver and the installed wheel versions.
- **Interrupted PyTorch installation:** rerun the same install command; do not recreate an environment containing other work.
- **YouTube blocks the request:** no sign-in/DRM/age-gate bypass is attempted. Use a lawfully obtained local video with `extract` instead.
- **Model mismatch:** stop; do not change `weights_only=True` or remove checksum validation. Inspect the expected release and checkout first.
- **Existing output:** select another name. Inference has no mid-run resume; rerun separation if interrupted before stem export.
- **Missing singing:** compare the speech stem against the music stem. Blindly adding speech back also restores dialogue.
- **Rights:** use material you are allowed to download and process. Separation does not grant redistribution or commercial-use rights to the video, song, or model weights.

Only public HTTPS download sources are used. Subprocesses receive argument lists rather than interpolated shell commands. yt-dlp configuration/plugins are disabled; no browser credentials are read. Model loading uses PyTorch's restricted `weights_only=True`. The checkpoint hash pins the bytes observed on the initial HTTPS download; it is not an independently signed publisher digest. The inference checkout is pinned and checked for tracked changes before import. This is a local tool, not a sandbox for malicious model repositories or a public upload service. Dependency CVE auditing and hostile-media fuzzing were not performed.

## Research and verification

See [docs/RESEARCH.md](docs/RESEARCH.md) for the `/steroids` source comparisons, pinned references, and verification scope.
