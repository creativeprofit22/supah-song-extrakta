# Research record — 2026-09-06

The `/steroids` corpus was used for downloader integration, separation-model inference, overlap-add reconstruction, and mastering. Repositories were indexed under the user's instruction to use `/steroids` for everything needed. Comparisons below concern source code and task fit, not controlled listening benchmarks.

## References and decisions

### Download: yt-dlp

- Repository: https://github.com/yt-dlp/yt-dlp
- Read via `/steroids`: `yt_dlp/__init__.py`, lines 970–1025, at `bbc809a1161d3bfca51fa36f59dda35556ee85a0`.
- Runtime version: `2026.8.19`, verified against PyPI and the installed CLI.
- Decision: use the established CLI through an argument list; do not implement YouTube extraction.
- Local boundary: canonicalize to an eleven-character video ID; HTTPS YouTube hosts only; disable user config and plugins; do not load cookies; one video per call; 1080p ceiling and 1 GiB per-download limit.
- Limit: `--max-filesize` is per downloaded resource, not an aggregate storage quota. YouTube availability and anti-bot behavior can change.

### Candidate considered: python-audio-separator

- Repository: https://github.com/nomadkaraoke/python-audio-separator
- Read via `/steroids`: `audio_separator/separator/separator.py` and `audio_separator/remote/deploy_modal.py` at `bf1164aa0f1ee1d1d0ef0f09b315f7659fc06bab`.
- Compared its `Separator.load_model` / `separate` integration and vocal/instrumental presets.
- Decision: do not install an additional general-purpose separation wrapper for a single cinematic model. Ordinary vocal/instrumental extraction risks removing singing together with dialogue; it is the wrong objective for this request.

### Chosen: Cinematic Bandit v2

- Inference source: https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/0e5f1159fc5ea87fc13b957584e178b4977e5dd3
- Read via `/steroids`: `models/bandit_v2/bandit.py` and `utils/model_utils.py`, particularly the model constructor/forward pass and windowed `demix` function.
- Model listing: https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json
- Published configuration: https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/config_dnr_bandit_v2_mus64.yaml
- Checkpoint: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/checkpoint-multi_fixed.ckpt
- Downloaded size: **149,133,378 bytes**.
- Observed SHA-256: `20bcd513dc7eb0541dd045909a4e7dff8dab474cc2efba4904101c76524aee85`.
- This digest was calculated from the initial public HTTPS download; the release did not supply a signed digest. It guards subsequent byte changes, not initial publisher identity independently of HTTPS.

The model targets **speech, music, and effects**, which fits the trailer task better than karaoke separation. This is a task-fit rationale, not proof that it retains every sung word.

The local adapter reuses the upstream model class unmodified. It uses the published 384,000-sample windows (8 seconds at 48 kHz), four-way overlap, edge fades and overlap-add averaging. Batch size is reduced to one stereo window to fit the GTX 1080; inference uses full precision rather than mixed precision on this older GPU. Outer reflection padding protects the start/end. The final short window uses zero padding. All original-length frames are preserved.

The constructor intentionally retains the config's default `fs=44100` for band definitions even though waveform input is 48 kHz; changing it independently would diverge from the published model configuration. Strict checkpoint loading succeeded with all keys matched.

The upstream inference script uses unrestricted checkpoint deserialization and training-oriented configuration loading. The local adapter does **not** invoke that script. It supplies the fixed constructor arguments directly and uses `torch.load(..., weights_only=True)`. No Python-tagged YAML is loaded. Source revision and tracked modifications are checked before import; weights are size/hash checked.

Simplification: the adapter accumulates stems in RAM with a ten-minute input ceiling. This is smaller than integrating the upstream training/validation stack but is not a streaming solution for full films. Setup retains the upstream Git checkout and its MIT license, including Roman Solovyev's copyright notice, under `.cache/msst`. Model-weight and media rights are separate from that code license.

### Mastering: FFmpeg, informed by ffmpeg-normalize

- Reference implementation: https://github.com/slhck/ffmpeg-normalize/blob/0809f68ab5cb3117f9f1fbf624bca5f51f11e5e8/src/ffmpeg_normalize/_streams.py
- Read via `/steroids`: `get_second_pass_opts_ebu`, lines 730–758.
- Verified available options with the installed `ffmpeg -hide_banner -h filter=loudnorm`.
- Decision: use native FFmpeg with measured first-pass values, not install another wrapper or normalize blindly in one pass.
- The same cleanup filters run before both passes. Finite measurement checks reject silent/broken inputs. Finished WAV and MP3 are measured separately, including lossy-encoding overshoot.
- Conservative target: −16 LUFS, −1.5 dBTP; no aggressive spectral reconstruction or widening without listening evidence.

### GPU environment

- Official installation reference: https://pytorch.org/get-started/previous-versions/
- Verified official PyTorch 2.7.1 / torchaudio 2.7.1 CUDA 11.8 commands before installation.
- CUDA tensor execution and strict model loading were exercised on the local GTX 1080.
- This compatibility selection is not a declaration that the old dependency versions are free of vulnerabilities.

## Runtime evidence

Completed on the supplied URL:

- Downloaded and merged the 1080p video with its Opus audio.
- Decoded **14,223,987 stereo frames at 48 kHz** (296.333063 seconds).
- Ran all **155 overlapping inference windows**, approximately **143 seconds** in the separation loop; exported music, speech and effects.
- Mastered the music to 24-bit WAV and 320 kbps MP3.
- Measured finished WAV: **−15.96 LUFS / −1.50 dBTP**.
- Measured finished MP3: **−15.96 LUFS / −1.44 dBTP**.
- Checked every WAV block for finite samples and checked exact frame count, sample rate and stereo layout against the extraction.
- Confirmed invalid protocols/host lookalikes/playlist URLs and NaN trim ranges are rejected without creating their intended outputs.
- Confirmed existing stem directories and output files are refused rather than overwritten.
- Revalidated model hash, pinned checkout and dependency consistency (`uv pip check`: 56 compatible packages).

The CLI help and the mastering command were exercised directly. Download/extraction used the same underlying commands/functions; separation was invoked through its module function. No new automated test framework was introduced into this previously empty project; validation was performed with targeted runtime assertions.

## Music-clarity diagnosis — 2026-09-07

Both existing `outputs/listening_0313-0338_*` folders contain identical 25-second,
1,200,000-frame stereo/48 kHz excerpts. Sample-by-sample comparison proved that
the original files equal `outputs/audio/trailer.wav` at [193, 218) seconds, and
the isolated files equal `outputs/stems/music.wav` at that same range (maximum
error zero). They are **not** excerpts of the mastered WAV. The user confirmed
that the previously supplied isolated sample is the reference for the reported
muffling. This localizes the reported problem before mastering; it is listening
evidence from the user, not an automated clarity measurement.

Affected range: **3:13–3:38**. Provisional control: **3:40–4:05**. The control was
selected automatically because the user does not remember a clean interval:
among 25-second windows starting every five seconds from 60 to 270, excluding
overlap with the affected sample and music mean-square energy <= 1e-5, it has
the lowest (speech + effects) / music mean-square energy ratio, **0.01272817**.
This is a low-interference proxy, **not proof of clean sound**; clean-control
listening acceptance remains unresolved. No onset trim is inferred.

Baseline retained: pinned Cinematic Bandit v2, full precision, batch one,
384,000-frame windows, 96,000-frame hop, reflection borders, waveform averaging.
Mastering stays unchanged: 25 Hz high-pass, 50 ms/500 ms endpoint fades,
two-pass -16 LUFS / -1.5 dBTP normalization. Existing exports are preserved.

### Experiment grounding and implementation

Re-inspected `bigshifts_wrapper` with Steroids search/show in
`ZFTurbo/Music-Source-Separation-Training`, revision
`0e5f1159fc5ea87fc13b957584e178b4977e5dd3`, `utils/model_utils.py:17–70`.
It shifts the mixture, calls inference, restores coordinates and averages.
Our adapter uses the same alignment principle but **does not circularly splice
song endpoints**. Channel-reversal TTA is omitted: independently processed
stereo channels make that a poor use of inference work here. No model changes,
training, resource-validation changes, downloads, new repositories or packages.

`separation.py` now shares only model loading and grid inference. The default
still uses the original padding/window/weight order. The opt-in excerpt path
loads one model and computes grids with zero and 48,000 additional left-padding
frames, cropping each back to source coordinates. Context begins on a full-source
two-second hop boundary. This preserves the baseline grid instead of silently
moving it when a requested excerpt begins at an odd second.

- Affected context: **184–226 seconds**, cropped to **193–218**; 27 + 28 windows.
- Control context: **212–253 seconds**, cropped to **220–245**; 27 + 27 windows.
- Roughly 59/58 seconds in the two inference loops respectively, plus model load.
- Both ran on CUDA, float32, batch one. CPU accumulation is excerpt-bounded;
  only the cropped music is retained between passes. Default input ceiling remains
  ten minutes; experiment excerpt ceiling is sixty seconds.
- Exports: `grid_music.wav`, `offset_music.wav`, `average_music.wav`, each exactly
  **1,200,000 stereo frames at 48 kHz**, finite, with source hash/settings manifest.

Comparisons reuse FFmpeg extraction and duration helpers. Installed SoundFile
0.14.0 `read` source/signature and FFmpeg `loudnorm` filter help were inspected.
FFmpeg loudness output is analyzed and discarded; saved comparisons use only
NumPy constant multiplication. A runtime probe exposed trailing FFmpeg text
after the JSON; parsing was fixed with `JSONDecoder.raw_decode`, matching the
existing mastering parser's approach. The same probe then passed.

Shared loudness is constrained by measured true peaks and +6 dB maximum boost,
with a -2 dBTP ceiling. Clips below -60 dBFS RMS or with unmeasurable integrated
loudness are excluded and not boosted. Speech/effects share a separately
documented reference gain (0 dB in these packs). No spectral or dynamic
processing is added, and no residual speech/effects is mixed back.

### Listening packs and measurements

Absolute locations (all existing exports were preserved):

- `E:\Projects\stranger\outputs\clarity_affected_baseline`
- `E:\Projects\stranger\outputs\clarity_control_baseline`
- `E:\Projects\stranger\outputs\clarity_affected_experiment`
- `E:\Projects\stranger\outputs\clarity_control_experiment`
- `E:\Projects\stranger\outputs\clarity_affected_candidates`
- `E:\Projects\stranger\outputs\clarity_control_candidates`

Each comparison has a listening guide and manifest with absolute input/output
paths, source-relative frame ranges, levels and gains. The main candidates
measured **-16.00 to -15.99 LUFS**. Selected comparison values:

| Passage / file | Applied gain (dB) | Actual LUFS | Actual dBTP |
| --- | ---: | ---: | ---: |
| Affected raw | +2.61 | -16.00 | -4.86 |
| Affected existing master | +1.86 | -16.00 | -3.28 |
| Affected offset | +2.63 | -16.00 | -4.85 |
| Affected average | +2.63 | -15.99 | -4.84 |
| Control raw | +0.52 | -15.99 | -2.91 |
| Control offset | +0.53 | -16.00 | -2.90 |
| Control average | +0.53 | -16.00 | -2.90 |

These are level measurements, **not a clarity score**. The tool catalog provided
no trustworthy audio-listening assessor. The user chose **“off set, but not
perfect either.”** This supports selecting the offset candidate for the affected
passage only. It does not establish perfect isolation, an artifact-free final
join, or approval of the provisional control candidate. The average was not
selected. No full-length re-separation or further denoising was performed.

### Accepted interval and final verification

Final folder: `E:\Projects\stranger\outputs\final\clarity-offset`.
Only source frames **[9,264,000, 10,464,000)** were patched. Linear 4,800-frame
(100 ms) crossfades lie *inside* the approved interval; the outermost samples
remain baseline values. The raw stem uses the offset at unity gain. The
24-bit listening WAV uses **+0.77 dB constant gain** on that candidate to match
the existing master excerpt, constrained by peak headroom. It is **not** a
full-track normalization pass and does not apply a second separator.

| Final file | Actual LUFS | Actual dBTP |
| --- | ---: | ---: |
| `music-offset.wav` (raw, float32) | -17.19 | -1.82 |
| `song-offset.wav` (24-bit listening copy) | -15.96 | -1.50 |
| `song-offset.mp3` (320 kbps listening copy) | -15.96 | -1.44 |

Runtime checks passed:

- Deterministic 101,904-frame stereo GPU input, before/after helper extraction:
  **zero sample difference for all three default stems**.
- Both contextual baseline-grid music excerpts match the original full-track
  baseline **exactly**, proving the retained source grid for these ranges.
- Identity-model overlap-add probes at 48,000 / 101,904 / 480,017 frames with
  both offsets reproduced input within 1e-7 absolute / 1e-6 relative tolerance.
  This tests coordinate restoration independently of the real model's behavior.
- Averaged candidates equal `(grid + offset) * 0.5` sample-for-sample. Every
  comparison export verified exact extraction coordinates, finite stereo/48 kHz
  samples, expected frame count, constant-gain values and true-peak headroom.
- Synthetic comparison probes covered silence, very quiet references, fixed gain,
  and invalid NaN/negative/reversed/too-long/out-of-bounds/too-short ranges.
  Invalid ranges created no output directory. Occupied outputs were refused.
- Mismatched experiment source-range metadata was rejected before output creation.
- Finish checks candidate against its auditioned gain-adjusted comparison and
  baselines against the archived excerpts. The source SHA-256 is revalidated.
- Both final WAVs contain **14,223,987 stereo/48 kHz frames**, finite. All samples
  before/after the approved interval are **identical to their baseline WAVs**.
  Joined region matches the calculated crossfade within 24-bit quantization.
- Adjacent-sample jumps at the two outer joins exactly equal the old baseline.
  At the inner fade boundaries the listening WAV steps were approximately
  0.00236 and 0.01344; measurements are recorded in `finish.json`. This is not
  a listening certification of inaudibility.
- MP3 decoded to the same **14,223,987 stereo/48 kHz frames**, all finite; lossy
  re-encoding means the MP3 is not sample-identical outside the patch.
- CLI routing was exercised for normal separation, both GPU experiments,
  comparisons, and finishing. Missing `--approved` and occupied finish output
  were refused. No new test framework or dependency was introduced.

The reported 58–60-second onset was not confirmed, so **nothing was trimmed**.
The plan's onset trim remains deferred, not guessed. No additional checkpoint
was installed. Remaining muffling is unresolved; a different cinematic
checkpoint would require separate compatibility, singing, license and pinned
resource research before any installation. Reproduction commands are in README.

## Not verified

- No automated listening review or final joined-track listening assessment; the user's partial excerpt preference is recorded above. No claim of perfect dialogue removal, singing preservation, or studio fidelity.
- No song-title identification or manually selected song boundaries; the full music timeline is retained.
- No source-separation SDR benchmark: clean reference stems do not exist for this trailer here.
- No third-party security certification, dependency CVE sweep, malicious-media fuzzing, or public-service deployment assessment.
- No CPU performance run, other-platform installation test, or commercial-use rights determination.
