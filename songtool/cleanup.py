"""One fixed, guarded experiment; verification is never listening approval."""

from contextlib import contextmanager, ExitStack
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

RATE = 48000
FRAMES = 14223987
FIRST, LAST = 9264000, 10464000
DELAY = 1200
FADE = 4800
WINDOWS = ((9768000, 9816000), (10008000, 10032000), (10152000, 10176000))
DENOISE = "afftdn=nr=3:nf=-50:tn=0:tr=0:gs=5"
EQ = "equalizer=f=3500:t=q:w=0.7:g=1.5:r=f64"
ROOT = Path(__file__).resolve().parent.parent
PREVIEWS = "outputs/firered-aed-gpu-193-218"
PREVIEW_HASHES = {
    "cleanup-preview-aligned.wav": "bea39c1c5b966f5ef509233adcf76fb089521706da70240079956671c98d600e",
    "brightness-preview.wav": "0e356bc6fdb959eed4c1059f4cb9656968e53448a768cb4f9e6a622b70f61216",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment() -> str:
    records = sorted((d.metadata.get("Name", ""), d.version, d.read_text("METADATA") or "",
                      d.read_text("RECORD") or "") for d in importlib.metadata.distributions())
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def execute(argv: list[str], timeout: float = 180) -> str:
    """Own only this process; disk-backed diagnostics, bounded reads, kill/reap on interruption."""
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        size = log.tell()
        require(size <= 1024 * 1024, "External diagnostics exceeded 1 MiB.")
        log.seek(0)
        text = log.read().decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"{Path(argv[0]).name} failed ({process.returncode}): {text[-4000:]}")
        return text


def read_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size <= 65536, "Invalid/oversized receipt.")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), "Receipt must be an object.")
    return result


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audio_info(path: Path, frames: int | None = None, subtype: str | None = None) -> sf._SoundFileInfo:
    require(path.is_file() and path.suffix.lower() == ".wav", f"Not a regular WAV: {path}")
    info = sf.info(path)
    # FFmpeg's listened PCM24 excerpt uses the extensible Microsoft WAV header.
    require(info.format in ("WAV", "WAVEX") and info.samplerate == RATE and info.channels == 2
            and 0 < info.frames <= 600 * RATE, f"Invalid audio timeline: {path}")
    require(frames is None or info.frames == frames, f"Frame count mismatch: {path}")
    require(subtype is None or info.subtype == subtype, f"Wrong encoding: {path}")
    return info


@contextmanager
def run_directory(destination: Path):
    require(not destination.is_symlink() and not destination.exists(), "Output must be a fresh directory.")
    directory = destination.parent.resolve(strict=True) / destination.name
    directory.mkdir(exist_ok=False)
    write_json(directory / "incomplete.json", {"status": "incomplete", "listening_approved": False})
    try:
        yield directory
    except BaseException as error:
        write_json(directory / "failed.json", {"status": "failed", "reason": str(error) or type(error).__name__,
                                                "listening_approved": False})
        raise


def preflight(root: Path = ROOT) -> dict:
    inputs = {
        "baseline": root / "outputs/final/clarity-offset/song-offset.wav",
        "original": root / "outputs/audio/trailer.wav",
        "raw_music": root / "outputs/stems/music.wav",
        "offset_music": root / "outputs/final/clarity-offset/music-offset.wav",
        "listened_cleanup": root / PREVIEWS / "cleanup-preview-aligned.wav",
        "listened_brightness": root / PREVIEWS / "brightness-preview.wav",
        "excerpt": root / PREVIEWS / "preferred-full-quality-193-218.wav",
    }
    inputs = {name: path.resolve(strict=True) for name, path in inputs.items()}
    for name, path in inputs.items():
        audio_info(path, LAST - FIRST if name.startswith("listened_") or name == "excerpt" else FRAMES,
                   "PCM_24" if name == "baseline" or name.startswith("listened_") else None)
    finish_path = root / "outputs/final/clarity-offset/finish.json"
    receipt_path = root / PREVIEWS / "results.json"
    finish, receipt = read_json(finish_path), read_json(receipt_path)
    require(finish.get("start_frame") == FIRST and finish.get("end_frame_exclusive") == LAST
            and finish.get("onset_trim_seconds") == 0, "Finish interval changed.")
    require(receipt.get("source_start_frame") == FIRST and receipt.get("source_end_frame_exclusive") == LAST
            and receipt.get("original_offset_seconds") == 193 and receipt.get("original_end_seconds") == 218
            and Path(receipt.get("source", "")).resolve() == inputs["baseline"], "Listening receipt changed.")
    for name in ("cleanup-preview-notes.md", "brightness-preview-notes.md", "targeted-diagnosis.md"):
        inputs[name] = (root / PREVIEWS / name).resolve(strict=True)
    inputs.update(finish_receipt=finish_path.resolve(), listening_receipt=receipt_path.resolve())
    hashes = {name: digest(path) for name, path in inputs.items()}
    require(hashes["original"] == finish.get("source_sha256"), "Original soundtrack hash mismatch.")
    require(hashes["baseline"] == receipt.get("source_sha256_before_and_after"), "Preferred master hash mismatch.")
    for name in ("listened_cleanup", "listened_brightness"):
        require(hashes[name] == PREVIEW_HASHES[inputs[name].name], "Listened preview hash mismatch.")
    with sf.SoundFile(inputs["baseline"]) as baseline, sf.SoundFile(inputs["excerpt"]) as excerpt:
        baseline.seek(FIRST)
        for block in excerpt.blocks(blocksize=RATE, dtype="float64", always_2d=True):
            require(np.array_equal(block, baseline.read(len(block), always_2d=True)), "Source excerpt changed.")
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable is None:
            raise ValueError(f"Missing installed {name}.")
        tools[name] = str(Path(executable).resolve())
        tools[name + "_version"] = execute([tools[name], "-version"], 10).splitlines()[0]
    filters = execute([tools["ffmpeg"], "-hide_banner", "-filters"], 10)
    for name in ("afftdn", "equalizer", "apad", "atrim", "asetpts", "loudnorm"):
        require(any(name in line.split() for line in filters.splitlines()), f"Missing filter: {name}")
    return {"inputs": inputs, "hashes_before": hashes, "environment_before": environment(), "tools": tools,
            "source_hashes": {str(p.relative_to(root)): digest(p) for p in (root / "songtool").glob("*.py")}}


def denoise_filter(frames: int) -> str:
    require(type(frames) is int and 0 < frames <= 600 * RATE, "Invalid render length.")
    return (f"apad=pad_len={DELAY},{DENOISE},"
            f"atrim=start_sample={DELAY}:end_sample={frames + DELAY},asetpts=PTS-STARTPTS")


def filter_audio(source: Path, destination: Path, chain: str, ffmpeg: str) -> None:
    # Only local callers supply fixed chains. The CLI never accepts a filter expression.
    require(not destination.exists() and not destination.is_symlink(), "Intermediate already exists.")
    execute([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source),
             "-map", "0:a:0", "-af", chain, "-c:a", "pcm_f64le", str(destination)])


def eq_mask(start: int, count: int, windows: tuple = WINDOWS) -> np.ndarray:
    require(start >= 0 and count >= 0, "Invalid mask interval.")
    mask = np.zeros(count, dtype=np.float64)
    for first, last in windows:
        require(0 <= first < last and last - first >= 2 * FADE, "Invalid EQ window.")
        left, right = max(start, first), min(start + count, last)
        if left >= right:
            continue
        position = np.arange(left, right)
        distance = np.minimum(position - first, last - 1 - position)
        weight = 0.5 - 0.5 * np.cos(np.pi * np.minimum(distance / (FADE - 1), 1))
        mask[left - start:right - start] = weight
    return mask[:, None]


def finite_audio(audio: np.ndarray) -> None:
    require(audio.ndim == 2 and audio.shape[1] == 2 and len(audio) > 0
            and bool(np.isfinite(audio).all()), "Empty, malformed or nonfinite audio.")


def pcm24(audio: np.ndarray) -> np.ndarray:
    """Use the actual final encoder, not an assumed rounding formula (bounded block)."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, RATE, format="WAV", subtype="PCM_24")
    buffer.seek(0)
    return sf.read(buffer, dtype="float64", always_2d=True)[0]


def blend_audio(dry: Path, wet: Path, destination: Path, windows: tuple = WINDOWS) -> None:
    info = audio_info(dry)
    audio_info(wet, info.frames)
    with sf.SoundFile(dry) as a, sf.SoundFile(wet) as b, sf.SoundFile(
            destination, mode="x", samplerate=RATE, channels=2, subtype="PCM_24", format="WAV") as output:
        position = 0
        for block in a.blocks(blocksize=RATE, dtype="float64", always_2d=True):
            filtered = b.read(len(block), always_2d=True)
            finite_audio(block)
            finite_audio(filtered)
            weight = eq_mask(position, len(block), windows)
            mixed = block + (filtered - block) * weight
            require(float(np.max(np.abs(mixed))) < 1, "EQ would clip; refusing PCM conversion.")
            output.write(mixed)
            position += len(block)
    audio_info(destination, info.frames, "PCM_24")


def verify_delay(directory: Path, ffmpeg: str) -> dict:
    """Installed-executable impulse gate, including frame zero and the final sample."""
    fixture = directory / "delay-input.wav"
    output = directory / "delay-output.wav"
    count = RATE + 137
    positions = (0, 10000, count - 1)
    data = np.zeros((count, 2))
    data[list(positions)] = (0.25, -0.2)
    with sf.SoundFile(fixture, mode="x", samplerate=RATE, channels=2, subtype="FLOAT") as stream:
        stream.write(data)
    filter_audio(fixture, output, denoise_filter(count), ffmpeg)
    audio_info(output, count)
    actual, _ = sf.read(output, always_2d=True)
    finite_audio(actual)
    lags = []
    for position in positions:
        first, last = max(0, position - 1500), min(count, position + 1501)
        for channel in range(2):
            lag = first + int(np.argmax(np.abs(actual[first:last, channel]))) - position
            require(lag == 0 and abs(actual[position, channel]) > 0.1, "Denoiser delay/tail gate failed.")
            lags.append(lag)
    return {"delay_samples": DELAY, "frames": count, "impulse_positions": positions, "lags": lags}


def render_candidate(baseline: Path, directory: Path, ffmpeg: str) -> Path:
    info = audio_info(baseline)
    technical = directory / "technical"
    technical.mkdir(exist_ok=True)
    dry, wet = technical / "denoised-float.wav", technical / "eq-float.wav"
    filter_audio(baseline, dry, denoise_filter(info.frames), ffmpeg)
    audio_info(dry, info.frames)
    filter_audio(dry, wet, EQ, ffmpeg)
    candidate = directory / "candidate.wav"
    blend_audio(dry, wet, candidate)
    return candidate


def db(value: float) -> float | None:
    require(math.isfinite(value) and value >= 0, "Invalid linear measurement.")
    return 20 * math.log10(value) if value else None


def rms(audio: np.ndarray) -> float:
    finite_audio(audio)
    return float(np.sqrt(np.mean(audio ** 2)))


def analyze_windows(path: Path, blocksize: int = RATE // 2) -> list[dict]:
    audio_info(path)
    require(type(blocksize) is int and 0 < blocksize <= 5 * RATE, "Invalid analysis block size.")
    rows = []
    position = 0
    with sf.SoundFile(path) as stream:
        for audio in stream.blocks(blocksize=blocksize, dtype="float64", always_2d=True):
            finite_audio(audio)
            power = np.abs(np.fft.rfft(audio * np.hanning(len(audio))[:, None], axis=0)) ** 2
            frequencies = np.fft.rfftfreq(len(audio), 1 / RATE)
            low = float(power[(frequencies >= 250) & (frequencies < 2000)].sum())
            high = float(power[(frequencies >= 2000) & (frequencies <= 8000)].sum())
            level = rms(audio)
            rows.append({"start_frame": position, "end_frame_exclusive": position + len(audio),
                         "start_seconds": position / RATE, "duration_seconds": len(audio) / RATE,
                         "rms": level, "rms_dbfs": db(level),
                         "left_rms_dbfs": db(rms(audio[:, [0, 0]])),
                         "right_rms_dbfs": db(rms(audio[:, [1, 1]])),
                         "sample_peak": float(np.max(np.abs(audio))),
                         "spectral_ratio": high / low if low > 0 else None,
                         "spectral_state": "measured" if low > 0 else "no_low_band_energy",
                         "state": "silence" if level == 0 else "quiet" if level <= 0.001 else "active"})
            position += len(audio)
    return rows


def overall(rows: list[dict]) -> float:
    count = sum(row["end_frame_exclusive"] - row["start_frame"] for row in rows)
    require(count > 0, "Empty measurements.")
    return math.sqrt(sum(row["rms"] ** 2 * (row["end_frame_exclusive"] - row["start_frame"])
                         for row in rows) / count)


def loudness(path: Path, ffmpeg: str, silent: bool) -> dict:
    text = execute([ffmpeg, "-hide_banner", "-nostats", "-nostdin", "-i", str(path),
                    "-map", "0:a:0", "-af", "loudnorm=print_format=json", "-f", "null", "-"])
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[text.rindex("{"):])
        result = {name: float(parsed[key]) for name, key in (("lufs", "input_i"), ("true_peak_dbtp", "input_tp"))}
    except (ValueError, KeyError) as error:
        raise ValueError("Invalid loudness diagnostics.") from error
    for name, value in result.items():
        require(math.isfinite(value) or (silent and value == -math.inf), "Nonfinite nonsilent loudness.")
        if not math.isfinite(value):
            result[name] = None
    return {**result, "state": "silence" if silent else "measured"}


def correlations(baseline: Path, candidate: Path) -> list[dict]:
    frames = audio_info(baseline).frames
    count = min(RATE, frames)
    records = []
    for first in sorted({0, (frames - count) // 2, frames - count}):
        a = sf.read(baseline, start=first, frames=count, always_2d=True)[0]
        b = sf.read(candidate, start=first, frames=count, always_2d=True)[0]
        if min(rms(a), rms(b)) <= 0.001:
            records.append({"start_frame": first, "frames": count, "state": "inconclusive_low_energy", "lag": None})
            continue
        size = 1 << (2 * count - 1).bit_length()
        cross = np.fft.irfft(np.conj(np.fft.rfft(a, size, axis=0)) * np.fft.rfft(b, size, axis=0), size, axis=0).sum(axis=1)
        radius = min(2400, count - 1)
        lags = np.arange(-radius, radius + 1)
        best = int(lags[np.argmax(cross[lags % size])])
        coefficient = float(cross[best % size] / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        records.append({"start_frame": first, "frames": count, "state": "measured", "lag": best,
                        "correlation": min(1.0, coefficient), "search_radius_frames": radius})
    return records


def verify_candidate(baseline: Path, candidate: Path, dry: Path, wet: Path,
                     ffmpeg: str, windows: tuple = WINDOWS) -> dict:
    frames = audio_info(baseline).frames
    for path in (candidate, dry, wet):
        audio_info(path, frames, "PCM_24" if path == candidate else None)
    before, after = analyze_windows(baseline), analyze_windows(candidate)
    failures, warnings, checks = [], [], []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(name)

    original_rms, candidate_rms = overall(before), overall(after)
    delta = db(candidate_rms / original_rms) if original_rms and candidate_rms else None
    check("overall_rms", abs(delta) <= 0.5 if delta is not None else original_rms == candidate_rms == 0, delta)
    for a, b in zip(before, after, strict=True):
        if a["state"] == "active":
            change = db(b["rms"] / a["rms"])
            check(f"half_second_{a['start_frame']}", change is not None and abs(change) <= 1, change)
        else:
            warnings.append(f"Quiet baseline window at frame {a['start_frame']}; absolute ceiling applied.")
            check(f"quiet_{a['start_frame']}", b["rms"] <= 0.001, b["rms_dbfs"])
    block_changes = []
    with ExitStack() as stack:
        streams = [stack.enter_context(sf.SoundFile(p)) for p in (baseline, candidate, dry, wet)]
        position = 0
        dry_energy = wet_energy = 0.0
        untouched = True
        exact_recipe = True
        while position < frames:
            a, b, d, w = [stream.read(min(5 * RATE, frames - position), always_2d=True) for stream in streams]
            for data in (a, b, d, w):
                finite_audio(data)
            level, difference = rms(a), rms(b - a)
            relative = db(difference / level) if level else None
            if level > 0.001:
                check(f"difference_{position}", relative is None or relative < -20, relative)
            else:
                check(f"quiet_block_{position}", rms(b) <= 0.001, db(rms(b)))
            block_changes.append({"start_frame": position, "frames": len(a), "difference_rms": difference,
                                  "relative_difference_db": relative, "baseline_rms": level})
            mask = eq_mask(position, len(d), windows)
            outside = mask[:, 0] == 0
            untouched &= np.array_equal(b[outside], pcm24(d)[outside])
            exact_recipe &= np.array_equal(b, pcm24(d + (w - d) * mask))
            dry_energy += float(np.sum(d * d))
            wet_energy += float(np.sum(w * w))
            position += len(a)
    check("eq_untouched_regions", untouched, "Compared with identical PCM24 encoder.")
    check("eq_exact_blend", exact_recipe, "Fixed +1.5dB peaking EQ; shared raised-cosine mask.")
    eq_gain = db(math.sqrt(wet_energy / dry_energy)) if dry_energy else None
    check("eq_energy_bound", eq_gain <= 1.5 if eq_gain is not None else wet_energy == 0, eq_gain)
    measured = {}
    for name, path, rows in (("baseline", baseline, before), ("candidate", candidate, after)):
        stats = loudness(path, ffmpeg, overall(rows) == 0)
        measured[name] = stats
        peak = max(row["sample_peak"] for row in rows)
        check(name + "_no_full_scale", peak < 1, db(peak))
        check(name + "_true_peak", stats["true_peak_dbtp"] is None or stats["true_peak_dbtp"] <= -1, stats)
    alignment = correlations(baseline, candidate)
    for row in alignment:
        if row["state"] != "measured":
            warnings.append(f"Alignment inconclusive at frame {row['start_frame']} (low energy).")
        else:
            check(f"alignment_{row['start_frame']}", row["lag"] == 0 and row["correlation"] >= 0.99, row)
    return {"checks": checks, "failures": failures, "warnings": warnings, "loudness": measured,
            "overall_rms_change_db": delta, "alignment": alignment, "block_changes": block_changes,
            "measurements": {"baseline": before, "candidate": after}}


def excerpt_comparison(candidate: Path, inputs: dict[str, Path]) -> dict:
    records = {}
    for name in ("listened_cleanup", "listened_brightness"):
        energy = reference = 0.0
        exact = True
        with sf.SoundFile(candidate) as a, sf.SoundFile(inputs[name]) as b:
            a.seek(FIRST)
            for block in b.blocks(blocksize=RATE, dtype="float64", always_2d=True):
                actual = a.read(len(block), always_2d=True)
                energy += float(np.sum((actual - block) ** 2))
                reference += float(np.sum(block ** 2))
                exact &= np.array_equal(actual, block)
        records[name] = {"sample_identical": exact,
                         "difference_relative_db": db(math.sqrt(energy / reference)) if reference else None}
    return {"start_frame": FIRST, "end_frame_exclusive": LAST, "comparisons": records,
            "limitation": "Full-song filter context differs from isolated excerpt, especially near its edges."}


def select_review_windows(changes: list[dict], frames: int) -> list[int]:
    length = 5 * RATE
    require(frames >= 208 * RATE, "Recorded review anchors do not fit this timeline.")
    chosen = [195 * RATE, 203 * RATE]
    for row in sorted(changes, key=lambda row: (-row["difference_rms"], row["start_frame"])):
        first = row["start_frame"]
        if first < 0 or first + length > frames or any(abs(first - old) < length for old in chosen):
            continue
        chosen.append(first)
        if len(chosen) == 4:
            break
    return chosen


def write_review_reel(baseline: Path, candidate: Path, directory: Path,
                      selected: list[int], ffmpeg: str) -> dict:
    frames = audio_info(baseline).frames
    audio_info(candidate, frames, "PCM_24")
    require(0 < len(selected) <= 4 and len(set(selected)) == len(selected), "Invalid review selection.")
    length, separator, fade = 5 * RATE, RATE // 2, RATE // 100
    edge = np.ones((length, 1))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    edge[:fade, 0], edge[-fade:, 0] = ramp, ramp[::-1]
    mapping = []
    position = 0
    path = directory / "review-reel.wav"
    with sf.SoundFile(path, mode="x", samplerate=RATE, channels=2, subtype="PCM_24") as output:
        for pair, first in enumerate(selected):
            require(0 <= first and first + length <= frames, "Review window outside source.")
            for label, source in (("A_baseline", baseline), ("B_candidate", candidate)):
                block = sf.read(source, start=first, frames=length, always_2d=True)[0]
                finite_audio(block)
                output.write(block * edge)
                mapping.append({"pair": pair + 1, "label": label, "reel_start_frame": position,
                                "reel_end_frame_exclusive": position + length, "source_start_frame": first,
                                "source_end_frame_exclusive": first + length,
                                "source_start_seconds": first / RATE, "source_end_seconds": (first + length) / RATE})
                position += length
                output.write(np.zeros((separator, 2)))
                mapping.append({"label": "separator", "reel_start_frame": position,
                                "reel_end_frame_exclusive": position + separator, "source_start_frame": None,
                                "source_end_frame_exclusive": None})
                position += separator
    audio_info(path, position, "PCM_24")
    require(position <= 44 * RATE, "Review reel exceeded duration cap.")
    stats = loudness(path, ffmpeg, overall(analyze_windows(path)) == 0)
    require(stats["true_peak_dbtp"] is None or stats["true_peak_dbtp"] <= -1, "Review reel headroom failed.")
    result = {"frames": position, "sample_rate": RATE, "channels": 2, "shared_gain_db": 0,
              "edge_fade_frames": fade, "fade": "10ms raised cosine, identical A/B, reel only",
              "separator_frames": separator, "mapping": mapping, "loudness": stats,
              "listening_approved": False, "sha256": digest(path)}
    write_json(directory / "review-reel.json", result)
    return result


def save_metrics(directory: Path, measurements: dict[str, list[dict]]) -> None:
    fields = ["source", *next(iter(measurements.values()))[0].keys()]
    with (directory / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source, rows in measurements.items():
            for row in rows:
                writer.writerow({"source": source, **row})


def save_report(directory: Path, report: dict) -> None:
    """Publish the authoritative JSON last, without exposing a partial receipt or replacing files."""
    lines = ["# Full-song cleanup trial", "", f"Status: **{report['status']}**; listening approval: **not granted**.",
             "Preferred master remains outputs/final/clarity-offset/song-offset.wav.",
             f"Runtime: {report['elapsed_seconds']:.3f} seconds.", "",
             "Intended settings: one continuous denoise pass, 1,200-frame compensation; fixed localized +1.5dB EQ only.",
             "Intended frame coverage: all original frames, including the opening. Onset remains unconfirmed.",
             "The intended recipe does not specifically target the brief sound at original 197–198s.",
             "No FireRed gate, separation, GPU scan, normalization, automatic promotion or iterative repair.",
             "Technical thresholds are conservative experiment limits, not perceptual guarantees.",
             "The excerpt preference does not establish full-song improvement; residual speech and dullness remain subjective.",
             "Full-song context can differ from the isolated listened excerpt near its edges.", "",
             "## Outcome", str(report.get("error", "All automatic guards passed.")),
             "This Markdown is provisional until final report.json exists; any failed.json overrides success."]
    delay = report.get("delay_fixture", {}).get("delay_samples")
    lines.append(f"Delay fixture: configured {delay:,}-frame compensation verified."
                 if delay is not None else "Delay fixture: not verified.")
    lines.append("Rendering: completed." if "render" in report.get("timings", {})
                 else "Rendering: not completed.")
    verification = report.get("verification")
    lines.append("Frame preservation: matching output frame counts verified; onset remains unconfirmed."
                 if verification is not None else "Frame preservation: not verified.")
    delta = verification.get("overall_rms_change_db") if verification is not None else None
    lines.append(f"Overall RMS change: {delta} dB." if delta is not None
                 else "Overall RMS change: not measured (silence or unavailable diagnostics).")
    if verification is not None:
        lines.append("Detailed checks and warnings are in report.json.")
    if "excerpt_comparison" in report:
        lines.append("Excerpt comparisons are in report.json.")
    else:
        lines.append("Excerpt comparisons: not completed.")
    lines.append("Available provenance, hashes, versions and timings are in report.json.")
    if report["status"] == "verified_candidate":
        lines.extend(["", "## Optional listening", "candidate.wav is the single unapproved full-song candidate.",
                      "review-reel.wav: A baseline then B candidate, unity gain, 0.5s separators, identical 10ms edge fades.",
                      "review-reel.json maps every frame range to source time. Do not change player gain between A and B."])
    else:
        lines.append("Partial audio is diagnostic only; do not treat it as a verified listening candidate.")
    with (directory / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    # Keep staging diagnostics: no fallible cleanup follows the exclusive completion commit.
    staged = directory / "report.pending.json"
    write_json(staged, report)
    os.link(staged, directory / "report.json")


def preview(destination: Path) -> Path:
    started = time.monotonic()
    report = {"status": "incomplete", "listening_approved": False, "timings": {},
              "recipe": {"denoise": DENOISE, "delay_frames": DELAY, "eq": EQ, "eq_gain_db": 1.5,
                         "windows": WINDOWS, "fade_frames": FADE, "frames": FRAMES, "sample_rate": RATE,
                         "channels": 2, "output_encoding": "PCM_24", "normalization": False},
              "reference": "AIEraDev/Clypra@6b8fcabcb23bb4bcbe58654a9cd6476e58ec9fdc export.rs:460-520"}
    context = None
    with run_directory(destination) as directory:
        try:
            require(shutil.disk_usage(directory).free >= FRAMES * 38 + 256 * 1024 * 1024,
                    "Insufficient disk for two float64 intermediates, candidate, diagnostics and reel.")
            context = preflight()
            report["provenance"] = {**context, "inputs": {k: str(v) for k, v in context["inputs"].items()}}
            report["timings"]["preflight"] = time.monotonic() - started
            baseline = context["inputs"]["baseline"]
            ffmpeg = context["tools"]["ffmpeg"]
            baseline_rows = analyze_windows(baseline)
            baseline_stats = loudness(baseline, ffmpeg, overall(baseline_rows) == 0)
            report["baseline_loudness"] = baseline_stats
            require(baseline_stats["true_peak_dbtp"] is None or baseline_stats["true_peak_dbtp"] <= -1,
                    "Baseline violates -1 dBTP ceiling; stopped without normalization.")
            technical = directory / "technical"
            technical.mkdir(exist_ok=False)
            report["delay_fixture"] = verify_delay(technical, ffmpeg)
            checkpoint = time.monotonic()
            candidate = render_candidate(baseline, directory, ffmpeg)
            report["timings"]["render"] = time.monotonic() - checkpoint
            candidate = directory / "candidate.wav"
            checkpoint = time.monotonic()
            verification = verify_candidate(baseline, candidate, technical / "denoised-float.wav",
                                            technical / "eq-float.wav", ffmpeg)
            report["verification"] = verification
            measurements = verification.pop("measurements")
            report["context_reference_summaries"] = {}
            diagnostics = report["diagnostics"] = {"status": "incomplete"}
            try:
                for name in ("original", "raw_music", "offset_music"):
                    diagnostics["stage"] = "context_reference:" + name
                    measurements[name] = analyze_windows(context["inputs"][name])
                    report["context_reference_summaries"][name] = {
                        "rms_dbfs": db(overall(measurements[name])), "role": "Context only, not a clean-music target"}
                diagnostics["stage"] = "metrics_csv"
                save_metrics(directory, measurements)
                diagnostics["stage"] = "excerpt_comparison"
                report["excerpt_comparison"] = excerpt_comparison(candidate, context["inputs"])
            except BaseException as error:
                diagnostics.update(status="failed", error=str(error) or type(error).__name__)
                raise
            report["diagnostics"] = {"status": "complete"}
            report["timings"]["diagnostics"] = time.monotonic() - checkpoint
            require(not verification["failures"], "Technical guards failed: " + ", ".join(verification["failures"]))
            checkpoint = time.monotonic()
            selected = select_review_windows(verification["block_changes"], FRAMES)
            report["reel"] = write_review_reel(baseline, candidate, directory, selected, ffmpeg)
            report["timings"]["reel"] = time.monotonic() - checkpoint
            report["candidate_sha256"] = digest(candidate)
        except BaseException as error:
            report.update(status="failed", error=str(error) or type(error).__name__)
            raise
        finally:
            try:
                if context is not None:
                    report["hashes_after"] = {name: digest(path) for name, path in context["inputs"].items()}
                    report["environment_after"] = environment()
                    require(report["hashes_after"] == context["hashes_before"], "Input preservation check failed.")
                    require(report["environment_after"] == context["environment_before"], "Package metadata changed.")
                    require(all(digest(ROOT / p) == h for p, h in context["source_hashes"].items()), "Implementation changed during run.")
                    report["preservation_verified"] = True
                if report["status"] != "failed":
                    report["status"] = "verified_candidate"
            except BaseException as error:
                report.update(status="failed", error=str(error) or type(error).__name__)
                raise
            finally:
                report["elapsed_seconds"] = time.monotonic() - started
                save_report(directory, report)
    print(f"Verified technical candidate (not listening approved): {directory / 'candidate.wav'}")
    print(f"Optional A/B reel: {directory / 'review-reel.wav'}")
    return directory
