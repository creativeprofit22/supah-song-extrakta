"""Aligned, fixed-gain listening comparisons; never modifies existing media."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from .media import duration, extract, new_output, run

RATE = 48000
PEAK_CEILING = -2.0
MAX_BOOST = 6.0
QUIET_RMS = -60.0


def levels(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid comparison audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    result = run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
                  "-map", "0:a:0", "-af", "loudnorm=print_format=json",
                  "-f", "null", "-"], capture=True)
    # Analysis only: the filter's processed audio is discarded, never exported.
    try:
        stats, _ = json.JSONDecoder().raw_decode(result.stderr[result.stderr.rindex("{"):])
        measured = {name: float(stats[key]) for name, key in
                    (("lufs", "input_i"), ("true_peak_dbtp", "input_tp"))}
    except (ValueError, KeyError) as error:
        raise ValueError("FFmpeg returned invalid loudness measurements.") from error
    if any(math.isnan(v) or v == math.inf for v in measured.values()):
        raise ValueError("Non-finite loudness measurements.")
    measured = {key: value if math.isfinite(value) else None for key, value in measured.items()}
    rms_db = 20 * math.log10(rms) if rms else None
    return {**measured, "frames": len(audio), "sample_rate": rate, "channels": 2,
            "sample_peak_dbfs": 20 * math.log10(peak) if peak else None,
            "rms_dbfs": rms_db, "quiet": rms_db is None or rms_db < QUIET_RMS or measured["lufs"] is None}


def compare(source: Path, stems: Path, master: Path, directory: Path,
            start: float, end: float, *, open_folder: bool = True,
            experiment: Path | None = None) -> Path:
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end or end - start > 60:
        raise ValueError("Choose a finite, positive range of at most 60 seconds.")
    first, last = round(start * RATE), round(end * RATE)
    if last - first < RATE:
        raise ValueError("Compare at least one second of audio.")
    inputs = {"original": source, "raw_music": stems / "music.wav", "mastered_music": master,
              "speech_reference": stems / "speech.wav", "effects_reference": stems / "sfx.wav"}
    inputs = {name: path.resolve(strict=True) for name, path in inputs.items()}
    infos = {name: sf.info(path) for name, path in inputs.items()}
    for name, info in infos.items():
        if info.samplerate != RATE or info.channels != 2 or info.frames != infos["original"].frames:
            raise ValueError(f"{name}: inputs must share a stereo 48 kHz, full-source timeline.")
        if last > info.frames or end > duration(inputs[name]):
            raise ValueError(f"Range extends beyond {name}.")
    origins = {name: first for name in inputs}
    experiment_settings = None
    if experiment is not None:
        experiment = experiment.resolve(strict=True)
        metadata = experiment / "experiment.json"
        if metadata.stat().st_size > 65536:
            raise ValueError("Oversized experiment metadata.")
        experiment_settings = json.loads(metadata.read_text(encoding="utf-8"))
        with inputs["original"].open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        required = {"version": 1, "source": str(inputs["original"]), "source_sha256": source_hash,
                    "start_frame": first, "end_frame_exclusive": last, "frames": last - first,
                    "sample_rate": RATE, "channels": 2}
        if not isinstance(experiment_settings, dict) or any(experiment_settings.get(k) != v for k, v in required.items()):
            raise ValueError("Experiment source/range does not match this comparison.")
        for name in ("grid_music", "offset_music", "average_music"):
            path = (experiment / f"{name}.wav").resolve(strict=True)
            info = sf.info(path)
            if path.parent != experiment or info.frames != last - first or info.samplerate != RATE or info.channels != 2:
                raise ValueError(f"Invalid candidate: {name}")
            inputs[name], origins[name] = path, 0
    directory = new_output(directory)
    directory.mkdir(exist_ok=False)
    unadjusted = directory / "unadjusted"
    unadjusted.mkdir()
    records = {}
    for name, path in inputs.items():
        clip = unadjusted / f"{name}.wav"
        origin = origins[name]
        extract(path, clip, origin / RATE, (origin + last - first) / RATE)
        audio, _ = sf.read(clip, dtype="float32", always_2d=True)
        expected, _ = sf.read(path, start=origin, frames=last - first, dtype="float32", always_2d=True)
        if len(audio) != last - first or not np.array_equal(audio, expected):
            raise ValueError(f"Excerpt is not sample-aligned: {name}")
        records[name] = {"source": str(path), "input_start_frame": origin,
                         "source_timeline_start_frame": first, "unadjusted": str(clip), "before": levels(clip)}
    main = [name for name in records if not name.endswith("_reference")]
    eligible = [records[name]["before"] for name in main if not records[name]["before"]["quiet"]]
    target = min([-16.0] + [min(v["lufs"] + MAX_BOOST, v["lufs"] + PEAK_CEILING - v["true_peak_dbtp"])
                           for v in eligible]) if eligible else None
    references = [v["before"]["true_peak_dbtp"] for name, v in records.items()
                  if name.endswith("_reference") and v["before"]["true_peak_dbtp"] is not None]
    reference_gain = min([0.0] + [PEAK_CEILING - peak for peak in references])
    for name, record in records.items():
        before = record["before"]
        if name.endswith("_reference"):
            gain = reference_gain
        elif before["quiet"]:
            gain = min(0.0, PEAK_CEILING - before["true_peak_dbtp"]) if before["true_peak_dbtp"] is not None else 0.0
        else:
            gain = target - before["lufs"]
        audio, _ = sf.read(record["unadjusted"], dtype="float32", always_2d=True)
        scaled = (audio.astype(np.float64) * 10 ** (gain / 20)).astype(np.float32)
        destination = new_output(directory / f"{name}.wav")
        sf.write(destination, scaled, RATE, subtype="FLOAT")
        actual, _ = sf.read(destination, dtype="float32", always_2d=True)
        if not np.array_equal(actual, scaled):
            raise ValueError(f"Fixed-gain verification failed: {name}")
        after = levels(destination)
        if after["frames"] != last - first or (after["true_peak_dbtp"] is not None and after["true_peak_dbtp"] > PEAK_CEILING + 0.05):
            raise ValueError(f"Comparison missed alignment/peak limits: {name}")
        record.update(output=str(destination), gain_db=gain, after=after,
                      loudness_matched=name in main and not before["quiet"])
    manifest = {"version": 1, "start_seconds": first / RATE, "end_seconds": last / RATE,
                "start_frame": first, "end_frame_exclusive": last, "frames": last - first,
                "sample_rate": RATE, "channels": 2, "shared_target_lufs": target,
                "peak_ceiling_dbtp": PEAK_CEILING, "max_boost_db": MAX_BOOST,
                "quiet_threshold_rms_dbfs": QUIET_RMS, "reference_gain_db": reference_gain,
                "processing": "Constant gain only; no EQ, fades, dynamic normalization or denoising.",
                "experiment": experiment_settings,
                "listening_outcome": "Pending; levels do not prove clarity or isolation.", "files": records}
    guide = ["# Listening comparison", "", f"Source timeline: {first / RATE:g}–{last / RATE:g} seconds.",
             f"Main comparison target: {target} LUFS; peak ceiling: {PEAK_CEILING} dBTP.",
             "Quiet clips are NOT loudness-matched and never boosted. Unadjusted clips are retained.",
             f"Speech/effects share {reference_gain:.3f} dB reference gain; they are NOT loudness-matched.", "",
             "Listen at unchanged player volume, with player normalization/enhancements disabled.",
             "Compare detail/brightness, singing intelligibility, percussion attacks, and dialogue/fighting leakage.",
             "Compare the control too: a brighter result is not better if clean sections regress.",
             "Numeric checks do not decide sound quality. No baseline files were replaced.", ""]
    for name, record in records.items():
        guide.extend([f"## {name}", f"`{record['output']}`", f"Gain: {record['gain_db']:.3f} dB; measured: {record['after']['lufs']} LUFS.", ""])
    (directory / "LISTEN.md").write_text("\n".join(guide), encoding="utf-8")
    # Written last: a folder without this manifest is incomplete.
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Comparison: {directory}")
    if open_folder and os.name == "nt":
        try:
            os.startfile(directory)
        except OSError as error:
            print(f"Folder could not be opened: {error}")
    return directory


def finish_offset(comparison: Path, directory: Path) -> Path:
    """Apply a listener-approved offset excerpt; preserve everything outside it."""
    from .mastering import measure

    comparison = comparison.resolve(strict=True)
    metadata = comparison / "manifest.json"
    if metadata.stat().st_size > 65536:
        raise ValueError("Oversized comparison manifest.")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    try:
        first, last = manifest["start_frame"], manifest["end_frame_exclusive"]
        settings = manifest["experiment"]
        source = Path(manifest["files"]["original"]["source"]).resolve(strict=True)
        baselines = {name: Path(manifest["files"][name]["source"]).resolve(strict=True)
                     for name in ("raw_music", "mastered_music")}
        if not isinstance(settings, dict):
            raise ValueError("A completed offset experiment is required.")
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid comparison manifest.") from error
    if type(first) is not int or type(last) is not int or not 0 <= first < last or not RATE <= last - first <= 60 * RATE:
        raise ValueError("Invalid approved interval.")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if (settings.get("source_sha256") != source_hash or settings.get("source") != str(source)
            or settings.get("start_frame") != first or settings.get("end_frame_exclusive") != last):
        raise ValueError("Experiment source/range changed.")
    info = sf.info(source)
    if info.samplerate != RATE or info.channels != 2 or last > info.frames or info.frames > 600 * RATE:
        raise ValueError("Finish only stereo 48 kHz sources up to ten minutes.")
    candidate_path = (comparison / "unadjusted" / "offset_music.wav").resolve(strict=True)
    if candidate_path.parent != comparison / "unadjusted":
        raise ValueError("Candidate must stay inside the comparison folder.")
    candidate, rate = sf.read(candidate_path, dtype="float32", always_2d=True)
    if rate != RATE or candidate.shape != (last - first, 2) or not np.isfinite(candidate).all():
        raise ValueError("Invalid approved candidate.")
    try:
        audition_gain = float(manifest["files"]["offset_music"]["gain_db"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing candidate comparison gain.") from error
    audition, audition_rate = sf.read(comparison / "offset_music.wav", dtype="float32", always_2d=True)
    if not math.isfinite(audition_gain) or audition_gain > MAX_BOOST + 0.001:
        raise ValueError("Invalid candidate comparison gain.")
    if audition_rate != RATE or not np.array_equal(
            audition, (candidate.astype(np.float64) * 10 ** (audition_gain / 20)).astype(np.float32)):
        raise ValueError("Candidate differs from the listened comparison.")
    candidate_levels = levels(candidate_path)
    if candidate_levels["quiet"]:
        raise ValueError("Cannot automatically join a silent/very quiet candidate.")
    for name, baseline in baselines.items():
        baseline_info = sf.info(baseline)
        if baseline_info.frames != info.frames or baseline_info.samplerate != RATE or baseline_info.channels != 2:
            raise ValueError("Baseline timeline changed.")
        if name == "mastered_music" and baseline_info.subtype != "PCM_24":
            raise ValueError("Finishing requires the existing 24-bit WAV master.")
        excerpt, _ = sf.read(baseline, start=first, frames=last - first, dtype="float32", always_2d=True)
        reference, reference_rate = sf.read(comparison / "unadjusted" / f"{name}.wav", dtype="float32", always_2d=True)
        if reference_rate != RATE or not np.array_equal(excerpt, reference):
            raise ValueError("Baseline no longer matches the listened comparison.")
    directory = new_output(directory)
    directory.mkdir(exist_ok=False)
    fade = RATE // 10
    weight = np.ones((last - first, 1), dtype=np.float64)
    weight[:fade, 0] = np.linspace(0, 1, fade)
    weight[-fade:, 0] = np.linspace(1, 0, fade)
    report = {"comparison": str(comparison), "source_sha256": source_hash,
              "start_frame": first, "end_frame_exclusive": last, "crossfade_frames": fade,
              "onset_trim_seconds": 0, "onset_status": "Unconfirmed; full timeline retained.",
              "selection": "offset_music", "outputs": {}}
    for name, baseline in baselines.items():
        audio, _ = sf.read(baseline, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError("Non-finite baseline audio.")
        baseline_levels = levels(comparison / "unadjusted" / f"{name}.wav")
        if baseline_levels["quiet"]:
            raise ValueError("Cannot automatically join into a silent/very quiet baseline interval.")
        gain = 0.0 if name == "raw_music" else min(
            baseline_levels["lufs"] - candidate_levels["lufs"], MAX_BOOST,
            PEAK_CEILING - candidate_levels["true_peak_dbtp"])
        if not math.isfinite(gain):
            raise ValueError("Invalid joining gain.")
        mixed = (audio[first:last].astype(np.float64) * (1 - weight)
                 + candidate.astype(np.float64) * 10 ** (gain / 20) * weight).astype(np.float32)
        if not np.isfinite(mixed).all() or np.max(np.abs(mixed)) >= 1:
            raise ValueError("Joined interval would clip.")
        result = audio.copy()
        result[first:last] = mixed
        subtype = "FLOAT" if name == "raw_music" else "PCM_24"
        output = directory / ("music-offset.wav" if name == "raw_music" else "song-offset.wav")
        sf.write(output, result, RATE, subtype=subtype)
        saved, saved_rate = sf.read(output, dtype="float32", always_2d=True)
        if (saved_rate != RATE or saved.shape != audio.shape
                or not np.array_equal(saved[:first], audio[:first])
                or not np.array_equal(saved[last:], audio[last:])
                or not np.allclose(saved[first:last], mixed, rtol=0, atol=2 ** -23)):
            raise ValueError("Joined export failed unchanged-region or sample verification.")
        boundaries = [i for i in (first, first + fade, last - fade, last) if 0 < i < len(audio)]
        joins = {str(i): {"baseline_step": float(np.max(np.abs(audio[i] - audio[i - 1]))),
                          "joined_step": float(np.max(np.abs(saved[i] - saved[i - 1])))} for i in boundaries}
        final = measure(output, "loudnorm=print_format=json")
        if float(final["input_tp"]) >= 0 or (name == "mastered_music" and float(final["input_tp"]) > -1.0):
            raise ValueError("Joined WAV exceeds peak headroom; no completion report written.")
        report["outputs"][name] = {"path": str(output), "baseline": str(baseline), "gain_db": gain,
                                    "frames": len(saved), "lufs": float(final["input_i"]),
                                    "true_peak_dbtp": float(final["input_tp"]), "joins": joins,
                                    "outside_interval": "Verified sample-identical to baseline."}
    mp3 = directory / "song-offset.mp3"
    run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(directory / "song-offset.wav"),
         "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)], capture=True)
    encoded = measure(mp3, "loudnorm=print_format=json")
    if float(encoded["input_tp"]) >= 0:
        raise ValueError("MP3 exceeds peak headroom; no completion report written.")
    report["outputs"]["mp3"] = {"path": str(mp3), "lufs": float(encoded["input_i"]),
                                    "true_peak_dbtp": float(encoded["input_tp"])}
    (directory / "finish.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Approved offset exports: {directory}")
    if os.name == "nt":
        try:
            os.startfile(directory)
        except OSError as error:
            print(f"Folder could not be opened: {error}")
    return directory
