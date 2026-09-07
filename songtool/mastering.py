"""Conservative two-pass loudness mastering; retains the raw separated stem."""

import json
import math
from pathlib import Path

from .media import duration, new_output, run


def statistics(stderr: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(stderr):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stderr[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "input_i" in value:
            for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
                if not math.isfinite(float(value[key])):
                    raise ValueError("Cannot master silent or non-finite audio.")
            return value
    raise ValueError("FFmpeg returned no loudness measurements.")


def measure(source: Path, filters: str) -> dict:
    result = run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(source),
                  "-map", "0:a:0", "-af", filters, "-f", "null", "-"], capture=True)
    return statistics(result.stderr)


def master(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.suffix.lower() != ".wav":
        raise ValueError("Master output must use .wav; an MP3 is also created automatically.")
    destination = new_output(destination)
    mp3 = new_output(destination.with_suffix(".mp3"))
    report = new_output(destination.with_suffix(".json"))
    seconds = duration(source)
    if seconds < 1:
        raise ValueError("Master at least one second of music.")
    cleanup = f"highpass=f=25,afade=t=in:d=0.05,afade=t=out:st={seconds - 0.5}:d=0.5"
    # A large allowed LRA preserves dynamics where linear gain meets the peak ceiling.
    target = "loudnorm=I=-16:TP=-1.5:LRA=50"
    first = measure(source, f"{cleanup},{target}:print_format=json")
    options = ":".join(f"{key}={float(first[value])}" for key, value in (
        ("measured_I", "input_i"), ("measured_TP", "input_tp"),
        ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"),
        ("offset", "target_offset")))
    filters = f"{cleanup},{target}:{options}:linear=true:print_format=json"
    run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(source),
         "-map", "0:a:0", "-af", filters, "-ar", "48000", "-c:a", "pcm_s24le", str(destination)], capture=True)
    final = measure(destination, target + ":print_format=json")
    if float(final["input_tp"]) > -1.0 or abs(float(final["input_i"]) + 16) > 1:
        raise ValueError("Master missed loudness/peak limits; inspect the WAV before using it.")
    run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(destination),
         "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)], capture=True)
    encoded = measure(mp3, target + ":print_format=json")
    if float(encoded["input_tp"]) >= 0:
        raise ValueError("Encoded MP3 exceeds the peak limit; use the WAV and lower gain before re-encoding.")
    report.write_text(json.dumps({"source": str(source), "duration_seconds": seconds,
                                 "target_lufs": -16, "target_true_peak_dbtp": -1.5,
                                 "before": first, "wav": final, "mp3": encoded,
                                 "listening_review": "Not performed; separation artifacts may remain."}, indent=2) + "\n", encoding="utf-8")
    print(f"Master: {destination}\nMP3: {mp3}\nMeasurements: {report}")
