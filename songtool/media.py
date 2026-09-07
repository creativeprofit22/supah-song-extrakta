"""Download and decode without shell interpolation or destructive overwrites."""

import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import parse_qs, urlparse


def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, capture_output=capture)


def new_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Output already exists; choose a new name: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, directory: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"youtube.com", "www.youtube.com", "youtu.be"}:
        raise ValueError("Use an HTTPS YouTube video URL.")
    video_id = parsed.path[1:] if parsed.hostname == "youtu.be" else parse_qs(parsed.query).get("v", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Expected a single YouTube video, not a playlist.")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-plugin-dirs",
         "--js-runtimes", "node", "--no-playlist", "--no-overwrites",
         "--max-filesize", "1G", "--socket-timeout", "30", "--retries", "3",
         "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mkv",
         "-o", str(directory / f"{video_id}.%(ext)s"),
         f"https://www.youtube.com/watch?v={video_id}"])


def duration(source: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "json", str(source.resolve(strict=True))], capture=True)
    value = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Input has no valid duration.")
    return value


def extract(source: Path, destination: Path, start: float = 0, end: float | None = None) -> None:
    source = source.resolve(strict=True)
    if destination.suffix.lower() != ".wav":
        raise ValueError("Extraction output must use .wav.")
    run(extraction_command(source, destination, start, end, duration(source)))


def extraction_command(source: Path, destination: Path, start: float, end: float | None,
                       total: float, *, threads: int | None = None) -> list[str]:
    """Shared extraction argv; job imports supply already-probed duration and CPU limits."""
    source = source.resolve(strict=True)
    if destination.suffix.lower() != ".wav":
        raise ValueError("Extraction output must use .wav.")
    end = total if end is None else end
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= total:
        raise ValueError(f"Select a range inside the input (0 to {total:.3f} seconds).")
    destination = new_output(destination)
    limits = [] if threads is None else ["-threads", str(threads)]
    filters = [] if threads is None else ["-filter_threads", str(threads), "-filter_complex_threads", str(threads)]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-n", *filters, *limits, "-i", str(source),
            "-ss", str(start), "-t", str(end - start), "-map", "0:a:0", "-vn",
            "-ac", "2", "-ar", "48000", "-c:a", "pcm_f32le", *limits, str(destination)]
