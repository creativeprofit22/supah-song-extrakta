"""Cinematic Bandit v2: music / speech / effects, with overlap-add inference."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .resources import CODE, REVISION, WEIGHTS, WEIGHTS_SHA256, validate_resources


def separate(source: Path, directory: Path, device: str = "cuda") -> None:
    import soundfile as sf

    source = source.resolve(strict=True)
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError("Choose a fresh separation directory; existing stems are preserved.")
    info = sf.info(source)
    if info.samplerate != 48000 or info.channels != 2 or not 0 < info.duration <= 600:
        raise ValueError("Extract a stereo 48 kHz WAV first, at most ten minutes long.")
    model = _load_model(device)
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    result = _infer_grid(audio, model, device)
    directory.mkdir(parents=True, exist_ok=False)
    for name, stem in zip(("speech", "music", "sfx"), result):
        sf.write(directory / f"{name}.wav", stem.T, rate, subtype="FLOAT")
    print(f"Stems saved: {directory}", flush=True)


def _load_model(device: str):
    import torch

    validate_resources()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu (much slower).")
    sys.path.insert(0, str(CODE))
    from models.bandit_v2.bandit import Bandit

    torch.set_num_threads(4)
    # Matches the published config's kwargs, including its default fs=44100.
    # The model's waveform input remains 48 kHz, as specified by audio.sample_rate.
    model = Bandit(in_channels=1, stems=["speech", "music", "sfx"],
                   n_sqm_modules=8, rnn_type="GRU", pad_mode="reflect")
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(device)
    return model


def _infer_grid(audio, model, device: str, offset: int = 0):
    import numpy as np
    import torch

    if offset not in (0, 48000):
        raise ValueError("Only the baseline or one-second offset grid is supported.")
    if not np.isfinite(audio).all():
        raise ValueError("Input contains non-finite samples.")
    frames = len(audio)
    chunk, step = 384000, 96000  # Published 8-second windows / 75% overlap.
    border = chunk - step
    left = border + offset
    mix = np.pad(audio.T, ((0, 0), (left, border)), mode="reflect")
    # simplification: RAM accumulation is capped at ten-minute inputs; stream for long films.
    result = np.zeros((3, 2, mix.shape[-1]), dtype=np.float32)
    counter = np.zeros(mix.shape[-1], dtype=np.float32)
    fade = chunk // 10
    window = np.ones(chunk, dtype=np.float32)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    starts = range(0, mix.shape[-1], step)
    begin = time.monotonic()
    with torch.inference_mode():
        for index, start in enumerate(starts):
            part = mix[:, start:start + chunk]
            size = part.shape[-1]
            part = np.pad(part, ((0, 0), (0, chunk - size)))
            prediction = model(torch.from_numpy(part.copy()).unsqueeze(0).to(device))[0].cpu().numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Model produced non-finite samples; no stems exported.")
            weight = window.copy()
            if start == 0:
                weight[:fade] = 1
            if start + chunk >= mix.shape[-1]:
                weight[-fade:] = 1
            result[:, :, start:start + size] += prediction[:, :, :size] * weight[:size]
            counter[start:start + size] += weight[:size]
            print(f"Separation {index + 1}/{len(starts)} | {time.monotonic() - begin:.0f}s elapsed", flush=True)
    result = result[:, :, left:left + frames]
    result /= counter[left:left + frames]
    if not np.isfinite(result).all():
        raise ValueError("Invalid overlap-add reconstruction.")
    return result


def experiment(source: Path, directory: Path, start: float, end: float,
               device: str = "cuda") -> None:
    """Two source-aligned grids, with real context; exports candidates, not a replacement."""
    import numpy as np
    import soundfile as sf

    source = source.resolve(strict=True)
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError("Choose a fresh experiment directory.")
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end or not 1 <= end - start <= 60:
        raise ValueError("Experiment on a finite range of one to sixty seconds.")
    info = sf.info(source)
    first, last = round(start * 48000), round(end * 48000)
    if info.samplerate != 48000 or info.channels != 2 or last > info.frames or end > info.duration:
        raise ValueError("Select a range within a stereo 48 kHz source.")
    # Keep the baseline grid anchored to the FULL source's two-second hop.
    context_first = max(0, ((first - 384000) // 96000) * 96000)
    context_last = min(info.frames, last + 384000)
    audio, rate = sf.read(source, start=context_first, frames=context_last - context_first,
                          dtype="float32", always_2d=True)
    if len(audio) != context_last - context_first or not np.isfinite(audio).all():
        raise ValueError("Invalid contextual source audio.")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    model = _load_model(device)
    crop = slice(first - context_first, last - context_first)
    directory.mkdir(parents=True, exist_ok=False)
    estimates = []
    for offset, name in ((0, "grid_music"), (48000, "offset_music")):
        # Copy only the cropped music so full three-stem CPU accumulation can be freed.
        music = _infer_grid(audio, model, device, offset)[1, :, crop].T.copy()
        if music.shape != (last - first, 2) or not np.isfinite(music).all():
            raise ValueError("Candidate failed frame/finite checks.")
        sf.write(directory / f"{name}.wav", music, rate, subtype="FLOAT")
        estimates.append(music)
    average = (estimates[0] + estimates[1]) * 0.5
    if not np.isfinite(average).all():
        raise ValueError("Invalid averaged candidate.")
    sf.write(directory / "average_music.wav", average, rate, subtype="FLOAT")
    manifest = {"version": 1, "source": str(source), "source_sha256": source_hash,
                "start_frame": first, "end_frame_exclusive": last, "frames": last - first,
                "sample_rate": rate, "channels": 2, "context_start_frame": context_first,
                "context_end_frame_exclusive": context_last, "window_frames": 384000,
                "hop_frames": 96000, "offset_frames": 48000, "average_weights": [0.5, 0.5],
                "source_revision": REVISION, "checkpoint_sha256": WEIGHTS_SHA256,
                "device": device, "precision": "float32", "batch_size": 1,
                "alignment": "Extra left reflection padding shifts the grid; crop removes padding and real context. No circular splice.",
                "listening_outcome": "Pending; neither offset nor averaging is assumed better."}
    (directory / "experiment.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Experiment: {directory}", flush=True)
