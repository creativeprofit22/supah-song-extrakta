"""CPU-only listening suggestions and optional exports; never listening approval.

Receipts live in exclusive runs/<id> directories, published last. No Run state is
invented here: orchestration/resource accounting belongs to the future workflow.
Only a completed receipt establishes completion; incomplete directories survive.
"""

from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from . import jobs
from .cleanup import analyze_windows, digest, require
from .comparison import levels, MAX_BOOST, PEAK_CEILING
from .jobs import RATE, record_feedback, mapped_protected_ranges


def _version(job, identifier):
    identifier = job.current_version if identifier == "current" else identifier
    version = next((v for v in job.versions if v.id == identifier), None)
    require(version is not None, "Unknown review version.")
    return version


def _workspace(directory, job, operation, size=0):
    root = jobs._plain(Path(directory), directory=True)
    parent = jobs._plain(root / "runs", directory=True)
    require(size <= job.policy.max_output_bytes and shutil.disk_usage(root).free >= size + jobs.MAX_METADATA_BYTES,
            "Insufficient review output space/budget.")
    output = parent / jobs.new_id()
    output.mkdir(exist_ok=False)
    _receipt(output / "incomplete.json", {"operation": operation, "job_id": job.id})
    return output


def _receipt(path, data, limit=jobs.MAX_RECEIPT_BYTES):
    jobs._publish(path, jobs._encode(data, limit))


def _clip(version, first, last):
    identity = f"review-clip-v1:{version.sha256}:{version.source_start_frame}:{first}:{last}"
    return {"id": hashlib.sha256(identity.encode()).hexdigest()[:32],
            "version_id": version.id, "parent_sha256": version.sha256,
            "start_frame": first, "end_frame_exclusive": last,
            "source_start_frame": version.source_start_frame + first,
            "source_end_frame_exclusive": version.source_start_frame + last}


def scan_version(directory: Path, version_id: str = "current", *,
                 speech_version_id: str | None = None, max_clips: int = 8,
                 _output: Path | None = None) -> Path:
    """Save half-second rankings and <=8 nonoverlapping, at-most-five-second clips.

    Speech must be a registered version with an explicitly attested source map;
    equal durations alone never establish alignment. Missing/quiet hints are None.
    Returns the immutable scan receipt path, not a promoted version.
    """
    jobs._integer(max_clips, 0, 8, "clip count")
    job = jobs.load_job(directory)
    version = _version(job, version_id)
    path = Path(directory) / version.path
    speech = _version(job, speech_version_id) if speech_version_id is not None else None
    if speech:
        require(speech.source_start_frame <= version.source_start_frame and
                version.source_start_frame + version.frames <= speech.source_start_frame + speech.frames,
                "Speech map does not cover selected version.")
        jobs._finite_audio(Path(directory) / speech.path)
    rows = analyze_windows(path)
    with sf.SoundFile(path) as stream:
        reference = sf.SoundFile(Path(directory) / speech.path) if speech else None
        try:
            if reference:
                reference.seek(version.source_start_frame - speech.source_start_frame)
            for row in rows:
                size = row["end_frame_exclusive"] - row["start_frame"]
                a = stream.read(size, dtype="float64", always_2d=True)
                row["speech_similarity_hint"] = None
                row["speech_reference_rms_dbfs"] = None
                if reference:
                    b = reference.read(size, dtype="float64", always_2d=True)
                    require(a.shape == b.shape and bool(np.isfinite(b).all()), "Invalid speech excerpt.")
                    rms = float(np.sqrt(np.mean(b * b)))
                    row["speech_reference_rms_dbfs"] = 20 * math.log10(rms) if rms else None
                    frequencies = np.fft.rfftfreq(size, 1 / RATE)
                    band = (frequencies >= 300) & (frequencies <= 3400)
                    window = np.hanning(size)[:, None]
                    x, y = (np.fft.rfft(c * window, axis=0)[band] for c in (a, b))
                    denominator = math.sqrt(float(np.sum(abs(x) ** 2)) * float(np.sum(abs(y) ** 2)))
                    if denominator and row["state"] == "active" and rms > 0.001:
                        row["speech_similarity_hint"] = min(1.0, abs(float(np.vdot(x, y).real)) / denominator)
                ratio = row["spectral_ratio"]
                row["tone_ratio_db"] = 10 * math.log10(ratio) if ratio is not None and ratio > 0 else None
                row["source_start_frame"] = version.source_start_frame + row["start_frame"]
                row["source_end_frame_exclusive"] = version.source_start_frame + row["end_frame_exclusive"]
        finally:
            if reference:
                reference.close()
    for index, row in enumerate(rows):
        neighbors = [other["tone_ratio_db"] for j, other in enumerate(rows[max(0, index - 20):index + 21], max(0, index - 20))
                     if j != index and other["state"] == "active" and other["tone_ratio_db"] is not None]
        row["tonal_dip_hint_db"] = (float(np.median(neighbors)) - row["tone_ratio_db"]
                                    if len(neighbors) >= 10 and row["state"] == "active"
                                    and row["tone_ratio_db"] is not None else None)
    clips = []
    for key in ("speech_similarity_hint", "tonal_dip_hint_db"):
        count = 0
        for row in sorted((r for r in rows if r[key] is not None and r[key] > 0),
                          key=lambda r: (-r[key], r["start_frame"])):
            if len(clips) >= max_clips or count >= 4:
                break
            first = max(0, min(version.frames - 5 * RATE, row["start_frame"] - 2 * RATE))
            last = min(version.frames, first + 5 * RATE)
            if any(first < c["end_frame_exclusive"] and last > c["start_frame"] for c in clips):
                continue
            clips.append({**_clip(version, first, last), "hint": key, "score": row[key],
                          "flag_frame": row["start_frame"]})
            count += 1
    require(digest(path) == version.sha256 and (speech is None or
            digest(Path(directory) / speech.path) == speech.sha256), "Scan input changed.")
    output = _workspace(directory, job, "scan") if _output is None else jobs._plain(_output, directory=True)
    require(output.is_relative_to(jobs._plain(Path(directory), directory=True) / "runs"),
            "Scan output must belong to this job.")
    timeline = jobs._encode({"rows": rows}, jobs.MAX_METADATA_BYTES)
    jobs._publish(output / "timeline.json", timeline)
    report = {"schema_version": 1, "job_id": job.id, "operation": "scan", "outcome": "analyzed_only",
              "version": asdict(version), "speech_reference": asdict(speech) if speech else None,
              "sample_rate": RATE, "windows_analyzed": len(rows), "clips": clips,
              "timeline_sha256": hashlib.sha256(timeline).hexdigest(), "defects_confirmed": False,
              "listening_approved": False, "gpu_used": False,
              "method": "Half-second Hann spectra; absolute zero-lag 300-3400Hz speech similarity; tonal dips against +/-10s median (>=10 active neighbors). Top four per hint, speech first.",
              "limits": "Suggestions only, not exhaustive. Singing/instruments/arrangement can trigger hints. Missing hints do not certify clean audio. Tonal dips are not user-reported muffling. No automatic correction."}
    _receipt(output / "scan.json", report)
    return output / "scan.json"


def _write_audio(path, audio):
    require(bool(np.isfinite(audio).all()), "Nonfinite export.")
    with path.open("xb") as stream:
        sf.write(stream, audio, RATE, format="WAV", subtype="DOUBLE")
        stream.flush()
        os.fsync(stream.fileno())
    actual, rate = sf.read(path, dtype="float64", always_2d=True)
    require(rate == RATE and np.array_equal(actual, audio), "Export sample verification failed.")
    return digest(path)


def export_review_clips(directory: Path, scan_id: str) -> Path:
    """Export exact unadjusted excerpts from runs/<scan_id>/scan.json, report last."""
    jobs._id(scan_id)
    job = jobs.load_job(directory)
    scan_path = Path(directory) / "runs" / scan_id / "scan.json"
    if not scan_path.exists():
        scan_path = scan_path.parent / "worker" / "scan.json"
    report, scan_hash = jobs._read(scan_path, jobs.MAX_RECEIPT_BYTES)
    require(report.get("job_id") == job.id and report.get("operation") == "scan", "Foreign/invalid scan.")
    version = _version(job, report["version"]["id"])
    expected = asdict(version)
    recorded = dict(report["version"])
    # Listening changes do not invalidate immutable media or its saved shortlist.
    expected.pop("listening")
    recorded.pop("listening", None)
    require(recorded == expected and report.get("schema_version") == 1, "Scan version changed.")
    _, timeline_hash = jobs._read(scan_path.parent / "timeline.json", jobs.MAX_METADATA_BYTES)
    require(timeline_hash == report.get("timeline_sha256"), "Scan timeline changed.")
    clips = report.get("clips")
    require(type(clips) is list and len(clips) <= 8, "Invalid clip count.")
    previous = []
    for clip in clips:
        first, last = clip["start_frame"], clip["end_frame_exclusive"]
        jobs._integer(first, 0, version.frames - 1, "clip start")
        jobs._integer(last, first + 1, min(version.frames, first + 5 * RATE), "clip end")
        require(all(clip.get(k) == v for k, v in _clip(version, first, last).items()), "Invalid clip identity/map.")
        require(not any(first < b and last > a for a, b in previous), "Overlapping clips.")
        previous.append((first, last))
    output = _workspace(directory, job, "review_clips", sum(b-a for a,b in previous) * 16 + 8192)
    exports = []
    for clip in clips:
        audio, _ = sf.read(Path(directory) / version.path, start=clip["start_frame"],
                           stop=clip["end_frame_exclusive"], dtype="float64", always_2d=True)
        name = clip["id"] + ".wav"
        exports.append({**clip, "file": name, "sha256": _write_audio(output / name, audio)})
    require(digest(Path(directory) / version.path) == version.sha256, "Clip input changed.")
    _receipt(output / "clips.json", {"schema_version": 1, "job_id": job.id, "operation": "review_clips",
              "scan_id": scan_id, "scan_sha256": scan_hash, "clips": exports,
              "gain_db": 0, "encoding": "DOUBLE", "listening_approved": False})
    return output / "clips.json"


def export_comparison(directory: Path, versions: dict[str, str], *,
                      source_start_frame: int, source_end_frame: int) -> Path:
    """Compare explicitly role-labelled registered versions on canonical source frames.

    Allowed roles are original_mix (canonical only), earlier_separation,
    current_candidate, speech_reference, effects_reference. No legacy output changes.
    """
    job = jobs.load_job(directory)
    first, last = source_start_frame, source_end_frame
    jobs._integer(first, 0, jobs.MAX_FRAMES - 1, "comparison start")
    jobs._integer(last, first + RATE, min(jobs.MAX_FRAMES, first + 60 * RATE), "comparison end")
    roles = {"original_mix", "earlier_separation", "current_candidate", "speech_reference", "effects_reference"}
    require(type(versions) is dict and 2 <= len(versions) <= 5 and set(versions) <= roles, "Invalid comparison roles.")
    selected = {role: _version(job, identifier) for role, identifier in versions.items()}
    for role, version in selected.items():
        require(role != "original_mix" or version == job.versions[0], "Original mix must be canonical source.")
        require(version.source_start_frame <= first < last <= version.source_start_frame + version.frames,
                "Comparison range outside explicit map.")
    output = _workspace(directory, job, "comparison", len(selected) * (last-first) * 32 + 65536)
    (output / "unadjusted").mkdir()
    records = {}
    for role, version in selected.items():
        start = first - version.source_start_frame
        audio, _ = sf.read(Path(directory) / version.path, start=start, frames=last-first,
                           dtype="float64", always_2d=True)
        path = output / "unadjusted" / f"{role}.wav"
        sha = _write_audio(path, audio)
        records[role] = {"version_id": version.id, "parent_sha256": version.sha256,
                         "input_start_frame": start, "input_end_frame_exclusive": start + last-first,
                         "unadjusted_sha256": sha, "before": levels(path)}
    eligible = [r["before"] for role, r in records.items() if not role.endswith("_reference") and not r["before"]["quiet"]]
    target = min([-16.0] + [min(v["lufs"] + MAX_BOOST, v["lufs"] + PEAK_CEILING - v["true_peak_dbtp"])
                            for v in eligible]) if eligible else None
    reference_gain = min([0.0] + [PEAK_CEILING - r["before"]["true_peak_dbtp"] for role, r in records.items()
                                 if role.endswith("_reference") and r["before"]["true_peak_dbtp"] is not None])
    for role, record in records.items():
        before = record["before"]
        gain = (reference_gain if role.endswith("_reference") else
                (min(0.0, PEAK_CEILING - before["true_peak_dbtp"]) if before["true_peak_dbtp"] is not None else 0.0)
                if before["quiet"] else target - before["lufs"])
        audio, _ = sf.read(output / "unadjusted" / f"{role}.wav", dtype="float64", always_2d=True)
        path = output / f"{role}.wav"
        sha = _write_audio(path, audio * 10 ** (gain / 20))
        after = levels(path)
        require(after["frames"] == last-first and (after["true_peak_dbtp"] is None or
                after["true_peak_dbtp"] <= PEAK_CEILING + 0.05), "Comparison peak/alignment failure.")
        record.update(file=path.name, sha256=sha, gain_db=gain, after=after,
                      loudness_matched=not role.endswith("_reference") and not before["quiet"])
    require(all(digest(Path(directory) / v.path) == v.sha256 for v in selected.values()), "Comparison input changed.")
    _receipt(output / "comparison.json", {"schema_version": 1, "job_id": job.id, "operation": "comparison",
             "source_start_frame": first, "source_end_frame_exclusive": last, "sample_rate": RATE,
             "shared_target_lufs": target, "reference_gain_db": reference_gain, "peak_ceiling_dbtp": PEAK_CEILING,
             "max_boost_db": MAX_BOOST, "files": records, "listening_approved": False,
             "processing": "Constant gain only; quiet inputs never boosted; no EQ, fades or dynamic normalization."})
    return output / "comparison.json"
