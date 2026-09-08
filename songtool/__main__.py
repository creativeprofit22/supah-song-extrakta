"""Independent CLI stages: download, extract, separate, master."""

import argparse
from pathlib import Path
import subprocess
import json
import os
import sys
from dataclasses import asdict

from . import mastering, media, resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a trailer's music, then master it locally.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-model", help="Fetch pinned inference source and model weights")
    download = commands.add_parser("download", help="Save a YouTube video at up to 1080p")
    download.add_argument("url")
    download.add_argument("--output", type=Path, default=Path("outputs/source"))
    extract = commands.add_parser("extract", help="Decode a video or trim audio to 48 kHz float WAV")
    extract.add_argument("source", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--start", type=float, default=0, help="Start time in seconds")
    extract.add_argument("--end", type=float, help="End time in seconds")
    separate = commands.add_parser("separate", help="Split music, speech, and sound effects")
    separate.add_argument("source", type=Path)
    separate.add_argument("output", type=Path, help="New directory for stems")
    separate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    offset_mode = separate.add_mutually_exclusive_group()
    offset_mode.add_argument("--window-offset", action="store_true", help="Opt-in two-grid excerpt experiment")
    offset_mode.add_argument("--full-song-offset", action="store_true",
                             help="Apply the preferred one-second offset grid across the full source")
    separate.add_argument("--start", type=float, help="Required with --window-offset, source seconds")
    separate.add_argument("--end", type=float, help="Required with --window-offset, source seconds")
    master = commands.add_parser("master", help="Create a 24-bit WAV, 320 kbps MP3, and measurements")
    master.add_argument("source", type=Path)
    master.add_argument("output", type=Path)
    compare = commands.add_parser("compare", help="Export aligned, fixed-gain listening references")
    compare.add_argument("output", type=Path, help="Fresh comparison directory")
    compare.add_argument("--source", type=Path, default=Path("outputs/audio/trailer.wav"))
    compare.add_argument("--stems", type=Path, default=Path("outputs/stems"))
    compare.add_argument("--master", type=Path, default=Path("outputs/final/song-master.wav"))
    compare.add_argument("--start", type=float, required=True)
    compare.add_argument("--end", type=float, required=True)
    compare.add_argument("--no-open", action="store_true", help="Do not open the comparison folder")
    compare.add_argument("--experiment", type=Path, help="Include candidates from a matching completed experiment")
    finish = commands.add_parser("finish-offset", help="Apply an approved offset excerpt, leaving the rest unchanged")
    finish.add_argument("comparison", type=Path, help="Completed candidate comparison directory")
    finish.add_argument("output", type=Path, help="Fresh final export directory")
    finish.add_argument("--approved", action="store_true", required=True, help="Confirm listening preference for offset_music")
    cleanup = commands.add_parser("cleanup-preview", help="One guarded full-song candidate; no promotion or playback")
    cleanup.add_argument("output", type=Path, help="Fresh directory for the recorded baseline's fixed cleanup recipe")
    _job_parser(commands)
    args = parser.parse_args()
    try:
        if args.command == "job":
            _job_command(args)
        elif args.command == "setup-model":
            resources.setup()
        elif args.command == "download":
            media.download(args.url, args.output)
        elif args.command == "extract":
            media.extract(args.source, args.output, args.start, args.end)
        elif args.command == "separate":
            if args.window_offset:
                if args.start is None or args.end is None:
                    raise ValueError("--window-offset requires --start and --end.")
                from .separation import experiment
                experiment(args.source, args.output, args.start, args.end, args.device)
            else:
                if args.start is not None or args.end is not None:
                    raise ValueError("--start/--end require --window-offset; default separation is unchanged.")
                from .separation import separate as separate_audio
                separate_audio(args.source, args.output, args.device, full_song_offset=args.full_song_offset)
        elif args.command == "master":
            mastering.master(args.source, args.output)
        elif args.command == "compare":
            from .comparison import compare as compare_audio
            compare_audio(args.source, args.stems, args.master, args.output,
                          args.start, args.end, open_folder=not args.no_open, experiment=args.experiment)
        elif args.command == "cleanup-preview":
            from .cleanup import preview
            preview(args.output)
        elif args.command == "finish-offset":
            from .comparison import finish_offset
            finish_offset(args.comparison, args.output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")


def _job_parser(commands):
    group = commands.add_parser("job", help="Explicit isolated audio jobs")
    actions = group.add_subparsers(dest="job_command", required=True)
    create = actions.add_parser("create")
    create.add_argument("source", type=Path)
    create.add_argument("directory", type=Path)
    create.add_argument("--intent", choices=("music", "spoken_audio", "unknown"), required=True)
    create.add_argument("--wanted-vocals-may-include-rap", action="store_true")
    blind = actions.add_parser("blind-ab", help="Read-only, memory-only blind A/B preview; no approval")
    blind.add_argument("directory", type=Path)
    blind.add_argument("--version-one", required=True, help="Exact registered ID, not current")
    blind.add_argument("--version-two", required=True, help="Distinct exact registered ID")
    blind.add_argument("--source-start-frame", required=True, type=int, help="Inclusive canonical-source 48 kHz frame")
    blind.add_argument("--source-end-frame", required=True, type=int, help="Exclusive frame; 1–2,880,000 selected frames")
    for name in ("status", "scan", "feedback", "run", "choose", "open", "recover", "copy"):
        command = actions.add_parser(name)
        command.add_argument("directory", type=Path)
        if name == "status":
            command.add_argument("--json", action="store_true")
        if name in ("scan", "run", "choose", "open"):
            command.add_argument("--version", required=True)
        if name in ("scan", "run"):
            command.add_argument("--timeout", type=int)
            command.add_argument("--retry-reason")
            command.add_argument("--speech-version-id")
        if name == "scan":
            command.add_argument("--clips", action="store_true")
        if name == "run":
            command.add_argument("--operation", required=True,
                                 choices=("trim", "scan", "bandit", "gentle-denoise", "local-eq"))
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--start-frame", type=int)
            command.add_argument("--end-frame", type=int)
            command.add_argument("--eq-interval", nargs=2, type=int, action="append", metavar=("START", "END"))
            command.add_argument("--parent-sha256")
            command.add_argument("--full-song-offset", action="store_true")
        if name in ("choose", "feedback"):
            command.add_argument("--note", required=True)
            choices = ("better", "worse", "unconfirmed") if name == "choose" else (
                "good", "residual_dialogue", "wanted_vocal_loss", "muffling", "warbling_reverse_like_artifact", "noise", "uncertain")
            command.add_argument("--verdict", choices=choices, required=True)
        if name == "feedback":
            command.add_argument("--clip", required=True, help="Stable clip id from the scan or clips report")
            command.add_argument("--version", metavar="ID",
                                 help="Exact version_id from that clip; required when its ID matches multiple versions")
            command.add_argument("--accepted", action="store_true")
        if name == "recover":
            command.add_argument("--run", required=True)
        if name == "copy":
            command.add_argument("destination", type=Path)


def _status(directory):
    from . import jobs, workflow
    job = jobs.load_job(directory)
    result = workflow.summarize_job(directory)
    result["listening_scope"] = "whole_version" if result["listening"] != "unreviewed" else "none"
    result["versions"] = [asdict(v) for v in job.versions]
    result["runs"] = [asdict(r) for r in job.runs]
    result["current_technical"] = next(v.technical for v in job.versions if v.id == job.current_version)
    latest = next((r for r in job.runs if r.id == job.latest_attempt), None)
    result["outcome"] = None
    if latest and latest.receipt_sha256:
        receipt, sha = jobs._read(Path(directory) / "runs" / latest.id / "receipt.json", jobs.MAX_RECEIPT_BYTES)
        jobs.require(sha == latest.receipt_sha256, "Run receipt mismatch.")
        result["outcome"] = receipt.get("outcome")
    return result


def _feedback_clip(args):
    from . import jobs, review
    job = jobs.load_job(args.directory)
    jobs._id(args.clip)
    if args.version is not None:
        jobs._id(args.version)
        jobs.require(any(v.id == args.version for v in job.versions), "Unknown feedback version ID in this job.")
    found = None
    root = jobs._plain(args.directory / "runs", directory=True)
    count = 0
    for entry in root.iterdir():
        count += 1
        jobs.require(count <= 1024, "Too many review directories.")
        jobs._plain(entry, directory=True)
        for path in (entry / "scan.json", entry / "worker" / "scan.json"):
            if not path.exists():
                continue
            report, _ = jobs._read(path, jobs.MAX_RECEIPT_BYTES)
            jobs.require(report.get("job_id") == job.id and report.get("operation") == "scan"
                         and report.get("schema_version") == 1, "Foreign scan.")
            _, timeline_sha = jobs._read(path.parent / "timeline.json", jobs.MAX_METADATA_BYTES)
            jobs.require(timeline_sha == report.get("timeline_sha256"), "Scan timeline mismatch.")
            version = next((v for v in job.versions if v.id == report["version"]["id"]), None)
            jobs.require(version is not None and version.sha256 == report["version"]["sha256"], "Stale scan.")
            clips = report.get("clips")
            jobs.require(type(clips) is list and len(clips) <= 8, "Invalid clips.")
            for clip in clips:
                if clip.get("id") != args.clip:
                    continue
                first, last = clip["start_frame"], clip["end_frame_exclusive"]
                jobs._integer(first, 0, version.frames - 1, "clip start")
                jobs._integer(last, first + 1, version.frames, "clip end")
                expected = review._clip(version, first, last)
                jobs.require(all(clip.get(k) == v for k, v in expected.items()), "Invalid stable clip map.")
                if args.version is not None and clip["version_id"] != args.version:
                    continue
                jobs.require(found is None or found == expected,
                             "Ambiguous clip ID across versions; pass --version ID using the clip's version_id from the scan or clips report.")
                found = expected
    jobs.require(found is not None, "Unknown stable clip ID for the selected version." if args.version is not None
                 else "Unknown stable clip ID.")
    return jobs.record_feedback(args.directory, version_id=found["version_id"], category=args.verdict,
        note=args.note, scope="interval", start_frame=found["start_frame"],
        end_frame=found["end_frame_exclusive"], accepted=args.accepted,
        expected_revision=job.revision, expected_revision_sha256=job.revision_sha256)


def _job_command(args):
    from . import jobs, workflow, review
    action = args.job_command
    if action == "blind-ab":
        from .blind_ab import prepare_session, serve_session
        session = prepare_session(args.directory, args.version_one, args.version_two,
                                  source_start_frame=args.source_start_frame,
                                  source_end_frame=args.source_end_frame)
        serve_session(session)
        return
    if action == "create":
        job = jobs.create_job(args.source, args.directory, intent=args.intent,
                             wanted_vocals_may_include_rap=args.wanted_vocals_may_include_rap)
        result = {"job_id": job.id, "current_version": job.current_version,
                  "outcome": "rendered_new", "operation": "canonical_conversion", "listening": "unreviewed"}
    elif action == "status":
        result = _status(args.directory)
        if not args.json:
            for key, value in result.items():
                print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            return
    elif action in ("run", "scan"):
        operation = "scan" if action == "scan" else args.operation
        parameters = {}
        for flag, key in (("start_frame", "start_frame"), ("end_frame", "end_frame"),
                          ("eq_interval", "intervals"), ("parent_sha256", "parent_sha256"),
                          ("speech_version_id", "speech_version_id")):
            value = getattr(args, flag, None)
            if value is not None:
                parameters[key] = value
        if getattr(args, "full_song_offset", False):
            parameters["full_song_offset"] = True
        result = workflow.run_operation(args.directory, operation, args.version, parameters=parameters,
            device=getattr(args, "device", "cpu"), timeout=args.timeout, retry_reason=args.retry_reason)
        if action == "scan" and args.clips and result["execution"] == "completed":
            result["clips_report"] = str(review.export_review_clips(args.directory, result["run"]["id"]))
    elif action == "open":
        version, path = jobs.resolve_version(args.directory, args.version)
        job = jobs.load_job(args.directory)
        run = next((r for r in job.runs if r.id == version.run_id), None)
        result = {"outcome": "opened_existing", "version": version.id, "path": str(path),
                  "execution": run.execution if run else "completed", "technical": version.technical,
                  "listening": version.listening, "listening_scope": "whole_version" if version.listening != "unreviewed" else "none",
                  "rendered_now": False}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)], check=True, shell=False)
        return
    else:
        if action == "feedback":
            job = _feedback_clip(args)
        elif action == "choose":
            job = jobs.select_version(args.directory, args.version, verdict=args.verdict, note=args.note)
        elif action == "recover":
            job = jobs.recover_run(args.directory, args.run)
        else:
            job = jobs.copy_job(args.directory, args.destination)
        result = {"job_id": job.id, "revision": job.revision, "current_version": job.current_version,
                  "action": action, "rendered_now": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
