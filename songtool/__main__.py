"""Independent CLI stages: download, extract, separate, master."""

import argparse
from pathlib import Path
import subprocess

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
    args = parser.parse_args()
    try:
        if args.command == "setup-model":
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


if __name__ == "__main__":
    main()
