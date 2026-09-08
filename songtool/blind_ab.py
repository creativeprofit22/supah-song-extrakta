"""Temporary, read-only, private-loopback A/B preview. No listening verdicts."""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf

from . import jobs
from .cleanup import digest, require

RATE = 48_000
# simplification: bounded 60-second RAM preview; longer scopes need a streaming design.
MAX_PREVIEW_FRAMES = 60 * RATE


@dataclass
class Session:
    first: int
    last: int
    versions: tuple[jobs.Version, ...]
    pcm: tuple[bytes, ...]
    capability: str
    revealed: bool = False

    def metadata(self) -> dict:
        return {"source_start_frame": self.first, "source_end_frame": self.last,
                "frames": self.last - self.first, "sample_rate": RATE, "channels": 2,
                "loading": False, "revealed": self.revealed}

    def identities(self) -> dict:
        return {slot: {"version_id": v.id, "sha256": v.sha256,
                       "source_start_frame": self.first, "source_end_frame": self.last,
                       "version_source_start_frame": v.source_start_frame,
                       "local_start_frame": self.first - v.source_start_frame,
                       "local_end_frame": self.last - v.source_start_frame}
                for slot, v in zip(("A", "B"), self.versions)}


def prepare_session(directory: Path, version_one: str, version_two: str, *,
                    source_start_frame: int, source_end_frame: int) -> Session:
    """Verify history and copy only the exact mapped samples into anonymous RAM."""
    jobs._id(version_one)
    jobs._id(version_two)
    require(version_one != version_two, "Select two distinct exact version IDs.")
    first, last = source_start_frame, source_end_frame
    jobs._integer(first, 0, jobs.MAX_FRAMES - 1, "source start frame")
    jobs._integer(last, first + 1, jobs.MAX_FRAMES, "source end frame")
    require(last - first <= MAX_PREVIEW_FRAMES, "Preview limit is 2,880,000 frames (60 seconds).")
    try:
        root = jobs._plain(directory, directory=True)
        job = jobs.load_job(root)
        selected = tuple(next(v for v in job.versions if v.id == identifier)
                         for identifier in (version_one, version_two))
        for v in selected:
            require(v.source_start_frame <= first < last <= v.source_start_frame + v.frames,
                    "Selected scope is not covered by both source maps.")
        payloads = []
        for v in selected:
            path = jobs._plain(jobs._internal(root, v.path))
            with sf.SoundFile(path) as audio:
                require(audio.samplerate == RATE and audio.channels == 2
                        and audio.frames == v.frames and audio.subtype == v.subtype,
                        "Invalid preview media format.")
                audio.seek(first - v.source_start_frame)
                samples = audio.read(last - first, dtype="float32", always_2d=True)
            require(samples.shape == (last - first, 2) and np.isfinite(samples).all(),
                    "Invalid preview samples.")
            require(digest(path) == v.sha256, "Preview media changed.")
            payloads.append(samples.astype("<f4", copy=False).tobytes())
        if secrets.randbelow(2):
            selected = selected[::-1]
            payloads.reverse()
        return Session(first, last, selected, tuple(payloads), secrets.token_urlsafe(32))
    except (OSError, ValueError, RuntimeError, StopIteration):
        raise ValueError("Cannot prepare preview: check exact IDs, source coverage and verified media.") from None


def make_handler(session: Session):
    page = files("songtool").joinpath("blind_ab.html").read_bytes()
    script = files("songtool").joinpath("blind_ab.js").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):
            pass

        def send_error(self, code, message=None, explain=None):
            self.reply(code, b"Request refused.", "text/plain")

        def reply(self, code, body, content_type):
            self.send_response_only(code)
            for key, value in {
                "Content-Type": content_type, "Content-Length": str(len(body)),
                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer", "Connection": "close",
                "Content-Security-Policy": "default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            }.items():
                self.send_header(key, value)
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(body)
            except (ConnectionError, TimeoutError):
                pass  # A closed browser must not produce a request traceback.

        def dispatch(self):
            host = f"127.0.0.1:{self.server.server_address[1]}"
            prefix = f"/{session.capability}/"
            if (self.headers.get_all("Host") != [host]
                    or self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get_all("Content-Length", []) not in ([], ["0"])
                    or not self.path.startswith(prefix)):
                return self.send_error(403)
            route = self.path[len(prefix):]
            if self.command == "POST":
                if route != "reveal" or self.headers.get_all("Origin") != [f"http://{host}"]:
                    return self.send_error(403)
                session.revealed = True
                return self.reply(200, json.dumps(session.identities()).encode(), "application/json")
            if self.command != "GET":
                return self.send_error(405)
            if route == "":
                return self.reply(200, page, "text/html; charset=utf-8")
            if route == "blind_ab.js":
                return self.reply(200, script, "text/javascript; charset=utf-8")
            if route == "session":
                return self.reply(200, json.dumps(session.metadata()).encode(), "application/json")
            if route in ("A", "B"):
                return self.reply(200, session.pcm[0 if route == "A" else 1], "application/octet-stream")
            return self.send_error(404)

        do_GET = dispatch
        do_POST = dispatch

    return Handler


def serve_session(session: Session) -> None:
    with HTTPServer(("127.0.0.1", 0), make_handler(session)) as server:
        print(f"http://127.0.0.1:{server.server_port}/{session.capability}/", flush=True)
        print(f"Source frames [{session.first}, {session.last}) at 48 kHz. CPU; no render/worker/model.")
        print("Open the URL manually. No autoplay. Ctrl+C shuts down this memory-only session.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            session.pcm = ()
            session.versions = ()
