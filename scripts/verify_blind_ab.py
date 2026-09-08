"""Focused synthetic checks only: no import worker, export, model or playback."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from http.client import HTTPConnection
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import soundfile as sf
from songtool import blind_ab as ab, jobs, review, workflow, runtime, resources


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob("*")}


def rejected(call):
    try:
        call()
    except (ValueError, OSError, RuntimeError):
        return
    raise AssertionError("Invalid selection accepted")


def fixture(root):
    """Bootstrap a labelled synthetic canonical fixture, then use real registration."""
    root.mkdir()
    for name in ("source", "state", "versions", "runs"):
        (root / name).mkdir()
    samples = np.arange(256, dtype=np.float64).reshape(128, 2) / 1024
    original = root / "source/original.wav"
    sf.write(original, samples, 48000, subtype="FLOAT")
    source = jobs.Source("source/original.wav", ab.digest(original), original.stat().st_size,
                         128 / 48000, 48000, 2, "pcm_f32le", "wav")
    identifier, version_id = jobs.new_id(), jobs.new_id()
    path = root / f"versions/{version_id}/audio.wav"
    path.parent.mkdir()
    sf.write(path, samples, 48000, subtype="FLOAT")
    version = jobs.Version(version_id, path.relative_to(root).as_posix(), ab.digest(path),
                           128, "FLOAT", "canonical", technical="passed")
    version = jobs._version_receipt(root, identifier, source.sha256, version,
                                    {"operation": "synthetic_fixture", "listening_approved": False})
    job = jobs.Job(identifier, source, versions=(version,), current_version=version.id, revision=1)
    jobs._publish(root / "state/00000001.json", jobs._encode(jobs._data(job), jobs.MAX_METADATA_BYTES))
    jobs.load_job(root)
    for start, data in ((7, samples[7:110]), (19, -samples[19:120]), (7, samples[7:110])):
        audio = root.parent / f"fixture-{len(job.versions)}.wav"
        sf.write(audio, data, 48000, subtype="DOUBLE")
        job = jobs.register_version(root, audio, parent_id=version.id, parent_start_frame=start,
                                   expected_revision=job.revision,
                                   expected_revision_sha256=jobs.load_job(root).revision_sha256)
    return job, samples


def checks():
    with tempfile.TemporaryDirectory(prefix="blind-ab-synthetic-") as temp:
        root = Path(temp) / "job"
        job, samples = fixture(root)
        one, two, identical = job.versions[1:]
        before = snapshot(root)
        def prepare(a=one.id, b=two.id, first=23, last=37):
            return ab.prepare_session(root, a, b, source_start_frame=first, source_end_frame=last)
        with ExitStack() as stack:
            for module, names in ((jobs, ("register_version", "commit_revision", "_publish", "record_feedback", "select_version", "create_job")),
                                  (review, ("export_comparison", "export_review_clips")),
                                  (workflow, ("run_operation",)), (runtime, ("run_owned",)),
                                  (resources, ("setup", "validate_resources"))):
                for name in names:
                    stack.enter_context(patch.object(module, name, side_effect=AssertionError(f"Forbidden: {name}")))
            for assignment in (0, 1):
                with patch.object(ab.secrets, "randbelow", return_value=assignment):
                    session = prepare()
                expected = (samples[23:37], -samples[23:37])[::1 if assignment == 0 else -1]
                for payload, data in zip(session.pcm, expected):
                    assert payload == data.astype("<f4").tobytes()
                for first, last in ((23, 24), (109, 110)):
                    bounded = prepare(first=first, last=last)
                    assert all(len(p) == 8 for p in bounded.pcm)
                    for v, p in zip(bounded.versions, bounded.pcm):
                        assert p == (samples[first:last] * (1 if v.id == one.id else -1)).astype("<f4").tobytes()
                logs = io.StringIO()
                with redirect_stderr(logs), HTTPServer(("127.0.0.1", 0), ab.make_handler(session)) as server:
                    thread = threading.Thread(target=server.serve_forever)
                    thread.start()
                    port = server.server_port
                    prefix = f"/{session.capability}/"
                    def fetch(route="", method="GET", headers=None, body=None, absolute=False):
                        conn = HTTPConnection("127.0.0.1", port, timeout=10)
                        try:
                            conn.request(method, route if absolute else prefix + route, body=body, headers=headers or {})
                            response = conn.getresponse()
                            return response.status, dict(response.getheaders()), response.read()
                        finally:
                            conn.close()
                    try:
                        identities = [v.id for v in job.versions] + [v.sha256 for v in job.versions] + [str(root), one.role]
                        for route in ("", "blind_ab.js", "session", "A", "B"):
                            code, headers, body = fetch(route)
                            assert code == 200
                            assert headers["Cache-Control"] == "no-store"
                            assert headers["X-Content-Type-Options"] == "nosniff"
                            assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
                            # PCM inherently permits fingerprinting; inspect only its headers for identity.
                            public = str(headers) + (body.decode() if route not in ("A", "B") else "")
                            for identity in identities:
                                assert identity not in public, (route, identity)
                            if route in ("A", "B"):
                                assert body == session.pcm[0 if route == "A" else 1] and len(body) == 14 * 8
                        assert json.loads(fetch("session")[2])["revealed"] is False
                        for route, method, headers, body, absolute in (
                            ("reveal", "GET", {}, None, False),
                            ("reveal", "POST", {}, None, False),
                            ("reveal", "POST", {"Origin": "http://foreign.invalid"}, None, False),
                            ("A", "GET", {"Host": "foreign.invalid"}, None, False),
                            ("A", "PUT", {}, None, False),
                            ("A", "HEAD", {}, None, False),
                            ("A", "OPTIONS", {}, None, False),
                            ("../A", "GET", {}, None, False),
                            ("%2e%2e/A", "GET", {}, None, False),
                            ("A?x=1", "GET", {}, None, False),
                            ("/wrong/A", "GET", {}, None, True),
                            ("reveal", "POST", {"Origin": f"http://127.0.0.1:{port}"}, b"x", False)):
                            code, headers_out, payload = fetch(route, method, headers, body, absolute)
                            assert code >= 400 and all(v.id.encode() not in payload for v in job.versions)
                            assert not session.revealed
                        origin = {"Origin": f"http://127.0.0.1:{port}"}
                        revealed = fetch("reveal", "POST", origin)
                        assert revealed[0] == 200 and json.loads(revealed[2]) == session.identities()
                        for slot, v in zip(("A", "B"), session.versions):
                            mapping = json.loads(revealed[2])[slot]
                            assert mapping["version_id"] == v.id and mapping["sha256"] == v.sha256
                            assert mapping["local_start_frame"] == 23 - v.source_start_frame
                            assert mapping["local_end_frame"] == 37 - v.source_start_frame
                        assert fetch("reveal", "POST", origin)[2] == revealed[2]
                        assert fetch()[0] == 200 and json.loads(fetch("session")[2])["revealed"] is True
                    finally:
                        server.shutdown()
                        thread.join(timeout=10)
                        assert not thread.is_alive()
                assert not logs.getvalue()
                with socket.socket() as probe:
                    assert probe.connect_ex(("127.0.0.1", port)) != 0
            control = prepare(b=identical.id)
            assert control.pcm[0] == control.pcm[1]
            for a, b in ((None, two.id), ("", two.id), ("current", two.id), ("f" * 32, two.id),
                         (one.id, one.id), (one.id, "current")):
                rejected(lambda: prepare(a, b))
            for first, last in ((23, 23), (24, 23), (-1, 25), (0, 2880001), (18, 25), (109, 111), (True, 25)):
                rejected(lambda: prepare(first=first, last=last))
            with patch.object(ab, "digest", return_value="0" * 64):
                rejected(prepare)
            real_file = sf.SoundFile
            class BrokenRead:
                def __init__(self, path):
                    self.file = real_file(path)
                def __enter__(self): return self
                def __exit__(self, *_): self.file.close()
                def __getattr__(self, name): return getattr(self.file, name)
                def read(self, *args, **kwargs): return bad_samples
            for bad_samples in (np.zeros((13, 2)), np.full((14, 2), np.nan)):
                with patch.object(ab.sf, "SoundFile", BrokenRead):
                    rejected(prepare)
            shutdown_session = prepare()
            output = io.StringIO()
            with redirect_stdout(output), patch.object(HTTPServer, "serve_forever", side_effect=KeyboardInterrupt):
                ab.serve_session(shutdown_session)
            assert shutdown_session.pcm == () and shutdown_session.versions == ()
            assert all(v.id not in output.getvalue() and v.sha256 not in output.getvalue() for v in job.versions)
            assert snapshot(root) == before
        # Corruption is isolated to disposable copies, not the immutable fixture under test.
        import shutil
        for kind in ("hash", "metadata", "missing"):
            copied = Path(temp) / kind
            shutil.copytree(root, copied)
            path = copied / one.path
            if kind == "hash":
                with path.open("r+b") as stream:
                    stream.seek(-1, 2); stream.write(b"x")
            elif kind == "metadata":
                sf.write(path, np.zeros((10, 1)), 44100, subtype="FLOAT")
            else:
                path.unlink()
            rejected(lambda: ab.prepare_session(copied, one.id, two.id, source_start_frame=23, source_end_frame=37))
        assert snapshot(root) == before
    print("PASS: synthetic alignment, validation, HTTP concealment/reveal, loopback boundaries, immutability, shutdown.")


def browser_server():
    """Only three public test assets; never expose job directories or play real audio."""
    assets = {"/": ROOT / "scripts/verify_blind_ab.html",
              "/blind_ab.html": ROOT / "songtool/blind_ab.html",
              "/blind_ab.js": ROOT / "songtool/blind_ab.js"}
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)
        def log_message(self, *_): pass
        def do_GET(self):
            if self.path not in assets:
                self.send_error(404); return
            body = assets[self.path].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript" if self.path.endswith(".js") else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(f"Synthetic speaker-free browser checks: http://127.0.0.1:{server.server_port}/", flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass


if __name__ == "__main__":
    browser_server() if sys.argv[1:] == ["--browser"] else checks()
