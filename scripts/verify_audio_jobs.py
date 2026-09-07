"""Portable synthetic workflow checks; no fixtures, inference, downloads or playback.

Run from the repository root with Python 3.12 and the existing project deps plus
FFmpeg. All destructive/injected failures are confined to TemporaryDirectory.
Private APIs are used only to verify publication boundaries and inject failures.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set before numerical imports, including in controlled subprocesses.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

from songtool import jobs, review, runtime, workflow
import numpy as np
import soundfile as sf


def rejected(call, exceptions=(ValueError, OSError, jobs.RevisionConflict), contains=None):
    try:
        call()
    except exceptions as error:
        if contains:
            assert contains.lower() in str(error).lower(), str(error)
        return
    raise AssertionError("Unsafe action was accepted")


def cli(*args, ok=True, raw=False):
    command = [sys.executable, "-B", "-m", "songtool", *map(str, args)]
    if args[:2] == ("job", "open"):
        # Even a hash-check regression must never launch real playback.
        bootstrap = """import os,runpy,sys
from unittest.mock import patch
sys.argv = ['songtool', *sys.argv[1:]]
with patch('os.startfile' if os.name == 'nt' else 'subprocess.run'):
    runpy.run_module('songtool', run_name='__main__')
"""
        command = [sys.executable, "-B", "-c", bootstrap, *map(str, args)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
    return result.stdout if raw or not ok else json.loads(result.stdout)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(root):
    return {p.name: p.read_bytes() for p in (root / "state").glob("*.json")}


def register(root, audio, **kwargs):
    job = jobs.load_job(root)
    return jobs.register_version(root, audio, parent_id=job.versions[0].id,
        parent_start_frame=0, expected_revision=job.revision,
        expected_revision_sha256=job.revision_sha256, **kwargs)


def child_mode(mode, root):
    """Real independent writers/controllers, with deterministic boundary injection."""
    if mode == "writer":
        job = jobs.load_job(root)
        (root / f"ready-{os.getpid()}").touch()
        deadline = time.monotonic() + 15
        while not (root / "go").exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        try:
            jobs.commit_revision(root, job)
        except jobs.RevisionConflict:
            return 23
        return 0
    if mode == "interrupt":
        with patch.object(runtime, "run_owned", side_effect=lambda *a, **k: os._exit(25)):
            workflow.run_operation(root, "trim", parameters={"start_frame": 500, "end_frame": 60000})
        raise AssertionError("Interruption injection was not reached")
    if mode == "crash":
        original = workflow._publish
        def publish(path, value, *args, **kwargs):
            sha = original(path, value, *args, **kwargs)
            if path.name == "receipt.json":
                (root / "receipt-ready").touch()
                deadline = time.monotonic() + 20
                while not (root / "crash-now").exists():
                    assert time.monotonic() < deadline
                    time.sleep(.01)
                os._exit(24)
            return sha
        with patch.object(workflow, "_publish", publish):
            workflow.run_operation(root, "trim", parameters={"start_frame": 321, "end_frame": 60000})
        raise AssertionError("Crash injection was not reached")
    raise AssertionError(mode)


def wait_for(predicate, process=None):
    deadline = time.monotonic() + 20
    while not predicate():
        if process is not None:
            assert process.poll() is None, "Controlled child exited before boundary"
        assert time.monotonic() < deadline, "Controlled child did not reach boundary"
        time.sleep(.01)


def spawn(mode, root):
    return subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                             "--child", mode, str(root)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(process, code):
    out, err = process.communicate(timeout=25)
    assert process.returncode == code, (process.returncode, out, err)


def integrity_checks(root, temp):
    original = snapshot(root)
    state = root / "state" / "00000001.json"
    saved = state.read_bytes()
    malformed = [b"{", b" " * (jobs.MAX_METADATA_BYTES + 1),
                 b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}']
    data = json.loads(saved)
    data["source"]["path"] = "../outside.wav"
    malformed.append(json.dumps(data).encode())
    for payload in malformed:
        try:
            state.write_bytes(payload)
            rejected(lambda: jobs.load_job(root))
        finally:
            state.write_bytes(saved)
    parent = root / jobs.load_job(root).versions[0].path
    audio = parent.read_bytes()
    try:
        parent.write_bytes(audio[:-1] + bytes([audio[-1] ^ 1]))
        rejected(lambda: jobs.resolve_version(root), contains="hash")
        cli("job", "open", root, "--version", "current", ok=False)
    finally:
        parent.write_bytes(audio)
    # A real link/reparse test, including Windows without symlink privileges.
    link = temp / "linked-job"
    if os.name == "nt":
        command = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(root)],
                                 capture_output=True, text=True)
        assert command.returncode == 0, command.stderr
    else:
        link.symlink_to(root, target_is_directory=True)
    try:
        rejected(lambda: jobs.load_job(link), contains="link")
    finally:
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()
    for target, error in (("fsync", OSError("injected flush failure")),
                          ("link", OSError("injected publication failure"))):
        with patch.object(jobs.os, target, side_effect=error):
            rejected(lambda: jobs.commit_revision(root, jobs.load_job(root)))
        assert snapshot(root) == original
        assert jobs.load_job(root).revision == 1
    # Fail during the write itself, before flush/link; retain prior snapshots.
    original_open = Path.open
    class FailedWrite:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def write(self, data):
            self.stream.write(data[:17])
            raise OSError("injected partial write")
    def open_injected(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailedWrite(stream) if path.name.startswith(".pending-") and args == ("xb",) else stream
    with patch.object(Path, "open", open_injected):
        rejected(lambda: jobs.commit_revision(root, jobs.load_job(root)), contains="partial write")
    assert snapshot(root) == original
    rejected(lambda: jobs._publish(state, b"must not overwrite"), exceptions=(FileExistsError,))
    assert state.read_bytes() == saved
    # Interrupted staged bytes do not constitute a commit.
    (root / "state" / ".pending-injected").write_bytes(b"partial")
    assert jobs.load_job(root).revision == 1
    committed = jobs.commit_revision(root, jobs.load_job(root))
    rejected(lambda: jobs.commit_revision(root, replace(committed, revision_sha256="0" * 64)),
             exceptions=(jobs.RevisionConflict,))
    second = root / "state" / "00000002.json"
    second.rename(root / "state" / "00000003.json")
    try:
        rejected(lambda: jobs.load_job(root), contains="revision")
    finally:
        (root / "state" / "00000003.json").rename(second)
    assert jobs.load_job(root).previous_revision_sha256 == jobs.digest(state)
    # Two processes load the same snapshot before release. Exactly one wins.
    writers = [spawn("writer", root) for _ in range(2)]
    try:
        wait_for(lambda: len(list(root.glob("ready-*"))) == 2)
        (root / "go").touch()
        results = []
        for process in writers:
            out, err = process.communicate(timeout=20)
            assert process.returncode in (0, 23), (out, err)
            results.append(process.returncode)
        assert sorted(results) == [0, 23]
        assert jobs.load_job(root).revision == committed.revision + 1
    finally:
        for process in writers:
            if process.poll() is None:
                process.kill()
                process.wait()
        for p in [*root.glob("ready-*"), root / "go"]:
            p.unlink(missing_ok=True)


def tree_checks(temp):
    # Worker records both identities before cancellation; never target guessed PIDs.
    code = '''import json,os,subprocess,sys,time
from pathlib import Path
from songtool.runtime import process_identity
p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps([[os.getpid(),process_identity(os.getpid())],[p.pid,process_identity(p.pid)]]))
Path(sys.argv[1]+'.ready').touch()
time.sleep(60)
'''
    for mode in ("timeout", "cancel"):
        identity = temp / f"{mode}-identities.json"
        event = threading.Event()
        errors = []
        def cancel_when_ready():
            try:
                wait_for(lambda: Path(str(identity) + ".ready").exists())
                event.set()
            except BaseException as error:
                errors.append(error)
                event.set()
        thread = threading.Thread(target=cancel_when_ready) if mode == "cancel" else None
        if thread:
            thread.start()
        try:
            result = runtime.run_owned([sys.executable, "-B", "-c", code, str(identity)],
                temp / f"tree-{mode}", workspace=temp,
                policy=replace(jobs.ResourcePolicy(), wall_time_seconds=2 if mode == "timeout" else 10,
                               max_output_bytes=8 * 1024**2), cancel=event)
        finally:
            if thread:
                thread.join(timeout=21)
                assert not thread.is_alive()
        assert not errors, errors
        assert result.status == ("timed_out" if mode == "timeout" else "cancelled"), result
        assert result.telemetry == result.inference_ran == "unavailable"
        assert not runtime.process_matches(result.process_id, result.process_identity)
        identities = read(identity)
        assert len(identities) == 2
        for pid, creation in identities:
            assert creation and not runtime.process_matches(pid, creation), (mode, pid)


def legacy_evidence_checks(temp, source, unrelated):
    root = temp / "legacy-job"
    job = jobs.create_job(source, root, intent="music")
    parent = job.versions[0]
    parameters = {"recipe_sha256": "a" * 64, "start_frame": 0, "end_frame": parent.frames}
    evidence = temp / "historical-failure.json"
    data = {"source_sha256": job.source.sha256, "parent_sha256": parent.sha256,
            "failure_reason": "Original guard failure — do not repeat.", "parameters": parameters}
    evidence.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    original = evidence.read_bytes()
    args = dict(expected_evidence_sha256=jobs.digest(evidence), parent_id=parent.id,
                parent_sha256=parent.sha256, source_sha256=job.source.sha256,
                failure_reason=data["failure_reason"], operation=workflow.LEGACY_OPERATION,
                parameters=parameters, evidence_mapping={key: [key] for key in data},
                attestation="Synthetic historical evidence; no historical execution verified.",
                expected_revision=job.revision, expected_revision_sha256=job.revision_sha256)
    before = snapshot(root)
    rejected(lambda: workflow.adopt_failed_evidence(root, evidence, **{**args, "parent_sha256": "0" * 64}))
    rejected(lambda: workflow.adopt_failed_evidence(root, evidence, **{**args, "expected_evidence_sha256": "0" * 64}))
    rejected(lambda: workflow.adopt_failed_evidence(root, evidence, **{**args, "failure_reason": "different"}))
    rejected(lambda: workflow.adopt_failed_evidence(root, evidence, **{**args, "operation": "gentle-denoise", "parameters": {}}))
    assert snapshot(root) == before and not list((root / "runs").iterdir())
    receipt = workflow.adopt_failed_evidence(root, evidence, **args)
    adopted = jobs.load_job(root)
    assert adopted.current_version == parent.id and adopted.latest_attempt is None
    assert adopted.runs[-1].execution == "failed" and adopted.runs[-1].technical == "not_run"
    assert receipt["verification"] is None and receipt["runtime"] is None
    assert workflow.summarize_job(root, parent.id)["legacy_failed_evidence"][0]["failure_reason"] == data["failure_reason"]
    for retry in (None, "pretend operational retry"):
        rejected(lambda: workflow.run_operation(root, workflow.LEGACY_OPERATION, parent.id,
                 parameters=parameters, retry_reason=retry), contains="Identical failed")
    assert workflow.describe_run(root, "gentle-denoise")["fingerprint"] != adopted.runs[-1].fingerprint
    assert not workflow.summarize_job(unrelated)["legacy_failed_evidence"]
    rejected(lambda: workflow.run_operation(root, workflow.LEGACY_OPERATION, parent.id,
             parameters={**parameters, "recipe_sha256": "b" * 64}), contains="no renderer")
    # A supported treatment needs complete equivalence evidence, not a legacy alias.
    treatment = workflow.describe_run(root, "trim", parent.id,
                                     parameters={"start_frame": 0, "end_frame": 60000})
    supported_data = {**data, "parameters": treatment["parameters"], "treatment": treatment}
    supported_evidence = temp / "supported-historical-failure.json"
    supported_evidence.write_text(json.dumps(supported_data), encoding="utf-8")
    job = jobs.load_job(root)
    workflow.adopt_failed_evidence(root, supported_evidence, **{**args,
        "expected_evidence_sha256": jobs.digest(supported_evidence), "operation": "trim",
        "parameters": treatment["parameters"],
        "evidence_mapping": {key: [key] for key in supported_data},
        "expected_revision": job.revision, "expected_revision_sha256": job.revision_sha256})
    rejected(lambda: workflow.run_operation(root, "trim", parent.id,
             parameters=treatment["parameters"], retry_reason="cannot waive historical failure"),
             contains="Identical failed")
    for budget_root in (root, unrelated):
        budget_job = jobs.load_job(budget_root)
        jobs.commit_revision(budget_root, replace(budget_job,
                             policy=replace(budget_job.policy, max_output_bytes=16 * 1024**2)))
    unrelated_result = workflow.run_operation(unrelated, "trim", parameters=treatment["parameters"])
    assert unrelated_result["execution"] == "completed", unrelated_result
    descendant = workflow.run_operation(root, "trim", parent.id,
                                       parameters={"start_frame": 1, "end_frame": 60001})["version"]
    assert len(workflow.summarize_job(root, descendant["id"])["legacy_failed_evidence"]) == 2
    adopted = jobs.load_job(root)
    restored = temp / "legacy-restored"
    assert jobs.copy_job(root, restored).runs == adopted.runs
    copied = restored / receipt["legacy_evidence"]["path"]
    assert copied.read_bytes() == evidence.read_bytes() == original
    rejected(lambda: workflow.run_operation(restored, workflow.LEGACY_OPERATION, parent.id,
             parameters=parameters, retry_reason="still blocked"), contains="Identical failed")
    copied.write_bytes(b"{}")
    rejected(lambda: jobs.load_job(restored), contains="evidence hash")
    print("PASS explicit legacy evidence adoption, exact blocking, copy and mapping rejection", flush=True)


def main():
    started = time.monotonic()
    assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required"
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    with tempfile.TemporaryDirectory(prefix="songtool-verify-") as directory:
        temp = Path(directory)
        frames = jobs.RATE * 2 + 137  # deliberately partial final half-second
        t = np.arange(frames) / jobs.RATE
        audio = np.column_stack((.12 * np.sin(2*np.pi*440*t), .10 * np.sin(2*np.pi*660*t)))
        source, silence = temp / "tone.wav", temp / "silence.wav"
        sf.write(source, audio, jobs.RATE, subtype="FLOAT")
        sf.write(silence, np.zeros_like(audio), jobs.RATE, subtype="FLOAT")
        a, b = temp / "a", temp / "b"
        cli("job", "create", source, a, "--intent", "music")
        cli("job", "create", silence, b, "--intent", "unknown")
        ja, jb = jobs.load_job(a), jobs.load_job(b)
        assert ja.id != jb.id and ja.source.sha256 != jb.source.sha256
        for root, job, src in ((a, ja, source), (b, jb, silence)):
            assert job.source.sha256 == jobs.digest(src)
            assert job.versions[0].frames == frames and job.versions[0].source_start_frame == 0
            assert not job.feedback and not job.protected_ranges and not job.runs
            assert job.versions[0].listening == "unreviewed"
            status = cli("job", "status", root, "--json")
            assert (status["execution"], status["technical"], status["listening"]) == ("not_run", "not_run", "unreviewed")
        cli("job", "create", source, a, "--intent", "music", ok=False)
        integrity_checks(b, temp)
        legacy_evidence_checks(temp, source, b)
        budget_job = jobs.load_job(a)
        jobs.commit_revision(a, replace(budget_job,
                             policy=replace(budget_job.policy, max_output_bytes=16 * 1024**2)))
        print("PASS isolated imports, hostile state, publication failure and concurrent writers", flush=True)

        silent_report = review.scan_version(b)
        report = read(silent_report)
        rows = read(silent_report.parent / "timeline.json")["rows"]
        assert len(rows) == 5 and rows[-1]["end_frame_exclusive"] == frames
        assert all(row["speech_similarity_hint"] is None for row in rows)
        assert not report["clips"] and not report["listening_approved"]
        rejected(lambda: review.scan_version(a, max_clips=9))
        rejected(lambda: review.scan_version(a, speech_version_id=jb.current_version))
        # Explicit self-reference provides deterministic nonzero suggestions, not defects.
        scan = cli("job", "scan", a, "--version", ja.current_version,
                   "--speech-version-id", ja.current_version, "--clips")
        assert scan["outcome"] == "analyzed_only" and not scan["rendered_now"]
        assert scan["runtime"]["inference_ran"] is False
        clips = read(Path(scan["clips_report"]))["clips"]
        assert 0 < len(clips) <= 8
        for clip in clips:
            assert 0 < clip["end_frame_exclusive"] - clip["start_frame"] <= 5 * jobs.RATE
            exported = sf.read(Path(scan["clips_report"]).parent / clip["file"], always_2d=True)[0]
            parent = sf.read(a / ja.versions[0].path, always_2d=True)[0]
            assert np.array_equal(exported, parent[clip["start_frame"]:clip["end_frame_exclusive"]])
        cli("job", "feedback", a, "--clip", clips[0]["id"], "--verdict", "good", "--note", "Exact accepted synthetic clip", "--accepted")
        assert jobs.load_job(a).versions[0].listening == "unreviewed"
        assert not jobs.load_job(b).feedback
        bad = temp / "nonfinite.wav"
        nonfinite = audio.copy(); nonfinite[0, 0] = np.nan
        sf.write(bad, nonfinite, jobs.RATE, subtype="DOUBLE")
        rejected(lambda: register(a, bad), contains="nonfinite")
        rejected(lambda: review.analyze_windows(bad))
        print("PASS scan silence/partial/nonfinite, bounded exact clips and isolated feedback", flush=True)

        args = ("job", "run", a, "--operation", "trim", "--version", ja.current_version,
                "--start-frame", 123, "--end-frame", 70000)
        rendered = cli(*args)
        assert rendered["outcome"] == "rendered_new" and rendered["rendered_now"]
        version = rendered["version"]
        assert version["source_start_frame"] == 123 and version["frames"] == 70000-123
        current = jobs.load_job(a)
        assert current.current_version == ja.current_version
        protected = jobs.mapped_protected_ranges(current, version["id"])
        assert [(p.start_frame, p.end_frame) for p in protected] == [(0, 70000-123)]
        candidate = sf.read(a / version["path"], always_2d=True)[0]
        assert np.array_equal(candidate, parent[123:70000])
        # Exercise real FFmpeg local EQ plus the production exact splice, not
        # merely trim copying. Entire accepted input must survive decoded DOUBLE.
        eq = cli("job", "run", a, "--operation", "local-eq", "--version", ja.current_version,
                 "--eq-interval", 0, frames, "--parent-sha256", ja.versions[0].sha256)
        assert eq["execution"] == "completed", eq
        assert np.array_equal(sf.read(a / eq["version"]["path"], always_2d=True)[0], parent)
        # Outside-only transitions are also tested with a nontrivial wet signal.
        interval = (30000, 40000)
        preserved, mapped = workflow._preserve(parent, parent * .999, [interval])
        assert mapped == [list(interval)] and np.array_equal(preserved[30000:40000], parent[30000:40000])
        assert not np.array_equal(preserved[:1000], parent[:1000])
        preserved_path = temp / "preserved.wav"
        review._write_audio(preserved_path, preserved)
        verified = workflow.validate_result(a / ja.versions[0].path, preserved_path,
            operation="local-eq", protected_intervals=[interval])
        assert verified["technical"] == "passed", verified
        rejected(lambda: workflow._preserve(parent, parent * .999, [(1, 100)]), contains="transition")
        before = snapshot(a)
        reused = cli(*args)
        assert reused["outcome"] == "reused_result" and not reused["rendered_now"]
        assert snapshot(a) == before
        first = workflow.describe_run(a, "trim", ja.current_version, parameters={"start_frame":123,"end_frame":70000})
        changed = workflow.describe_run(a, "trim", ja.current_version, parameters={"start_frame":124,"end_frame":70000})
        assert first["fingerprint"] != changed["fingerprint"]
        changed_audio = parent.copy(); changed_audio[1000, 0] += .01
        violation = temp / "violation.wav"
        sf.write(violation, changed_audio, jobs.RATE, subtype="DOUBLE")
        check = workflow.validate_result(a / ja.versions[0].path, violation, operation="bandit", protected_intervals=[(900,1100)])
        assert check["technical"] == "failed" and "protected_samples" in check["failures"]
        failed = register(a, violation, technical="failed", verification=check).versions[-1]
        cli("job", "choose", a, "--version", failed.id, "--verdict", "better", "--note", "must refuse", ok=False)
        unverified = register(a, source).versions[-1]
        cli("job", "choose", a, "--version", unverified.id, "--verdict", "better", "--note", "must refuse", ok=False)
        rejected(lambda: register(a, source, role="canonical"))
        j = jobs.load_job(a)
        rejected(lambda: jobs.register_version(a, source, parent_id=version["id"], parent_start_frame=0,
            expected_revision=j.revision, expected_revision_sha256=j.revision_sha256), contains="mapped")
        cli("job", "choose", a, "--version", version["id"], "--verdict", "unconfirmed", "--note", "not approved")
        assert jobs.load_job(a).current_version == ja.current_version
        cli("job", "choose", a, "--version", version["id"], "--verdict", "better", "--note", "synthetic selection test only")
        assert jobs.load_job(a).current_version == version["id"]
        # Only OS playback is replaced; parser, resolving, hashes and stdout are real.
        from songtool import __main__ as entry
        output = io.StringIO()
        opener = patch.object(entry.os, "startfile") if os.name == "nt" else patch.object(entry.subprocess, "run")
        with opener as opened, patch.object(sys, "argv", ["songtool", "job", "open", str(a), "--version", "current"]), redirect_stdout(output):
            entry.main()
        opened.assert_called_once()
        opened_result = json.loads(output.getvalue())
        assert opened_result["outcome"] == "opened_existing" and not opened_result["rendered_now"]
        assert opened_result["version"] == version["id"] and opened_result["listening"] == "better"
        cli("job", "choose", a, "--version", version["id"], "--verdict", "worse", "--note", "rejection remains distinct")
        cli(*args, ok=False)
        status = cli("job", "status", a, "--json")
        plain = cli("job", "status", a, raw=True)
        assert status["listening"] == "worse" and 'listening: "worse"' in plain
        assert next(v for v in status["versions"] if v["id"] == failed.id)["technical"] == "failed"
        # A genuine worker numerical failure publishes failed evidence, no candidate
        # and no promotion; even a retry reason cannot bypass technical guards.
        loud_path = temp / "over-peak.wav"
        sf.write(loud_path, parent * 12, jobs.RATE, subtype="DOUBLE")
        loud_version = register(a, loud_path).versions[-1]
        before_guard = jobs.load_job(a)
        guarded_args = ("job", "run", a, "--operation", "trim", "--version", loud_version.id,
                        "--start-frame", 0, "--end-frame", 70000)
        guarded = cli(*guarded_args)
        assert guarded["technical"] == "failed" and guarded["failure_kind"] == "guard"
        assert guarded["version"] is None and not guarded["rendered_now"]
        after_guard = jobs.load_job(a)
        assert after_guard.versions == before_guard.versions
        assert after_guard.current_version == before_guard.current_version
        cli(*guarded_args, "--retry-reason", "must not bypass guards", ok=False)
        guard_status = cli("job", "status", a, "--json")
        guard_plain = cli("job", "status", a, raw=True)
        assert guard_status["technical"] == "failed" and guard_status["listening"] == "worse"
        assert 'technical: "failed"' in guard_plain and 'listening: "worse"' in guard_plain
        print("PASS trim maps/protection, fingerprints/reuse, selection and open-existing distinctions", flush=True)

        # Operational failure is injected at owned execution, not by fabricating receipts.
        with patch.object(runtime, "run_owned", side_effect=OSError("injected spawn failure")):
            failure = workflow.run_operation(b, "scan")
        assert failure["failure_kind"] == "operational" and failure["execution"] == "failed"
        rejected(lambda: workflow.run_operation(b, "scan"), contains="failed attempt")
        retry = workflow.run_operation(b, "scan", retry_reason="controlled failure removed")
        assert retry["execution"] == "completed" and retry["outcome"] == "analyzed_only"
        assert workflow.run_operation(b, "scan")["outcome"] == "reused_result"
        tree_checks(temp)
        print("PASS failed retry/reuse and real timeout/cancel child+grandchild teardown", flush=True)

        controller = spawn("crash", b)
        try:
            wait_for(lambda: (b / "receipt-ready").exists(), controller)
            running = jobs.load_job(b)
            run_id = running.latest_attempt
            assert running.runs[-1].execution == "running"
            assert (b / "runs" / run_id / "receipt.json").exists()
            rejected(lambda: jobs.recover_run(b, run_id), contains="controller")
            rejected(lambda: workflow.run_operation(b, "scan"), contains="active run")
            (b / "crash-now").touch()
            finish(controller, 24)
            cli("job", "recover", b, "--run", run_id)
            recovered = jobs.load_job(b)
            assert recovered.runs[-1].execution == "completed" and recovered.versions[-1].technical == "passed"
            assert recovered.current_version == jb.current_version
            before = snapshot(b)
            jobs.recover_run(b, run_id)
            assert snapshot(b) == before
        finally:
            if controller.poll() is None:
                controller.kill(); controller.wait()
            for p in (b / "receipt-ready", b / "crash-now"):
                p.unlink(missing_ok=True)
        # Independent controller dies after durable intent, before any worker.
        interrupted = spawn("interrupt", b)
        finish(interrupted, 25)
        dead = jobs.load_job(b)
        assert dead.runs[-1].execution == "running"
        assert not (b / "runs" / dead.latest_attempt / "receipt.json").exists()
        cli("job", "recover", b, "--run", dead.latest_attempt)
        dead = jobs.load_job(b)
        assert dead.runs[-1].execution == "interrupted" and dead.runs[-1].technical == "not_run"
        assert dead.current_version == jb.current_version
        # A failed copy write must leave an explicitly unhealthy destination.
        incomplete = temp / "failed-copy"
        original_publish = jobs._publish
        def fail_completion(path, data):
            if path == incomplete / "copy-complete.json":
                raise OSError("injected copy completion failure")
            return original_publish(path, data)
        with patch.object(jobs, "_publish", fail_completion):
            rejected(lambda: jobs.copy_job(a, incomplete), contains="completion failure")
        rejected(lambda: jobs.load_job(incomplete))
        restored = temp / "restored"
        cli("job", "copy", a, restored)
        original, copy = jobs.load_job(a), jobs.load_job(restored)
        assert original == copy and original.feedback
        assert cli("job", "status", restored, "--json") == cli("job", "status", a, "--json")
        assert jobs.digest(restored / copy.source.path) == copy.source.sha256
        assert (restored / copy.source.path).stat().st_ino != (a / original.source.path).stat().st_ino
        cli("job", "copy", a, restored, ok=False)
        rejected(lambda: jobs.copy_job(a, a / "nested"), contains="overlap")
        (restored / "copy-complete.json").unlink()
        rejected(lambda: jobs.load_job(restored))
        cli("job", "status", restored, "--json", ok=False)
        # Restored media really are independent, not hard-linked aliases.
        restored_source = restored / copy.source.path
        source_before = jobs.digest(a / original.source.path)
        with restored_source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"x")
        assert jobs.digest(a / original.source.path) == source_before
        print("PASS separate-process receipt crash/recovery, live controller refusal, verified restore/incomplete rejection", flush=True)
    assert not any(name == "torch" or name.startswith("torch.") or name == "songtool.separation" for name in sys.modules)
    cli("cleanup-preview", "--help", raw=True)
    cli("--help", raw=True)
    print(f"PASS portable audio jobs ({time.monotonic()-started:.1f}s); no model/GPU or listening-quality verification")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        raise SystemExit(child_mode(sys.argv[2], Path(sys.argv[3])))
    main()
