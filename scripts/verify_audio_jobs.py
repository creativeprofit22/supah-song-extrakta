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
    if args[:2] == ("job", "status"):
        bootstrap = """import runpy,subprocess,sys
from pathlib import Path
from unittest.mock import patch
popen = subprocess.Popen
def status_process(argv, *args, **kwargs):
    # Historical receipt verification may measure loudness; never launch an audio worker.
    assert Path(argv[0]).name in ('ffmpeg', 'ffmpeg.exe'), argv
    return popen(argv, *args, **kwargs)
sys.argv = ['songtool', *sys.argv[1:]]
with patch('subprocess.Popen', side_effect=status_process), \
\
     patch('songtool.runtime.run_owned', side_effect=AssertionError('Status launched a worker')), \
     patch('songtool.resources.setup', side_effect=AssertionError('Status fetched a model')), \
     patch('songtool.resources.validate_resources', side_effect=AssertionError('Status accessed a model')):
    runpy.run_module('songtool', run_name='__main__')
assert not any(n == 'torch' or n.startswith('torch.') or n == 'songtool.separation' for n in sys.modules)
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


def child_mode(mode, root, operation="trim"):
    """Real independent writers/controllers, with deterministic boundary injection."""
    if mode.startswith("bounded-"):
        _, operation, mode = mode.split("-", 2)
        with patch.object(jobs, "MAX_REVISIONS", 4), patch.object(jobs, "MAX_METADATA_BYTES", 6500):
            return child_mode(mode, root, operation)
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
    if mode == "owned-interrupt":
        from verify_owned_runtime import WORKER, hold_teardown
        owned = runtime.run_owned
        def hanging(argv, output, **kwargs):
            return owned([sys.executable, "-B", "-c", WORKER, str(root / "ids.json")], output, **kwargs)
        with hold_teardown(), patch.object(runtime, "run_owned", hanging):
            workflow.run_operation(root, "trim", parameters={"start_frame": 321, "end_frame": 60000})
        raise AssertionError("Controller was not terminated")
    if mode == "interrupt":
        with patch.object(runtime, "run_owned", side_effect=lambda *a, **k: os._exit(25)):
            workflow.run_operation(root, operation,
                parameters={"start_frame": 500, "end_frame": 60000} if operation == "trim" else {})
        raise AssertionError("Interruption injection was not reached")
    if mode == "render-interrupt":
        owned = runtime.run_owned
        def stop_after_render(*args, **kwargs):
            owned(*args, **kwargs)
            os._exit(26)
        with patch.object(runtime, "run_owned", stop_after_render):
            workflow.run_operation(root, "trim", parameters={"start_frame": 321, "end_frame": 60000})
        raise AssertionError("Post-render interruption was not reached")
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
            workflow.run_operation(root, operation,
                parameters={"start_frame": 321, "end_frame": 60000} if operation == "trim" else {})
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


def owned_recovery_checks(temp, source):
    from verify_owned_runtime import kill_exact
    root = temp / "owned-recovery"
    jobs.create_job(source, root)
    process = spawn("owned-interrupt", root)
    controller_identity = runtime.process_identity(process.pid)
    assert controller_identity is not None
    identities, output = [], None
    try:
        wait_for(lambda: (root / "ids.json.ready").exists(), process)
        active = jobs.load_job(root)
        output = root / "runs" / active.latest_attempt / "worker"
        identities = read(root / "ids.json")
        for name in ("process.json", "guardian.json"):
            owner = read(output / name)
            identities.append([owner["pid"], owner["identity"]])
        kill_exact(process.pid, controller_identity)
        process.wait(timeout=10)
        wait_for(lambda: (output / "teardown-waiting").exists())
        before = snapshot(root)
        rejected(lambda: jobs.recover_run(root, active.latest_attempt), contains="guardian")
        assert snapshot(root) == before and jobs.load_job(root).runs[-1].execution == "running"
        assert all(runtime.process_matches(*item) for item in identities)
        (output / "allow-stop").touch()
        wait_for(lambda: not any(runtime.process_matches(*item) for item in identities))
        # Even dead identities alone are insufficient for a guardian-owned run.
        stopped = output / "guardian-stopped"
        stopped.rename(output / "saved-stopped")
        try:
            rejected(lambda: jobs.recover_run(root, active.latest_attempt), contains="unconfirmed")
            assert snapshot(root) == before
        finally:
            (output / "saved-stopped").rename(stopped)
        jobs.recover_run(root, active.latest_attempt)
        terminal = jobs.load_job(root).runs[-1]
        assert terminal.execution == "interrupted" and terminal.technical == "not_run"
        print("PASS recovery retains ownership during failed teardown and requires stop evidence", flush=True)
    finally:
        if output is not None:
            (output / "allow-stop").touch(exist_ok=True)
        kill_exact(process.pid, controller_identity)
        process.communicate(timeout=10)
        if (root / "ids.json.ready").exists():
            identities.extend(read(root / "ids.json"))
        for path in (root / "runs").glob("*/worker/*json"):
            if path.name in ("process.json", "guardian.json"):
                owner = read(path)
                identities.append([owner["pid"], owner["identity"]])
        for pid, identity in reversed(identities):
            kill_exact(pid, identity)
        wait_for(lambda: not any(runtime.process_matches(*item) for item in identities))


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


def import_budget_checks(temp, source):
    original = source.read_bytes()
    environment = os.environ.copy()
    for name, policy in (("invalid-threads", replace(jobs.ResourcePolicy(), cpu_threads=0)),
                         ("invalid-time", replace(jobs.ResourcePolicy(), wall_time_seconds=0)),
                         ("tiny-output", replace(jobs.ResourcePolicy(), max_output_bytes=1))):
        root = temp / name
        with patch.object(runtime, "run_owned", side_effect=AssertionError("Invalid import launched")):
            rejected(lambda: jobs.create_job(source, root, policy=policy))
        assert not root.exists()

    owned = runtime.run_owned
    # Inject at real import probe/conversion boundaries inside the owned worker.
    # Both the hanging child and grandchild record creation identities before waiting.
    hanging = '''import json,os,subprocess,sys,time
from pathlib import Path
from songtool.runtime import process_identity
p = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps({
    'identities': [[os.getpid(), process_identity(os.getpid())], [p.pid, process_identity(p.pid)]],
    'threads': [os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')],
    'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}))
sys.stdout.buffer.write(b'x' * (2 * 1024**2)); sys.stdout.flush()
time.sleep(60)
'''
    for boundary in ("probe", "conversion"):
        root = temp / f"import-hang-{boundary}"
        identities = root / "test-identities.json"
        prefix = f'''import subprocess,sys,time
from songtool import jobs
real_capture = jobs._import_capture
def hang(*args, **kwargs):
    subprocess.Popen([sys.executable, '-B', '-c', {hanging!r}, {str(identities)!r}]).wait()
'''
        if boundary == "probe":
            prefix += "jobs._import_capture = hang\n"
        else:
            # Earlier successful probes consume this same deadline, not fresh budgets.
            prefix += "def delayed(argv):\n    time.sleep(.3)\n    return real_capture(argv)\njobs._import_capture = delayed\njobs.subprocess.run = hang\n"
        def inject(argv, *args, **kwargs):
            command = list(argv)
            command[3] = prefix + command[3]
            return owned(command, *args, **kwargs)
        started = time.monotonic()
        with patch.object(runtime, "run_owned", side_effect=inject):
            rejected(lambda: jobs.create_job(source, root,
                     policy=replace(jobs.ResourcePolicy(), cpu_threads=3, wall_time_seconds=3)), contains="timed_out")
        assert time.monotonic() - started < 5, "Import did not honor its shared deadline"
        result = read(root / "runtime.json")
        assert result["status"] == "timed_out" and result["requested_device"] == "cpu"
        assert result["log_bytes"] <= 1024**2 and result["log_discarded_bytes"] > 0
        assert not runtime.process_matches(result["process_id"], result["process_identity"])
        recorded = read(identities)
        assert recorded["threads"] == ["3"] * 3 and recorded["cuda"] == ""
        for pid, identity in recorded["identities"]:
            assert identity and not runtime.process_matches(pid, identity)
        assert (root / "incomplete.json").exists()
        assert not (root / "state" / "00000001.json").exists()
        rejected(lambda: jobs.load_job(root))
    root = temp / "import-failed-after-ready"
    def fail_after_ready(argv, *args, **kwargs):
        command = list(argv)
        command[3] += "\nraise RuntimeError('controlled failure after verification')\n"
        return owned(command, *args, **kwargs)
    with patch.object(runtime, "run_owned", side_effect=fail_after_ready):
        rejected(lambda: jobs.create_job(source, root), contains="Import failed")
    assert (root / "import-ready.json").exists()
    assert read(root / "runtime.json")["status"] == "failed"
    assert not (root / "state" / "00000001.json").exists()
    rejected(lambda: jobs.load_job(root))
    assert source.read_bytes() == original and os.environ == environment
    print("PASS import preflight, shared deadlines, bounded logs and owned descendant teardown", flush=True)


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


def verification_receipt_checks(temp, source):
    root = temp / "verification-history"
    cli("job", "create", source, root, "--intent", "music")
    initial = jobs.load_job(root)
    parent = initial.versions[0]
    parameters = {"start_frame": 123, "end_frame": 60000}
    publish = jobs._version_receipt
    context_keys = ("parent_sha256", "parent_frames", "parent_start_frame", "operation",
                    "parameters", "protected_intervals")

    def old_worker_receipt(root, job_id, source_sha256, version, details):
        old_report = {k: v for k, v in details["verification"].items() if k not in context_keys}
        return publish(root, job_id, source_sha256, version, {**details, "verification": old_report})

    # Produce a real worker result in the prior receipt format, without editing history.
    with patch.object(jobs, "_version_receipt", side_effect=old_worker_receipt):
        result = workflow.run_operation(root, "trim", parameters=parameters)
    assert result["execution"] == "completed", result
    receipt_path = (root / result["version"]["path"]).parent / "receipt.json"
    old_bytes = receipt_path.read_bytes()
    wet = temp / "verification-wet.wav"
    samples = sf.read(root / parent.path, dtype="float64", always_2d=True)[0]
    sf.write(wet, samples * .999, jobs.RATE, subtype="DOUBLE")
    report = workflow.validate_result(root / parent.path, wet, operation="bandit")
    before_link = snapshot(root)
    rejected(lambda: register(root, wet, technical="passed", verification=report,
                              run_id=result["run"]["id"]), contains="run")
    assert snapshot(root) == before_link
    linked_audio = root / result["version"]["path"]
    linked_report = workflow.validate_result(root / parent.path, linked_audio, operation="trim", parameters=parameters)
    current = jobs.load_job(root)
    linked = jobs.register_version(root, linked_audio, parent_id=parent.id, parent_start_frame=123,
        technical="passed", verification=linked_report, run_id=result["run"]["id"],
        expected_revision=current.revision, expected_revision_sha256=current.revision_sha256)
    assert jobs.load_job(root).versions[-1].run_id == result["run"]["id"]
    assert linked.current_version == initial.current_version
    # Persist an invalid association as old unchecked registration could do.
    copied_link = temp / "invalid-run-link"
    jobs.copy_job(root, copied_link)
    with patch.object(jobs, "_validate_version_verification", return_value=report):
        invalid_link = register(copied_link, wet, technical="passed", verification=report, run_id=result["run"]["id"])
    rejected(lambda: jobs.load_job(copied_link), contains="linked run")
    rejected(lambda: jobs.select_version(copied_link, invalid_link.versions[-1].id,
                                         verdict="better", note="must refuse"), contains="linked run")
    with patch.object(runtime, "run_owned", side_effect=OSError("Synthetic operational failure")):
        failed_run = workflow.run_operation(root, "trim", parameters={"start_frame": 124, "end_frame": 60000})
    assert failed_run["execution"] == "failed" and not failed_run["rendered_now"]
    before_failed_link = snapshot(root)
    rejected(lambda: register(root, wet, technical="passed", verification=report,
                              run_id=failed_run["run"]["id"]), contains="run mismatch")
    assert snapshot(root) == before_failed_link
    registered = register(root, wet, technical="passed", verification=report)
    # Feedback added later must not retroactively alter an immutable technical check.
    jobs.record_feedback(root, version_id=parent.id, category="good", note="Later synthetic acceptance",
                         scope="whole", accepted=True, expected_revision=registered.revision,
                         expected_revision_sha256=registered.revision_sha256)
    loaded = jobs.load_job(root)
    assert loaded.versions[-1].technical == "passed" and receipt_path.read_bytes() == old_bytes
    history = snapshot(root)
    for index, altered in enumerate(({"technical": "failed", "failures": ["protected_samples"]},
                                     {"candidate_sha256": "0" * 64}, {})):
        copied = temp / f"invalid-verification-{index}"
        jobs.copy_job(root, copied)
        # Model old unchecked registration by bypassing only the new gate while
        # writing a fresh synthetic receipt/state. Then restore the real loader.
        with patch.object(jobs, "_validate_version_verification", return_value={**report, **altered}):
            invalid = register(copied, wet, technical="passed", verification={**report, **altered})
        assert invalid.current_version == initial.current_version
        rejected(lambda: jobs.load_job(copied))
        rejected(lambda: jobs.select_version(copied, invalid.versions[-1].id,
                                             verdict="better", note="must refuse"))
        assert read(sorted((copied / "state").glob("*.json"))[-1])["current_version"] == initial.current_version
    assert snapshot(root) == history
    print("PASS receipt-load rejection, immutable old worker receipts and creation-time protection", flush=True)


def render_activity_checks(temp, source):
    parameters = {"start_frame": 0, "end_frame": 60000}
    for mode in ("partial", "publication", "feedback", "invalid-ack", "unacknowledged"):
        root = temp / f"render-activity-{mode}"
        initial = jobs.create_job(source, root, intent="music")
        owned = runtime.run_owned

        def concurrent_feedback(*args, **kwargs):
            accounting = owned(*args, **kwargs)
            job = jobs.load_job(root)
            jobs.record_feedback(root, version_id=job.current_version, category="uncertain",
                note="Concurrent synthetic feedback", scope="whole", expected_revision=job.revision,
                expected_revision_sha256=job.revision_sha256)
            return accounting

        def incomplete_write(path, audio):
            with path.open("xb") as stream:
                stream.write(b"RIFFpartial")
            raise OSError("Injected partial audio write")

        def inline_worker(argv, output, **kwargs):
            output.mkdir(exist_ok=False)
            workflow._worker(root, argv[-1])
            raise OSError("Injected controller failure after worker")

        def invalid_ack(path, value, *args, **kwargs):
            if path.name == "render-complete.json":
                if mode == "unacknowledged":
                    raise OSError("Injected acknowledgement publication failure")
                value = {**value, "run_id": jobs.new_id()}
            return publish(path, value, *args, **kwargs)

        publish = workflow._publish
        if mode == "partial":
            with patch.object(runtime, "run_owned", inline_worker), patch.object(review, "_write_audio", incomplete_write):
                receipt = workflow.run_operation(root, "trim", parameters=parameters)
        elif mode == "publication":
            with patch.object(jobs, "_version_receipt", side_effect=OSError("Injected publication failure")):
                receipt = workflow.run_operation(root, "trim", parameters=parameters)
        elif mode == "feedback":
            with patch.object(runtime, "run_owned", concurrent_feedback):
                receipt = workflow.run_operation(root, "trim", parameters=parameters)
        else:
            with patch.object(runtime, "run_owned", inline_worker), patch.object(workflow, "_publish", invalid_ack):
                receipt = workflow.run_operation(root, "trim", parameters=parameters)
        completed = mode in ("publication", "feedback")
        assert receipt["rendered_now"] is completed, receipt
        assert (receipt["render_acknowledgement"] is not None) is completed
        assert receipt["outcome"] == receipt["execution"] == "failed" and receipt["version"] is None
        assert not receipt["listening_approved"]
        after = jobs.load_job(root)
        assert after.versions == initial.versions and after.current_version == initial.current_version
        assert after.latest_technically_passed_candidate is None
        worker = root / "runs" / receipt["run"]["id"] / "worker"
        if mode == "partial":
            assert (worker / "candidate.wav").read_bytes() == b"RIFFpartial"
            assert not (worker / "render-complete.json").exists()
        elif mode == "invalid-ack":
            assert "acknowledgement invalid" in receipt["error"]
        elif mode == "unacknowledged":
            assert sf.info(worker / "candidate.wav").frames == parameters["end_frame"]
            assert not (worker / "render-complete.json").exists()
        else:
            assert jobs.digest(worker / "candidate.wav") == receipt["render_acknowledgement"]["audio_sha256"]
            assert receipt["verification"]["technical"] == "passed", receipt
            if mode == "feedback":
                assert receipt["technical"] == "failed"

    loud = temp / "render-recovery-loud.wav"
    sf.write(loud, np.ones((72000, 2)) * 1.1, jobs.RATE, subtype="DOUBLE")
    for mode in ("crash", "render-interrupt"):
        root = temp / f"render-recovery-{mode}"
        initial = jobs.create_job(loud, root, intent="music")
        controller = spawn(mode, root)
        try:
            if mode == "crash":
                wait_for(lambda: (root / "receipt-ready").exists(), controller)
                (root / "crash-now").touch()
            finish(controller, 24 if mode == "crash" else 26)
            running = jobs.load_job(root)
            run_id = running.latest_attempt
            path = root / "runs" / run_id / "receipt.json"
            old_bytes = path.read_bytes() if path.exists() else None
            recovered = jobs.recover_run(root, run_id)
            receipt = read(path)
            assert receipt["rendered_now"] and receipt["version"] is None
            assert receipt["render_acknowledgement"]["frames"] == 60000 - 321
            assert receipt["technical"] == ("failed" if mode == "crash" else "not_run")
            assert recovered.runs[-1].execution == ("failed" if mode == "crash" else "interrupted")
            assert recovered.versions == initial.versions and recovered.current_version == initial.current_version
            assert recovered.latest_technically_passed_candidate is None
            if old_bytes is not None:
                assert path.read_bytes() == old_bytes
            history = snapshot(root)
            assert jobs.recover_run(root, run_id) == recovered and snapshot(root) == history
            copied = temp / f"render-recovery-copy-{mode}"
            assert jobs.copy_job(root, copied) == recovered
            ack_path = copied / "runs" / run_id / "worker" / "render-complete.json"
            for change in ({"frames": 123}, {"audio_sha256": "0" * 64}, {"encoding": "FLOAT"}):
                # Corrupt disposable copied evidence, never the original receipt/history.
                ack_path.write_text(json.dumps({**receipt["render_acknowledgement"], **change}), encoding="utf-8")
                rejected(lambda: jobs.load_job(copied))
            for key, value in (("rendered_now", False), ("render_acknowledgement", None)):
                rejected(lambda: jobs._validate_render_activity(root, recovered, recovered.runs[-1],
                                                               {**receipt, key: value}))
        finally:
            if controller.poll() is None:
                controller.kill(); controller.wait()
    legacy_root = temp / "historical-render-activity"
    jobs.create_job(loud, legacy_root, intent="music")
    publish = workflow._publish
    def historical_receipt(path, value, *args, **kwargs):
        if path.name == "receipt.json":
            value = {k: v for k, v in value.items() if k != "render_acknowledgement"}
            value["rendered_now"] = False
        return publish(path, value, *args, **kwargs)
    with patch.object(workflow, "_publish", historical_receipt):
        result = workflow.run_operation(legacy_root, "trim", parameters=parameters)
    receipt_path = Path("runs") / result["run"]["id"] / "receipt.json"
    historical_bytes = (legacy_root / receipt_path).read_bytes()
    assert not read(legacy_root / receipt_path)["rendered_now"]
    legacy = jobs.load_job(legacy_root)
    assert legacy.runs[-1].technical == "failed"
    copied = temp / "historical-render-copy"
    assert jobs.copy_job(legacy_root, copied) == legacy
    assert (copied / receipt_path).read_bytes() == historical_bytes == (legacy_root / receipt_path).read_bytes()
    print("PASS completed vs partial renders, publication/feedback failures and immutable crash recovery", flush=True)


def partial_diagnostic_copy_checks(temp, source):
    root = temp / "partial-diagnostic-job"
    job = jobs.create_job(source, root, intent="music")
    job = jobs.commit_revision(root, replace(job,
                               policy=replace(job.policy, max_output_bytes=16 * 1024**2)))
    jobs.record_feedback(root, version_id=job.current_version, category="uncertain",
                         note="Keep this exact failed-run feedback.", scope="whole",
                         expected_revision=job.revision,
                         expected_revision_sha256=job.revision_sha256)
    before = cli("job", "status", root, "--json")
    original_dump = json.dump
    partial = b'{"status":'

    def interrupted_dump(value, stream, *args, **kwargs):
        if Path(stream.name).name == "runtime.json":
            stream.write(partial.decode("ascii"))
            stream.flush()
            raise OSError("injected partial runtime diagnostic write")
        return original_dump(value, stream, *args, **kwargs)

    # One real tiny CPU worker, two numerical threads, 16 MiB / 30 seconds.
    with patch.object(runtime.json, "dump", side_effect=interrupted_dump):
        receipt = workflow.run_operation(root, "trim", parameters={"start_frame": 0, "end_frame": 9600},
                                         device="cpu", timeout=30)
    assert receipt["execution"] == "failed" and receipt["failure_kind"] == "operational"
    assert receipt["version"] is None and receipt["runtime"] is None
    assert receipt["rendered_now"] and receipt["render_acknowledgement"] is not None
    failed = jobs.load_job(root)
    assert failed.current_version == job.current_version and failed.versions == job.versions
    diagnostic = Path("runs") / failed.runs[-1].id / "worker" / "runtime.json"
    assert (root / diagnostic).read_bytes() == partial
    status = cli("job", "status", root, "--json")
    assert status["execution"] == "failed" and status["technical"] == "not_run"
    assert before["execution"] == "not_run"
    history = snapshot(root)
    # Failed publication staging and arbitrary uncommitted diagnostics are evidence too.
    unfinished = Path("runs") / jobs.new_id() / "worker"
    (root / unfinished).mkdir(parents=True)
    extras = [diagnostic.parent / "extra.json", diagnostic.parent / "scan.json",
              unfinished / "verification.json", Path("state") / ".pending-interrupted"]
    for relative in extras:
        with (root / relative).open("xb") as stream:
            stream.write(partial)
    # A published but uncommitted report's retained alias is internal, not media sharing.
    aliased = unfinished / "scan.json"
    jobs._publish(root / aliased, partial)
    extras.append(aliased)
    copied = temp / "partial-diagnostic-copy"
    assert jobs.copy_job(root, copied) == failed
    assert (copied / diagnostic).read_bytes() == partial == (root / diagnostic).read_bytes()
    assert jobs.digest(copied / diagnostic) == jobs.digest(root / diagnostic)
    assert cli("job", "status", copied, "--json") == status
    assert snapshot(root) == history and jobs.load_job(root) == failed
    assert jobs.load_job(copied).feedback == failed.feedback
    for relative in extras:
        assert (root / relative).read_bytes() == (copied / relative).read_bytes() == partial
    restored = temp / "partial-diagnostic-restored"
    assert jobs.copy_job(copied, restored) == failed
    assert (restored / diagnostic).read_bytes() == partial
    rejected(lambda: jobs.copy_job(root, copied), contains="fresh")
    for index, relative in enumerate((Path("state") / f"{failed.revision:08d}.json",
                                      Path(failed.versions[0].path).parent / "receipt.json",
                                      diagnostic.parent.parent / "receipt.json",
                                      diagnostic.parent.parent / "intent.json",
                                      Path("import-ready.json"), Path("copy-complete.json"))):
        corrupt = temp / f"partial-diagnostic-corrupt-{index}"
        jobs.copy_job(root, corrupt)
        # Deliberately corrupt only disposable copied metadata, never the source evidence.
        with (corrupt / relative).open("wb") as stream:
            stream.write(partial)
        destination = temp / f"refused-corrupt-copy-{index}"
        rejected(lambda: jobs.copy_job(corrupt, destination))
        assert not destination.exists()
    linked = copied / diagnostic.parent / "linked.json"
    os.link(copied / diagnostic, linked)
    rejected(lambda: jobs.copy_job(copied, temp / "refused-linked-diagnostic"), contains="Hard-linked")
    oversized = restored / diagnostic.parent / "oversized.json"
    with oversized.open("xb") as stream:
        stream.truncate(jobs.MAX_METADATA_BYTES + 1)
    rejected(lambda: jobs.copy_job(restored, temp / "refused-oversized-diagnostic"), contains="Oversized")
    assert jobs.load_job(root) == failed and (root / diagnostic).read_bytes() == partial
    print("PASS partial runtime/staging copy and restore; corrupt metadata, links and size refused", flush=True)


def clip_version_feedback_checks(temp, source):
    root = temp / "clip-version-feedback"
    job = jobs.create_job(source, root, intent="music")
    canonical = job.versions[0]
    versions = [register(root, root / canonical.path).versions[-1] for _ in range(2)]
    assert versions[0].id != versions[1].id
    assert versions[0].sha256 == versions[1].sha256 == canonical.sha256
    assert versions[0].source_start_frame == versions[1].source_start_frame == 0
    scans = [review.scan_version(root, v.id, speech_version_id=v.id) for v in versions]
    clips = [read(path)["clips"][0] for path in scans]
    assert clips[0]["id"] == clips[1]["id"]
    assert [c["version_id"] for c in clips] == [v.id for v in versions]
    immutable = {p: p.read_bytes() for scan in scans for p in (scan, scan.parent / "timeline.json")}
    args = ("job", "feedback", root, "--clip", clips[0]["id"], "--verdict", "good", "--accepted")
    before = jobs.load_job(root)
    result = subprocess.run([sys.executable, "-B", "-m", "songtool", *map(str, args),
                             "--note", "Ambiguous acceptance must not persist"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--version" in result.stderr, result.stderr
    assert jobs.load_job(root) == before
    for identifier in ("invalid", jobs.new_id(), canonical.id):
        cli(*args, "--version", identifier, "--note", "Invalid target", ok=False)
        assert jobs.load_job(root) == before
    for index, version in enumerate(versions):
        note = f"Exact acceptance for sibling {index}"
        cli(*args, "--version", version.id, "--note", note)
        job = jobs.load_job(root)
        item = job.feedback[-1]
        assert len(job.feedback) == len(job.protected_ranges) == index + 1
        assert (item.version_id, item.note, item.scope, item.accepted) == (version.id, note, "interval", True)
        assert (item.start_frame, item.end_frame) == (clips[index]["start_frame"], clips[index]["end_frame_exclusive"])
        for sibling_index, sibling in enumerate(versions):
            protected = jobs.mapped_protected_ranges(job, sibling.id)
            assert len(protected) == (1 if sibling_index <= index else 0)
            if protected:
                assert protected[0].feedback_id == job.feedback[sibling_index].id
                assert (protected[0].start_frame, protected[0].end_frame) == (item.start_frame, item.end_frame)
        assert not jobs.mapped_protected_ranges(job, canonical.id)
        assert job.current_version == canonical.id and job.versions == before.versions
    assert all(p.read_bytes() == data for p, data in immutable.items())
    print("PASS colliding clips: explicit version feedback/protection, invalid targets and immutable scans", flush=True)


def feedback_scope_checks(temp, source):
    # Synthetic mapped trims only; denoise dispatch always fails before any worker.
    for category in ("wanted_vocal_loss", "warbling_reverse_like_artifact"):
        root = temp / category
        jobs.create_job(source, root, intent="music")

        def trim(parent, first, last):
            job = jobs.load_job(root)
            audio = sf.read(root / parent.path, start=first, stop=last, always_2d=True)[0]
            path = temp / f"{jobs.new_id()}.wav"
            sf.write(path, audio, jobs.RATE, subtype="FLOAT")
            return jobs.register_version(root, path, parent_id=parent.id, parent_start_frame=first,
                expected_revision=job.revision, expected_revision_sha256=job.revision_sha256).versions[-1]

        def feedback(version, scope, first=None, last=None):
            job = jobs.load_job(root)
            return jobs.record_feedback(root, version_id=version.id, category=category,
                note="Keep this exact scoped wording.", scope=scope, start_frame=first, end_frame=last,
                expected_revision=job.revision, expected_revision_sha256=job.revision_sha256)

        def check(version, blocked):
            before = jobs.load_job(root)
            status = workflow.summarize_job(root, version.id)
            assert status["repair_status"] == ("no_supported_repair" if blocked else "explicit_operation_required")
            with patch.object(runtime, "run_owned", side_effect=RuntimeError("controlled scope dispatch failure")) as worker:
                if blocked:
                    rejected(lambda: workflow.run_operation(root, "gentle-denoise", version.id),
                             contains="no_supported_repair")
                    worker.assert_not_called()
                    assert jobs.load_job(root) == before
                else:
                    receipt = workflow.run_operation(root, "gentle-denoise", version.id)
                    worker.assert_called_once()
                    assert receipt["outcome"] == "failed" and receipt["failure_kind"] == "operational"
                    assert "controlled scope dispatch failure" in receipt["error"]
            after = jobs.load_job(root)
            assert after.feedback == before.feedback and after.versions == before.versions

        canonical = jobs.load_job(root).versions[0]
        origin = trim(canonical, 200, 2000)
        disjoint = trim(origin, 300, 800)
        overlap = trim(origin, 150, 800)
        boundary = trim(origin, 200, 800)
        left_boundary = trim(origin, 0, 100)
        nested_disjoint = trim(overlap, 50, 650)
        nested_overlap = trim(overlap, 25, 650)
        sibling = trim(canonical, 200, 2000)  # Equal length/map is not ancestry.
        recorded = feedback(origin, "interval", 100, 200)
        assert recorded.feedback[-1].start_frame == 100 and recorded.feedback[-1].end_frame == 200
        for version, blocked in ((origin, True), (disjoint, False), (overlap, True),
                                 (boundary, False), (left_boundary, False),
                                 (nested_disjoint, False), (nested_overlap, True),
                                 (sibling, False), (canonical, False)):
            check(version, blocked)
        feedback(origin, "whole")
        for version in (origin, disjoint, overlap, boundary, left_boundary, nested_disjoint, nested_overlap):
            check(version, True)
    print("PASS scoped feedback: disjoint/overlap/boundary/nested/whole and status-run agreement", flush=True)


def run_headroom_checks(temp, source):
    base = temp / "headroom-base"
    initial = jobs.create_job(source, base, policy=replace(jobs.ResourcePolicy(),
                              max_output_bytes=256 * 1024**2, wall_time_seconds=30))
    original_source = source.read_bytes()
    originals = {}

    def remember(root):
        originals[root] = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    remember(base)

    def fresh(name):
        root = temp / f"headroom-{name}"
        jobs.copy_job(base, root)
        remember(root)
        return root

    def unchanged(root):
        assert source.read_bytes() == original_source
        assert all((root / p).read_bytes() == data for p, data in originals[root].items())

    def run(root, operation):
        return workflow.run_operation(root, operation, device="cpu", timeout=30,
            parameters={"start_frame": 123, "end_frame": 60000} if operation == "trim" else {})

    # Both dispatch and direct commits must refuse before publishing active state.
    for operation in ("scan", "trim"):
        for limit in (2, 3, "bytes"):
            root = fresh(f"refuse-{operation}-{limit}")
            parent = initial.versions[0]
            active = jobs.Run(jobs.new_id(), operation, "f" * 64, parent.id, parent.sha256)
            pending = replace(initial, runs=(active,), latest_attempt=active.id)
            size = len(jobs._encode(jobs._data(replace(pending, revision=2,
                                   previous_revision_sha256=initial.revision_sha256)), jobs.MAX_METADATA_BYTES))
            with patch.object(jobs, "MAX_REVISIONS", 8 if limit == "bytes" else limit), \
                    patch.object(jobs, "MAX_METADATA_BYTES", size + 10 if limit == "bytes" else 6500), \
                    patch.object(runtime, "run_owned", side_effect=AssertionError("Refused run launched")) as owned:
                before = snapshot(root)
                rejected(lambda: run(root, operation))
                owned.assert_not_called()
                assert not list((root / "runs").iterdir())
                rejected(lambda: jobs.commit_revision(root, pending))
                assert jobs.load_job(root) == initial and snapshot(root) == before
                assert jobs.copy_job(root, temp / f"copy-refused-{operation}-{limit}") == initial
            unchanged(root)

    owned = runtime.run_owned
    for operation in ("scan", "trim"):
        for outcome in ("completed", "spawn-failed", "failed", "interrupted"):
            root = fresh(f"terminal-{operation}-{outcome}")
            def execute(argv, *args, **kwargs):
                if outcome == "spawn-failed":
                    raise OSError("Controlled spawn failure")
                started = kwargs["on_started"]
                def on_started(pid, identity):
                    started(pid, identity)
                    if outcome == "failed":
                        raise OSError("Controlled post-identity failure")
                    if outcome == "interrupted":
                        raise KeyboardInterrupt("Controlled interruption")
                return owned(argv, *args, **{**kwargs, "on_started": on_started})
            with patch.object(jobs, "MAX_REVISIONS", 4), patch.object(jobs, "MAX_METADATA_BYTES", 6500), \
                    patch.object(runtime, "run_owned", side_effect=execute):
                receipt = run(root, operation)
                terminal = jobs.load_job(root)
                assert terminal.runs[-1].execution == ("completed" if outcome == "completed" else "failed"), receipt
                assert terminal.revision == (3 if outcome == "spawn-failed" else 4)
                assert bool(receipt["version"]) == (operation == "trim" and outcome == "completed")
                assert terminal.runs[-1].receipt_sha256 == jobs.digest(root / "runs" / terminal.latest_attempt / "receipt.json")
                assert jobs.copy_job(root, temp / f"copy-terminal-{operation}-{outcome}") == terminal
            unchanged(root)

    # Interleave actual feedback API commits with the controller, before/after
    # child identity. Byte refusal must happen while the requested snapshot fits.
    for operation in ("scan", "trim"):
        for ceiling in ("revisions", "bytes"):
            for boundary in ("intent", "child"):
                root = fresh(f"feedback-{operation}-{ceiling}-{boundary}")
                accepted = []
                def feedback_until_reserved():
                    while True:
                        current = jobs.load_job(root)
                        note = "Concurrent wording " + "x" * 180
                        item = jobs.Feedback(jobs.new_id(), current.current_version,
                                             current.versions[0].sha256, "uncertain", note, "whole")
                        proposal = replace(current, feedback=(*current.feedback, item))
                        history = snapshot(root)
                        try:
                            result = jobs.record_feedback(root, version_id=current.current_version,
                                category="uncertain", note=note, scope="whole",
                                expected_revision=current.revision, expected_revision_sha256=current.revision_sha256)
                        except ValueError:
                            # Not an oversized input: shared commit must protect future growth.
                            jobs.validate_job(proposal)
                            rejected(lambda: jobs.commit_revision(root, proposal))
                            assert snapshot(root) == history
                            if ceiling == "revisions":
                                rejected(lambda: jobs.commit_revision(root, replace(current,
                                    policy=replace(current.policy, cpu_threads=3))), contains="headroom")
                            break
                        accepted.append(result.feedback[-1])
                def execute(argv, *args, **kwargs):
                    if boundary == "intent":
                        feedback_until_reserved()
                    started = kwargs["on_started"]
                    def on_started(pid, identity):
                        started(pid, identity)
                        if boundary == "child":
                            feedback_until_reserved()
                    return owned(argv, *args, **{**kwargs, "on_started": on_started})
                with patch.object(jobs, "MAX_REVISIONS", 6 if ceiling == "revisions" else 32), \
                        patch.object(jobs, "MAX_METADATA_BYTES", 6500), \
                        patch.object(runtime, "run_owned", side_effect=execute):
                    receipt = run(root, operation)
                    terminal = jobs.load_job(root)
                    assert accepted and terminal.feedback == tuple(accepted)
                    assert terminal.runs[-1].execution == "failed" and receipt["version"] is None
                    assert jobs.copy_job(root, temp / f"copy-feedback-{operation}-{ceiling}-{boundary}") == terminal
                unchanged(root)

    # Real controller death, with and without an already-published success receipt.
    for operation, mode in ((op, mode) for op in ("scan", "trim") for mode in ("interrupt", "crash")):
        root = fresh(f"recover-{operation}-{mode}")
        process = spawn(f"bounded-{operation}-{mode}", root)
        try:
            if mode == "crash":
                wait_for(lambda: (root / "receipt-ready").exists(), process)
                (root / "crash-now").touch()
            finish(process, 25 if mode == "interrupt" else 24)
            with patch.object(jobs, "MAX_REVISIONS", 4), patch.object(jobs, "MAX_METADATA_BYTES", 6500):
                dead = jobs.load_job(root)
                assert dead.runs[-1].execution == "running"
                history = snapshot(root)
                terminal = jobs.recover_run(root, dead.latest_attempt)
                assert terminal.runs[-1].execution == ("interrupted" if mode == "interrupt" else "completed")
                assert all(snapshot(root)[name] == data for name, data in history.items())
                assert jobs.recover_run(root, dead.latest_attempt) == terminal
                assert jobs.copy_job(root, temp / f"copy-recovered-{operation}-{mode}") == terminal
            unchanged(root)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    unchanged(base)
    print("PASS reserved revisions/bytes: refusal, scan/render terminals, concurrent feedback and dead-run recovery", flush=True)


def intent_status_checks(temp, source):
    intents: tuple[jobs.Intent, ...] = ("music", "spoken_audio", "unknown")
    for intent in intents:
        for rap in (False, True):
            root = temp / f"intent-{intent}-{rap}"
            if rap:
                cli("job", "create", source, root, "--intent", intent, "--wanted-vocals-may-include-rap")
            else:
                jobs.create_job(source, root, intent=intent, wanted_vocals_may_include_rap=rap)
            original = jobs.load_job(root)
            assert original.revision_sha256 is not None
            copied = temp / f"intent-copy-{intent}-{rap}"
            jobs.copy_job(root, copied)
            for directory in (root, copied):
                before = {p.relative_to(directory): jobs.digest(p) for p in directory.rglob("*") if p.is_file()}
                with patch.object(runtime, "run_owned", side_effect=AssertionError("Status launched a worker")), \
                     patch.object(subprocess, "Popen", side_effect=AssertionError("Status launched a process")):
                    status = workflow.summarize_job(directory)
                public = cli("job", "status", directory, "--json")
                text = cli("job", "status", directory, raw=True)
                assert all(public[key] == value for key, value in status.items())
                assert status["intent"] == intent and status["wanted_vocals_may_include_rap"] is rap
                assert f'intent: "{intent}"' in text
                assert f'wanted_vocals_may_include_rap: {json.dumps(rap)}' in text
                assert "recommendations:" in text
                guidance = " ".join(status["recommendations"])
                assert "CPU" in guidance and "CUDA only by explicit choice" in guidance
                if intent == "music":
                    assert "Music intent does not automatically select" in guidance
                else:
                    assert f"Recording intent is {intent}" in guidance
                    assert "requires explicit selection" in guidance and "may remove wanted speech" in guidance
                assert ("Wanted vocals may include rap" in guidance) is rap
                if rap:
                    assert "Do not use speech-stem reinsertion or denoise as restoration" in guidance
                assert status["repair_status"] == "explicit_operation_required"
                assert status["listening"] == "unreviewed" and public["listening_scope"] == "none"
                assert jobs.load_job(directory) == original
                assert before == {p.relative_to(directory): jobs.digest(p) for p in directory.rglob("*") if p.is_file()}
            # Intent advice cannot override the existing feedback-based repair restriction.
            jobs.record_feedback(root, version_id=original.versions[0].id, category="wanted_vocal_loss",
                note="Wanted rap or speech is missing", scope="whole", expected_revision=original.revision,
                expected_revision_sha256=original.revision_sha256)
            before = jobs.load_job(root)
            blocked = workflow.summarize_job(root)
            assert blocked["repair_status"] == "no_supported_repair"
            assert blocked["recommendations"] == status["recommendations"]
            assert cli("job", "status", root, "--json")["repair_status"] == "no_supported_repair"
            assert jobs.load_job(root) == before
    assert not any(n == "torch" or n.startswith("torch.") or n == "songtool.separation" for n in sys.modules)
    print("PASS intent/rap API, CLI and copied status: advisory only, no workers/models or approval changes", flush=True)


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
        intent_status_checks(temp, source)
        run_headroom_checks(temp, source)
        clip_version_feedback_checks(temp, source)
        feedback_scope_checks(temp, source)
        render_activity_checks(temp, source)
        partial_diagnostic_copy_checks(temp, source)
        import_budget_checks(temp, source)
        a, b = temp / "a", temp / "b"
        cli("job", "create", source, a, "--intent", "music")
        cli("job", "create", silence, b, "--intent", "unknown")
        ja, jb = jobs.load_job(a), jobs.load_job(b)
        assert ja.id != jb.id and ja.source.sha256 != jb.source.sha256
        for root, job, src in ((a, ja, source), (b, jb, silence)):
            assert job.source.sha256 == jobs.digest(src)
            assert (root / job.source.path).read_bytes() == src.read_bytes()
            info = sf.info(root / job.versions[0].path)
            assert (info.samplerate, info.channels, info.subtype) == (48000, 2, "FLOAT")
            imported = read(root / "runtime.json")
            assert imported["status"] == "completed" and imported["requested_device"] == "cpu"
            assert not runtime.process_matches(imported["process_id"], imported["process_identity"])
            assert job.versions[0].frames == frames and job.versions[0].source_start_frame == 0
            assert not job.feedback and not job.protected_ranges and not job.runs
            assert job.versions[0].listening == "unreviewed"
            status = cli("job", "status", root, "--json")
            assert (status["execution"], status["technical"], status["listening"]) == ("not_run", "not_run", "unreviewed")
        cli("job", "create", source, a, "--intent", "music", ok=False)
        integrity_checks(b, temp)
        legacy_evidence_checks(temp, source, b)
        verification_receipt_checks(temp, source)
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
            operation="local-eq", parameters={"parent_sha256": ja.versions[0].sha256,
                                               "intervals": [[0, frames]]}, protected_intervals=[interval])
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
        before_registration = snapshot(a)
        current_id = jobs.load_job(a).current_version
        rejected(lambda: register(a, violation, technical="passed", verification=check), contains="passing verification")
        # A genuine pass computed without this job's protection must not confer a pass.
        unprotected = workflow.validate_result(a / ja.versions[0].path, violation, operation="bandit")
        assert unprotected["technical"] == "passed"
        rejected(lambda: register(a, violation, technical="passed", verification=unprotected), contains="protected_samples")
        protected_report = workflow.validate_result(a / ja.versions[0].path, a / eq["version"]["path"],
            operation="bandit", protected_intervals=[[r.start_frame, r.end_frame]
                for r in jobs.mapped_protected_ranges(jobs.load_job(a), ja.versions[0].id)])
        for altered in ({"candidate_sha256": "0" * 64}, {"parent_sha256": "0" * 64},
                        {"failures": ["true_peak"]}, {"parent_start_frame": 1},
                        {"protected_intervals": []}, {"frames": frames - 1}, {"encoding": "FLOAT"}):
            rejected(lambda: register(a, a / eq["version"]["path"], technical="passed",
                                      verification={**protected_report, **altered}))
        assert snapshot(a) == before_registration
        assert jobs.load_job(a).current_version == current_id
        # Failed registrations leave only diagnostics, never selectable versions.
        registered_ids = {v.id for v in jobs.load_job(a).versions}
        for orphan in (a / "versions").iterdir():
            if orphan.name not in registered_ids:
                rejected(lambda: jobs.select_version(a, orphan.name, verdict="better", note="must refuse"),
                         contains="Unknown selection")
        validated = register(a, a / eq["version"]["path"], technical="passed",
                             verification=protected_report).versions[-1]
        assert jobs.load_job(a).current_version == current_id
        selected = jobs.select_version(a, validated.id, verdict="better", note="Synthetic validated registration")
        assert selected.current_version == validated.id
        jobs.select_version(a, current_id, verdict="better", note="Restore synthetic test preference")
        print("PASS registration revalidates status, hashes, timeline, encoding and protected samples", flush=True)
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
        assert guarded["version"] is None and guarded["rendered_now"]
        assert guarded["outcome"] == guarded["execution"] == "failed"
        assert not guarded["listening_approved"]
        diagnostic = a / "runs" / guarded["run"]["id"] / "worker" / "candidate.wav"
        decoded, rate = sf.read(diagnostic, dtype="float64", always_2d=True)
        assert rate == jobs.RATE and np.array_equal(decoded, (parent * 12)[:70000])
        acknowledgement = guarded["render_acknowledgement"]
        assert acknowledgement == read(diagnostic.parent / "render-complete.json")
        assert acknowledgement["frames"] == len(decoded) == 70000
        assert acknowledgement["encoding"] == sf.info(diagnostic).subtype == "DOUBLE"
        assert acknowledgement["audio_sha256"] == jobs.digest(diagnostic)
        assert acknowledgement["run_id"] == guarded["run"]["id"]
        after_guard = jobs.load_job(a)
        assert after_guard.versions == before_guard.versions
        assert after_guard.current_version == before_guard.current_version
        assert after_guard.latest_technically_passed_candidate == before_guard.latest_technically_passed_candidate
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
        assert not failure["rendered_now"] and failure["render_acknowledgement"] is None
        rejected(lambda: workflow.run_operation(b, "scan"), contains="failed attempt")
        retry = workflow.run_operation(b, "scan", retry_reason="controlled failure removed")
        assert retry["execution"] == "completed" and retry["outcome"] == "analyzed_only"
        assert workflow.run_operation(b, "scan")["outcome"] == "reused_result"
        tree_checks(temp)
        from verify_owned_runtime import check_controller_death
        check_controller_death()
        owned_recovery_checks(temp, source)
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
