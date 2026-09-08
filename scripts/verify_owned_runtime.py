"""Synthetic controller-death gate; standard library only, no audio or GPU work."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songtool import runtime

WORKER = """import json,os,subprocess,sys,time
from pathlib import Path
from songtool.runtime import process_identity
p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps([[os.getpid(),process_identity(os.getpid())],[p.pid,process_identity(p.pid)]]))
Path(sys.argv[1]+'.ready').touch()
time.sleep(60)
"""


def wait_for(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "Owned tree did not reach the expected boundary"
        time.sleep(.01)


def kill_exact(pid, identity):
    if not runtime.process_matches(pid, identity):
        return
    if os.name == "nt":
        # Open and recheck creation identity before terminating this exact handle.
        import ctypes
        handle = runtime._checked(runtime._open(1 | 0x1000 | 0x100000, False, pid))
        try:
            times = [runtime.w.FILETIME() for _ in range(4)]
            runtime._checked(runtime._times(handle, *(ctypes.byref(t) for t in times)))
            actual = f"windows:{(times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}"
            assert actual == identity
            terminate = runtime._api("TerminateProcess", [runtime.w.HANDLE, runtime.w.UINT])
            runtime._checked(terminate(handle, 1))
        finally:
            runtime._close(handle)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def hold_teardown():
    """Inject a failed teardown in the real guardian, never in production code."""
    popen = subprocess.Popen
    bootstrap = '''import sys
from pathlib import Path
from songtool import runtime
output=Path(sys.argv[1])
stop=runtime._Tree.stop
def pending(tree, process):
    if not (output / 'allow-stop').exists():
        (output / 'teardown-waiting').touch()
        raise RuntimeError('Injected pending teardown')
    return stop(tree, process)
runtime._Tree.stop=pending
runtime._guardian(output)
'''
    def launch(argv, *args, **kwargs):
        if len(argv) > 3 and argv[2:4] == ["-m", "songtool.runtime"]:
            argv = [argv[0], "-B", "-c", bootstrap, argv[4]]
        return popen(argv, *args, **kwargs)
    with patch.object(subprocess, "Popen", launch):
        yield


def controller(root, boundary):
    def started(pid, identity):
        (root / "started.json").write_text(json.dumps([pid, identity]))
        (root / "started.ready").touch()
        if boundary == "before":
            time.sleep(60)
    with hold_teardown() if boundary == "after-held" else nullcontext():
        runtime.run_owned([sys.executable, "-B", "-c", WORKER, str(root / "ids.json")],
                          root / "out", workspace=root, device="cuda", on_started=started,
                          policy=replace(runtime._DefaultPolicy(), wall_time_seconds=30,
                                         max_output_bytes=8 * 1024**2))


def check_controller_death():
    for boundary in ("before", "after", "after-held"):
        with tempfile.TemporaryDirectory(prefix="songtool-owner-") as directory:
            root = Path(directory)
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                        "--controller", str(root), boundary], cwd=ROOT,
                                       stdout=subprocess.DEVNULL)
            controller_id = runtime.process_identity(process.pid)
            assert controller_id is not None
            identities = []
            try:
                wait_for(lambda: (root / "started.ready").exists())
                identities.append(json.loads((root / "started.json").read_text()))
                worker = json.loads((root / "out" / "process.json").read_text())
                workers = [[worker["pid"], worker["identity"]]]
                identities.extend(workers)
                if boundary.startswith("after"):
                    wait_for(lambda: (root / "ids.json.ready").exists())
                    workers.extend(json.loads((root / "ids.json").read_text()))
                    identities.extend(workers)
                try:
                    with runtime.gpu_lease(root):
                        raise AssertionError("Second lease acquired before controller death")
                except RuntimeError:
                    pass
                kill_exact(process.pid, controller_id)
                process.wait(timeout=10)
                if boundary == "after-held":
                    wait_for(lambda: (root / "out" / "teardown-waiting").exists())
                    assert all(runtime.process_matches(*item) for item in workers)
                    try:
                        with runtime.gpu_lease(root):
                            raise AssertionError("Lease released on failed teardown after controller death")
                    except RuntimeError:
                        pass
                    (root / "out" / "allow-stop").touch()
                deadline = time.monotonic() + 15
                while any(runtime.process_matches(*item) for item in identities):
                    assert time.monotonic() < deadline, ("Surviving owned identities", identities)
                    try:
                        with runtime.gpu_lease(root):
                            # Recheck under the lease: teardown may finish during acquisition.
                            assert not any(runtime.process_matches(*item) for item in workers), \
                                "GPU lease released while owned tree survives"
                    except RuntimeError:
                        pass
                    time.sleep(.01)
                wait_for(lambda: not runtime.process_matches(process.pid, controller_id))
                with runtime.gpu_lease(root):
                    pass
                if boundary == "before":
                    assert not (root / "ids.json").exists(), "Work released before controller acknowledgement"
                print(f"controller death {boundary} release: passed", flush=True)
            finally:
                if (root / "out").exists():
                    (root / "out" / "allow-stop").touch(exist_ok=True)
                kill_exact(process.pid, controller_id)
                process.wait(timeout=10)
                # Read receipts even if a preceding assertion failed.
                for path in (root / "out" / "process.json", root / "out" / "guardian.json"):
                    if path.exists():
                        record = json.loads(path.read_text())
                        identities.append([record["pid"], record["identity"]])
                if (root / "ids.json.ready").exists():
                    identities.extend(json.loads((root / "ids.json").read_text()))
                for pid, identity in reversed(identities):
                    kill_exact(pid, identity)
                wait_for(lambda: not any(runtime.process_matches(*item) for item in identities))


def check_cooperative():
    for mode in ("timeout", "cancel"):
        with tempfile.TemporaryDirectory(prefix="songtool-cancel-") as directory:
            root = Path(directory)
            event = threading.Event()
            errors = []
            def cancel_ready():
                try:
                    wait_for(lambda: (root / "ids.json.ready").exists())
                except BaseException as error:
                    errors.append(error)
                finally:
                    event.set()
            thread = threading.Thread(target=cancel_ready) if mode == "cancel" else None
            if thread:
                thread.start()
            try:
                result = runtime.run_owned(
                    [sys.executable, "-B", "-c", WORKER, str(root / "ids.json")],
                    root / "out", workspace=root, cancel=event,
                    policy=replace(runtime._DefaultPolicy(), wall_time_seconds=2 if mode == "timeout" else 20,
                                   max_output_bytes=8 * 1024**2))
                assert not errors, errors
                assert result.status == ("timed_out" if mode == "timeout" else "cancelled"), result
                identities = json.loads((root / "ids.json").read_text())
                assert all(not runtime.process_matches(*item) for item in identities)
                print(f"owned {mode}: passed", flush=True)
            finally:
                if thread:
                    thread.join(timeout=16)
                    assert not thread.is_alive()
                if (root / "ids.json.ready").exists():
                    identities = json.loads((root / "ids.json").read_text())
                    for pid, identity in reversed(identities):
                        kill_exact(pid, identity)
                    wait_for(lambda: not any(runtime.process_matches(*item) for item in identities))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--controller":
        controller(Path(sys.argv[2]), sys.argv[3])
    else:
        check_controller_death()
        check_cooperative()
