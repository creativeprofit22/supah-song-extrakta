"""Owned, budgeted argv execution; standard library only.

Call run_owned with a fresh run directory and an explicit shared workspace.
Persist on_started(pid, creation_identity) into job state before work is released.
Workers may call mark_inference_started immediately before model inference;
without that acknowledgement inference is 'unavailable', never inferred from exit.
POSIX ownership is a new session/process group: trusted workers must not detach
with setsid/setpgid. Windows Job Objects also contain detached descendants.
Disk budgets are sampled, not filesystem quotas; workers must write inside output.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .jobs import ResourcePolicy


@dataclass(frozen=True)
class _DefaultPolicy:
    # Mirrors jobs.ResourcePolicy without importing cleanup/numerical libraries.
    cpu_threads: int = 2
    max_active_operations: int = 1
    automatic_gpu: bool = False
    wall_time_seconds: int = 1800
    max_output_bytes: int = 2 * 1024 ** 3


if os.name == "nt":
    import ctypes
    from ctypes import wintypes as w

    _k = ctypes.WinDLL("kernel32", use_last_error=True)

    def _api(name, args, result=w.BOOL):
        fn = getattr(_k, name)
        fn.argtypes, fn.restype = args, result
        return fn

    _close = _api("CloseHandle", [w.HANDLE])
    _open = _api("OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE)
    _times = _api("GetProcessTimes", [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4)
    _wait = _api("WaitForSingleObject", [w.HANDLE, w.DWORD], w.DWORD)
    _create_job = _api("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE)
    _set_job = _api("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD])
    _assign = _api("AssignProcessToJobObject", [w.HANDLE, w.HANDLE])
    _terminate = _api("TerminateJobObject", [w.HANDLE, w.UINT])
    _query = _api("QueryInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p])

    class _Basic(ctypes.Structure):
        _fields_ = [("per_process", ctypes.c_longlong), ("per_job", ctypes.c_longlong),
                    ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class _Extended(ctypes.Structure):
        _fields_ = [("basic", _Basic), ("io", ctypes.c_ulonglong * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):
        _fields_ = [("times", ctypes.c_longlong * 4), ("faults", w.DWORD),
                    ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]

    def _checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value


def process_identity(pid: int) -> str | None:
    """Creation identity of a live process; None means dead, errors fail closed.

    Windows uses the native 100ns creation time. Linux uses boot ID/start ticks.
    Other POSIX systems fail closed until a native creation-identity API is added.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid PID")
    if os.name == "nt":
        handle = _open(0x1000 | 0x100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return None
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if _wait(handle, 0) == 0:
                return None
            times = [w.FILETIME() for _ in range(4)]
            _checked(_times(handle, *(ctypes.byref(t) for t in times)))
            return f"windows:{(times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}"
        finally:
            _close(handle)
    if not sys.platform.startswith("linux"):
        raise NotImplementedError("Native process creation identity is required on this platform")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"linux:{boot}:{fields[19]}"
    except FileNotFoundError:
        return None


def process_matches(pid: int, identity: str) -> bool:
    """Safe recovery predicate; never signal a PID merely because it exists."""
    return process_identity(pid) == identity


@contextmanager
def gpu_lease(workspace: Path):
    """Nonblocking same-workspace lease. Kernel releases dead owners' locks.

    Lock file is permanent: never unlink it (which would permit split ownership).
    Metadata is diagnostic only; acquisition depends on the OS lock, not a PID.
    """
    workspace = Path(workspace).resolve(strict=True)
    path = workspace / ".songtool-gpu.lock"
    if path.is_symlink() or (path.exists() and path.is_junction()):
        raise ValueError("GPU lease must be a regular local file")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Invalid GPU lease file")
        if os.fstat(fd).st_size == 0:
            os.write(fd, b" ")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Workspace GPU lease is busy or locking is unsupported") from error
        locked = True
        data = json.dumps({"pid": os.getpid(), "identity": process_identity(os.getpid())}).encode()
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, data)
        os.ftruncate(fd, len(data) + 1)
        os.fsync(fd)
        yield
    finally:
        if locked and os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


@dataclass(frozen=True)
class RuntimeResult:
    status: str
    returncode: int
    elapsed_seconds: float
    requested_device: str
    process_id: int
    process_identity: str
    output_bytes: int
    log_bytes: int
    log_discarded_bytes: int
    inference_ran: bool | str
    telemetry: str = "unavailable"


def mark_inference_started() -> None:
    """Worker acknowledgement, called at the actual inference boundary."""
    with Path(os.environ["SONGTOOL_INFERENCE_MARKER"]).open("xb") as stream:
        stream.write(b"inference started\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bytes(directory: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(root) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Output links/reparse points are forbidden")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _group_alive(pid):
    """Linux group check, including descendants whose leader has already exited."""
    # simplification: owned workers do not detach; use cgroups for daemonizing
    # executables. The subreaper still retains the lease if an adoptee survives.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pid and fields[0] != "Z":
            return True
    return False


class _Tree:
    def __init__(self):
        self.job = None
        self.handles = {}
        if os.name == "nt":
            self.job = _checked(_create_job(None, None))
            limits = _Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            try:
                _checked(_set_job(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            except BaseException:
                _close(self.job)
                raise

    def attach(self, process):
        if self.job:
            _checked(_assign(self.job, int(process._handle)))

    def stop(self, process):
        # Do not close ownership handles on failure: the guardian retries while
        # holding its lease, including when ActiveProcesses precedes real exit.
        if self.job:
            class Members(ctypes.Structure):
                _fields_ = [("assigned", w.DWORD), ("count", w.DWORD),
                            ("pids", ctypes.c_size_t * 65536)]
            members = Members()
            _checked(_query(self.job, 3, ctypes.byref(members), ctypes.sizeof(members), None))
            for pid in members.pids[:members.count]:
                if pid in self.handles:
                    continue
                handle = _open(0x100000, False, pid)
                if handle:
                    self.handles[pid] = handle
                elif ctypes.get_last_error() != 87:
                    raise ctypes.WinError(ctypes.get_last_error())
            _checked(_terminate(self.job, 1))
            for handle in self.handles.values():
                if _wait(handle, 10000) != 0:
                    raise RuntimeError("Owned process did not signal exit")
            deadline = time.monotonic() + 10
            while True:
                info = _Accounting()
                _checked(_query(self.job, 1, ctypes.byref(info), ctypes.sizeof(info), None))
                if not info.active:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned Windows process tree did not terminate")
                time.sleep(0.01)
        elif process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while _group_alive(process.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned POSIX process group did not terminate")
                time.sleep(0.01)
        if process is not None:
            if process.poll() is None:
                process.kill()  # also handles failure before job assignment
            process.wait()
        if self.job:
            for handle in self.handles.values():
                _close(handle)
            self.handles.clear()
            _close(self.job)
            self.job = None


_BOOTSTRAP = """import os,sys,subprocess
if sys.stdin.buffer.read(1) != b'G': sys.exit(125)
sys.stdin.close()
args=sys.argv[1:]
if os.name != 'nt': os.execvpe(args[0],args,os.environ)
sys.exit(subprocess.call(args,stdin=subprocess.DEVNULL))
"""


def run_owned(argv: list[str], output: Path, *, workspace: Path,
              policy: ResourcePolicy | _DefaultPolicy | None = None, device: str = "cpu",
              cancel: threading.Event | None = None,
              on_started: Callable[[int, str], None] | None = None,
              log_limit: int = 1024 * 1024, poll_seconds: float = 0.05) -> RuntimeResult:
    """Execute once; return terminal accounting for exit/failure/time/disk/cancel.

    output must not exist; logs/identity/result are created exclusively there.
    output_bytes counts the entire directory before the final runtime receipt.
    Only one active run per job must be committed by the caller before invoking.
    argv is trusted application argv, never shell text. CPU hides CUDA only in
    the child environment; cuda requires explicit selection and workspace lease.
    Callback exceptions/KeyboardInterrupt tear down the tree and propagate.
    Inference remains unavailable unless the worker acknowledges it. The caller
    owns job-state/technical publication; exit zero is not listening approval.
    """
    policy = _DefaultPolicy() if policy is None else policy
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or "\0" in a for a in argv):
        raise ValueError("Expected nonempty argv")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or explicit cuda")
    if (type(policy.cpu_threads) is not int or not 1 <= policy.cpu_threads <= 64
            or policy.max_active_operations != 1 or policy.automatic_gpu is not False
            or not math.isfinite(policy.wall_time_seconds) or policy.wall_time_seconds <= 0
            or type(policy.max_output_bytes) is not int or policy.max_output_bytes <= 0
            or type(log_limit) is not int or log_limit <= 0
            or not math.isfinite(poll_seconds) or not 0.01 <= poll_seconds <= 1):
        raise ValueError("Invalid resource budget")
    # Fail before spawn on platforms lacking recoverable creation identities.
    process_identity(os.getpid())
    output = Path(output)
    output = output.parent.resolve(strict=True) / output.name
    workspace = Path(workspace).resolve(strict=True)
    if shutil.disk_usage(output.parent).free < policy.max_output_bytes + log_limit:
        raise ValueError("Insufficient output space for declared budget")
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = str(policy.cpu_threads)
    env["SONGTOOL_DEVICE"] = device
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    marker = output / "inference-started"
    env["SONGTOOL_INFERENCE_MARKER"] = str(marker)
    env["SONGTOOL_OUTPUT_DIRECTORY"] = str(output)
    result = _supervise(argv, output, workspace, env, policy, device, cancel,
                        on_started, log_limit, poll_seconds)
    with (output / "runtime.json").open("x", encoding="utf-8") as stream:
        json.dump(asdict(result), stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def _record(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _supervise(argv, output, workspace, env, policy, device, cancel, on_started,
               log_limit, poll_seconds):
    # Only this controller holds the write end. Neither guardian nor workers
    # inherit it: EOF survives os._exit/SIGKILL, unlike controller finally blocks.
    start = time.monotonic()
    guardian = subprocess.Popen([sys.executable, "-B", "-m", "songtool.runtime", str(output)],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, env=env, close_fds=True,
                                start_new_session=os.name != "nt")
    requested_status = None
    try:
        identity = process_identity(guardian.pid)
        if identity is None:
            raise RuntimeError("Guardian died before ownership publication")
        _record(output / "guardian.json", {"pid": guardian.pid, "identity": identity})
        configuration = {"argv": argv, "workspace": str(workspace),
                         "policy": asdict(policy), "device": device,
                         "log_limit": log_limit, "poll_seconds": poll_seconds}
        guardian.stdin.write(json.dumps(configuration).encode() + b"\n")
        guardian.stdin.flush()
        released = False
        while guardian.poll() is None:
            if cancel is not None and cancel.is_set():
                requested_status = "cancelled"
                break
            if time.monotonic() - start >= policy.wall_time_seconds:
                requested_status = "timed_out"
                break
            if not released and (output / "guardian-ready").exists():
                if on_started:
                    on_started(guardian.pid, identity)
                guardian.stdin.write(b"G")
                guardian.stdin.flush()
                released = True
            time.sleep(poll_seconds)
    finally:
        # Never kill the guardian to implement a timeout: it owns teardown and
        # retains the lease if the kernel cannot yet stop an owned descendant.
        try:
            guardian.stdin.close()
        except BrokenPipeError:
            pass
        guardian.wait()
    error_path = output / "guardian-error.json"
    if error_path.exists():
        raise RuntimeError(json.loads(error_path.read_text(encoding="utf-8"))["error"])
    if guardian.returncode != 0:
        raise RuntimeError("Guardian failed; ownership requires explicit recovery")
    data = json.loads((output / "guardian-result.json").read_text(encoding="utf-8"))
    if requested_status:
        data["status"] = requested_status
    data["elapsed_seconds"] = time.monotonic() - start
    return RuntimeResult(**data)


def _guardian(output):
    from contextlib import nullcontext
    config_line = sys.stdin.buffer.readline()
    if not config_line:
        return  # Controller died before publishing configuration; no worker exists.
    config = json.loads(config_line)
    policy = _DefaultPolicy(**config["policy"])
    cancelled, released = threading.Event(), threading.Event()

    def watch_controller():
        try:
            # Raw reads avoid holding Python's buffered-stdin lock during
            # normal interpreter shutdown while the controller is still alive.
            if os.read(sys.stdin.fileno(), 1) == b"G":
                released.set()
                os.read(sys.stdin.fileno(), 1)  # EOF (or any further byte) means stop.
        finally:
            cancelled.set()
            released.set()

    threading.Thread(target=watch_controller, daemon=True).start()

    def ready(pid, identity):
        (output / "guardian-ready").touch(exist_ok=False)
        if not released.wait(policy.wall_time_seconds):
            cancelled.set()

    try:
        if sys.platform.startswith("linux"):
            # Adopt/reap orphaned grandchildren, not just the direct worker.
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            prctl = libc.prctl
            prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
            prctl.restype = ctypes.c_int
            if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise OSError(ctypes.get_errno(), "Cannot establish owned subreaper")
        with gpu_lease(Path(config["workspace"])) if config["device"] == "cuda" else nullcontext():
            result = _run(config["argv"], output, os.environ.copy(), policy, config["device"],
                          cancelled, ready, config["log_limit"], config["poll_seconds"],
                          output / "inference-started")
            _record(output / "guardian-result.json", asdict(result))
    except BaseException as error:
        _record(output / "guardian-error.json", {"error": f"{type(error).__name__}: {str(error)[:2000]}"})


def _run(argv, output, env, policy, device, cancel, on_started, log_limit, poll_seconds, marker):
    start = time.monotonic()
    process = None
    reader = None
    counts = [0, 0]
    errors = []
    status = "failed"
    with (output / "worker.log").open("xb") as log:
        tree = _Tree()
        try:
            (output / "guardian-started").touch(exist_ok=False)
            process = subprocess.Popen([sys.executable, "-B", "-c", _BOOTSTRAP, *argv],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, env=env, shell=False, close_fds=True,
                                       start_new_session=os.name != "nt")
            tree.attach(process)
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("Worker died before identity publication")
            with (output / "process.json").open("x", encoding="utf-8") as stream:
                json.dump({"pid": process.pid, "identity": identity, "device": device}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if on_started:
                on_started(process.pid, identity)

            def drain():
                try:
                    while chunk := process.stdout.read(65536):
                        keep = min(len(chunk), max(0, log_limit - counts[0]))
                        log.write(chunk[:keep])
                        log.flush()
                        counts[0] += keep
                        counts[1] += len(chunk) - keep
                except BaseException as error:
                    errors.append(error)

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            if ((cancel is None or not cancel.is_set())
                    and time.monotonic() - start < policy.wall_time_seconds):
                process.stdin.write(b"G")
                process.stdin.flush()
            process.stdin.close()
            while True:
                if cancel is not None and cancel.is_set():
                    status = "cancelled"
                    break
                if time.monotonic() - start >= policy.wall_time_seconds:
                    status = "timed_out"
                    break
                if (_bytes(output) > policy.max_output_bytes
                        or shutil.disk_usage(output).free < log_limit):
                    status = "output_limit"
                    break
                if errors:
                    raise RuntimeError("Worker log write failed") from errors[0]
                if os.name == "nt":
                    code = process.poll()
                else:
                    # Keep the leader unreaped, pinning its PID/process group
                    # until killpg and descendant verification have finished.
                    exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    code = (0 if exited.si_code == os.CLD_EXITED and exited.si_status == 0 else 1) if exited else None
                if code is not None:
                    status = "completed" if code == 0 else "failed"
                    break
                time.sleep(poll_seconds)
        finally:
            while True:
                try:
                    tree.stop(process)
                    break
                except (OSError, RuntimeError):
                    # Fail closed: retain guardian identity and GPU lease until
                    # the exact owned tree really stops (e.g. uninterruptible I/O).
                    time.sleep(poll_seconds)
            if sys.platform.startswith("linux"):
                # The guardian is a subreaper; all adopted children must be dead
                # and reaped before it releases either ownership or the lease.
                while True:
                    try:
                        os.waitpid(-1, 0)
                    except ChildProcessError:
                        break
            (output / "guardian-stopped").touch(exist_ok=False)
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if reader:
                    reader.join(timeout=10)
                    if reader.is_alive():
                        raise RuntimeError("Owned descendant retained the diagnostic pipe")
                process.stdout.close()
        if errors:
            raise RuntimeError("Worker log write failed") from errors[0]
    size = _bytes(output)
    if size > policy.max_output_bytes:
        status = "output_limit"
    return RuntimeResult(status, process.returncode, time.monotonic() - start, device,
                         process.pid, identity, size, *counts,
                         True if marker.is_file() else "unavailable")


if __name__ == "__main__":
    _guardian(Path(sys.argv[1]))
