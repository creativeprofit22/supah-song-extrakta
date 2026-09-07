"""Bounded local audio jobs (schema 1), not an automatic treatment workflow.

Only fixed revision filenames commit state. Pending files and unregistered media
are diagnostics, never candidates. Requires a private, trusted local filesystem:
links/reparse points are rejected, but this is not a sandbox against another
process maliciously replacing directories between filesystem calls. Hard-link
publication is tested before import; fsync protects file contents, not against
physical disk loss. History is limited to 4096 revisions, 256 versions/runs,
2000 feedback/protection records, 1 MiB per snapshot and 64 KiB per receipt.

Frozen records use tuples, not mutable metadata dictionaries. Use dataclasses.replace
and commit_revision with the loaded revision/digest for optimistic concurrency.
Explicit selection alone promotes a verified candidate; registration and review
never promote. Recovery never reruns work; copies publish completion last.
"""

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Literal
from uuid import uuid4

from .cleanup import audio_info, digest, execute, require
from .media import extract

SCHEMA_VERSION = 1
RATE = 48000
MAX_FRAMES = 600 * RATE
MAX_MEDIA_BYTES = 1024 ** 3
MAX_METADATA_BYTES = 1024 ** 2
MAX_RECEIPT_BYTES = 65536
MAX_REVISIONS = 4096
MAX_RECORDS = 256
MAX_FEEDBACK = 2000

Intent = Literal["music", "spoken_audio", "unknown"]
Technical = Literal["not_run", "passed", "failed"]
Listening = Literal["unreviewed", "better", "worse", "uncertain"]


class RevisionConflict(RuntimeError):
    """Reload status; do not retry processing from stale state."""


def new_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class ResourcePolicy:
    cpu_threads: int = 2
    max_active_operations: int = 1
    automatic_gpu: bool = False
    wall_time_seconds: int = 1800
    max_output_bytes: int = 2 * 1024 ** 3


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str
    container: str


@dataclass(frozen=True)
class Version:
    id: str
    path: str
    sha256: str
    frames: int
    subtype: str
    role: str
    parent_id: str | None = None
    parent_sha256: str | None = None
    # Exact translation into canonical source frames; never inferred by duration.
    source_start_frame: int = 0
    run_id: str | None = None
    technical: Technical = "not_run"
    listening: Listening = "unreviewed"
    receipt_sha256: str = ""


@dataclass(frozen=True)
class Run:
    id: str
    operation: str
    fingerprint: str
    parent_id: str
    parent_sha256: str
    execution: Literal["running", "completed", "failed", "interrupted"] = "running"
    technical: Technical = "not_run"
    device: Literal["cpu", "cuda"] = "cpu"
    process_id: int | None = None
    process_identity: str | None = None
    receipt_sha256: str | None = None


@dataclass(frozen=True)
class Feedback:
    id: str
    version_id: str
    version_sha256: str
    category: str
    note: str
    scope: Literal["whole", "interval"]
    start_frame: int | None = None
    end_frame: int | None = None
    accepted: bool = False


@dataclass(frozen=True)
class ProtectedRange:
    version_id: str
    version_sha256: str
    start_frame: int
    end_frame: int
    feedback_id: str


@dataclass(frozen=True)
class Job:
    id: str
    source: Source
    intent: Intent = "unknown"
    wanted_vocals_may_include_rap: bool = False
    policy: ResourcePolicy = ResourcePolicy()
    versions: tuple[Version, ...] = ()
    runs: tuple[Run, ...] = ()
    feedback: tuple[Feedback, ...] = ()
    protected_ranges: tuple[ProtectedRange, ...] = ()
    current_version: str | None = None
    latest_attempt: str | None = None
    latest_technically_passed_candidate: str | None = None
    schema_version: int = SCHEMA_VERSION
    revision: int = 0
    previous_revision_sha256: str | None = None
    # Derived from committed bytes, deliberately excluded from serialization.
    revision_sha256: str | None = None


def _integer(value: int, low: int, high: int, name: str) -> None:
    require(type(value) is int and low <= value <= high, f"Invalid {name}.")


def _text(value: str, name: str, maximum: int = 256) -> None:
    require(type(value) is str and 0 < len(value.encode("utf-8")) <= maximum
            and "\x00" not in value, f"Invalid {name}.")


def _id(value: str) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None, "Invalid generated ID.")


def _hash(value: str) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA256.")


def _plain(path: Path, *, directory: bool = False) -> Path:
    """Check every existing component without resolving away a link."""
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        info = component.lstat()
        require(not stat.S_ISLNK(info.st_mode)
                and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                f"Links/reparse points are not allowed: {component}")
    info = path.stat()
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            f"Not a regular {'directory' if directory else 'file'}: {path}")
    if not directory:
        require(info.st_nlink == 1, f"Hard-linked media/metadata is not allowed: {path}")
    return path


def _internal(root: Path, relative: str) -> Path:
    require(type(relative) is str and re.fullmatch(r"[A-Za-z0-9_./-]+", relative) is not None
            and not relative.startswith("/") and all(p not in ("", ".", "..") for p in relative.split("/")),
            "Invalid internal relative path.")
    return root.joinpath(*relative.split("/"))


def _data(job: Job) -> dict:
    value = asdict(job)
    del value["revision_sha256"]
    return value


def _encode(value: dict, limit: int) -> bytes:
    try:
        data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Invalid JSON metadata.") from error
    require(len(data) <= limit, f"Metadata exceeds {limit} byte ceiling; history cannot be pruned.")
    return data


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key.")
        result[key] = value
    return result


def _read(path: Path, limit: int) -> tuple[dict, str]:
    # Published JSON has a retained staging hard link, unlike job media.
    _plain(path.parent, directory=True)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT and info.st_size <= limit, "Invalid/oversized metadata.")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "Oversized metadata.")
    try:
        value = json.loads(data, object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Malformed metadata.") from error
    require(type(value) is dict, "Metadata must be an object.")
    _encode(value, limit)
    return value, hashlib.sha256(data).hexdigest()


def _record(cls, value: dict):
    require(type(value) is dict and set(value) == {f.name for f in fields(cls)}, f"Invalid {cls.__name__} fields.")
    return cls(**value)


def _decode(value: dict, sha256: str) -> Job:
    value = dict(value)
    require("revision_sha256" not in value, "Digest must not be embedded in snapshot.")
    value["source"] = _record(Source, value.get("source"))
    value["policy"] = _record(ResourcePolicy, value.get("policy"))
    for name, cls, limit in (("versions", Version, MAX_RECORDS), ("runs", Run, MAX_RECORDS),
                             ("feedback", Feedback, MAX_FEEDBACK), ("protected_ranges", ProtectedRange, MAX_FEEDBACK)):
        records = value.get(name)
        require(type(records) is list and len(records) <= limit, f"Invalid/bounded {name}.")
        value[name] = tuple(_record(cls, record) for record in records)
    value["revision_sha256"] = sha256
    job = _record(Job, value)
    validate_job(job)
    return job


def validate_job(job: Job) -> None:
    """Validate typed boundaries, references, limits and unambiguous frame maps."""
    require(type(job) is Job and type(job.source) is Source and type(job.policy) is ResourcePolicy, "Invalid job records.")
    _id(job.id)
    _integer(job.schema_version, 1, 1, "schema version")
    _integer(job.revision, 0, MAX_REVISIONS, "revision")
    for value in (job.previous_revision_sha256, job.revision_sha256):
        if value is not None:
            _hash(value)
    require(job.intent in ("music", "spoken_audio", "unknown"), "Invalid intent.")
    require(type(job.wanted_vocals_may_include_rap) is bool, "Invalid vocal intent.")
    policy = job.policy
    _integer(policy.cpu_threads, 1, 64, "CPU threads")
    _integer(policy.max_active_operations, 1, 1, "active operations")
    require(policy.automatic_gpu is False, "Automatic GPU use is not supported.")
    _integer(policy.wall_time_seconds, 1, 86400, "wall time")
    _integer(policy.max_output_bytes, 1, 64 * 1024 ** 3, "output budget")
    source = job.source
    require(type(source.path) is str and re.fullmatch(r"source/original\.[a-z0-9]{1,10}", source.path) is not None,
            "Invalid source path.")
    _hash(source.sha256)
    _integer(source.bytes, 1, MAX_MEDIA_BYTES, "source byte size")
    require(type(source.duration_seconds) in (int, float) and math.isfinite(source.duration_seconds)
            and 0 < source.duration_seconds <= 600, "Source exceeds ten-minute duration limit.")
    _integer(source.sample_rate, 1, 768000, "source sample rate")
    _integer(source.channels, 1, 64, "source channels")
    _text(source.codec, "codec")
    _text(source.container, "container")
    for records, cls, limit in ((job.versions, Version, MAX_RECORDS), (job.runs, Run, MAX_RECORDS),
                              (job.feedback, Feedback, MAX_FEEDBACK), (job.protected_ranges, ProtectedRange, MAX_FEEDBACK)):
        require(type(records) is tuple and len(records) <= limit and all(type(r) is cls for r in records),
                f"Invalid {cls.__name__} records or history limit reached; no history is dropped.")
    require(bool(job.versions), "Job requires a canonical version.")
    versions = {}
    for version in job.versions:
        _id(version.id)
        require(version.id not in versions, "Duplicate version ID.")
        require(version.path == f"versions/{version.id}/audio.wav", "Invalid version path.")
        _hash(version.sha256)
        _hash(version.receipt_sha256)
        _integer(version.frames, 1, MAX_FRAMES, "frames")
        _integer(version.source_start_frame, 0, MAX_FRAMES - version.frames, "source offset")
        _text(version.subtype, "subtype")
        _text(version.role, "role")
        require(version.technical in ("not_run", "passed", "failed")
                and version.listening in ("unreviewed", "better", "worse", "uncertain"), "Invalid version status.")
        if not versions:
            require(version.parent_id is None and version.parent_sha256 is None and version.source_start_frame == 0
                    and version.role == "canonical" and version.technical == "passed", "Invalid canonical root.")
        else:
            require(version.parent_id in versions, "Version must have an earlier explicit parent.")
            parent = versions[version.parent_id]
            require(version.parent_sha256 == parent.sha256, "Parent hash mismatch.")
            require(parent.source_start_frame <= version.source_start_frame
                    and version.source_start_frame + version.frames <= parent.source_start_frame + parent.frames,
                    "Ambiguous/out-of-parent timeline map.")
        versions[version.id] = version
    runs = {}
    for run in job.runs:
        _id(run.id)
        require(run.id not in runs, "Duplicate run ID.")
        _text(run.operation, "operation")
        _hash(run.fingerprint)
        require(run.parent_id in versions and run.parent_sha256 == versions[run.parent_id].sha256, "Invalid run parent.")
        require(run.execution in ("running", "completed", "failed", "interrupted")
                and run.technical in ("not_run", "passed", "failed") and run.device in ("cpu", "cuda"), "Invalid run status.")
        require(run.technical != "passed" or run.execution == "completed", "Only completed runs can pass.")
        require((run.process_id is None) == (run.process_identity is None), "Incomplete process identity.")
        if run.process_id is not None:
            _integer(run.process_id, 1, 2 ** 32 - 1, "PID")
            _text(run.process_identity, "process identity")
        if run.receipt_sha256 is not None:
            _hash(run.receipt_sha256)
        runs[run.id] = run
    require(sum(r.execution == "running" for r in job.runs) <= 1, "Only one active run per job.")
    for version in job.versions:
        require(version.run_id is None or version.run_id in runs, "Unknown producing run.")
    feedback = {}
    for item in job.feedback:
        _id(item.id)
        require(item.id not in feedback, "Duplicate feedback ID.")
        require(item.version_id in versions and item.version_sha256 == versions[item.version_id].sha256, "Invalid feedback parent.")
        require(item.category in ("good", "residual_dialogue", "wanted_vocal_loss", "muffling", "warbling_reverse_like_artifact",
                                  "noise", "uncertain"), "Invalid feedback category.")
        _text(item.note, "original feedback wording", 8192)
        require(type(item.accepted) is bool and (not item.accepted or item.category == "good"), "Invalid acceptance.")
        require(item.scope in ("whole", "interval"), "Invalid feedback scope.")
        if item.scope == "interval":
            _integer(item.start_frame, 0, versions[item.version_id].frames - 1, "feedback start")
            _integer(item.end_frame, item.start_frame + 1, versions[item.version_id].frames, "feedback end")
        else:
            require(item.start_frame is None and item.end_frame is None, "Whole feedback has no interval.")
        feedback[item.id] = item
    for item in job.protected_ranges:
        require(item.version_id in versions and item.version_sha256 == versions[item.version_id].sha256, "Invalid protection parent.")
        _integer(item.start_frame, 0, versions[item.version_id].frames - 1, "protected start")
        _integer(item.end_frame, item.start_frame + 1, versions[item.version_id].frames, "protected end")
        require(item.feedback_id in feedback and feedback[item.feedback_id].accepted, "Protection requires accepted feedback.")
        accepted = feedback[item.feedback_id]
        origin = versions[accepted.version_id]
        target = versions[item.version_id]
        ancestor = target
        while ancestor.id != origin.id and ancestor.parent_id is not None:
            ancestor = versions[ancestor.parent_id]
        require(ancestor.id == origin.id, "Protection requires accepted ancestor feedback.")
        start = origin.source_start_frame + (accepted.start_frame if accepted.scope == "interval" else 0)
        end = origin.source_start_frame + (accepted.end_frame if accepted.scope == "interval" else origin.frames)
        require(start <= target.source_start_frame + item.start_frame
                < target.source_start_frame + item.end_frame <= end,
                "Protection lies outside the accepted source-time interval.")
    require(job.current_version in versions, "Missing current version.")
    current = versions[job.current_version]
    require(current.technical == "passed", "Current version requires technical pass.")
    require(job.latest_attempt is None or job.latest_attempt in runs, "Invalid latest attempt.")
    require(job.latest_technically_passed_candidate is None or
            (job.latest_technically_passed_candidate in versions and
             versions[job.latest_technically_passed_candidate].technical == "passed"), "Invalid passed candidate.")
    _encode(_data(job), MAX_METADATA_BYTES)


def _publish(path: Path, data: bytes) -> None:
    """Report-last publication, no overwrite fallback and no fallible post-commit cleanup."""
    _plain(path.parent, directory=True)
    staged = path.parent / f".pending-{new_id()}"
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staged, path)


def _test_publication(directory: Path) -> None:
    first, second = directory / f".probe-{new_id()}", directory / f".probe-{new_id()}"
    with first.open("xb") as stream:
        stream.write(b"publication probe")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(first, second)
        try:
            os.link(first, second)
        except FileExistsError:
            pass
        else:
            raise ValueError("Filesystem does not provide exclusive link publication.")
        require(second.read_bytes() == b"publication probe", "Filesystem publication probe failed.")
    finally:
        if second.exists():
            second.unlink()
        first.unlink()


def _verify_media(root: Path, job: Job) -> None:
    source = _plain(_internal(root, job.source.path))
    require(source.stat().st_size == job.source.bytes and digest(source) == job.source.sha256, "Source hash/size mismatch.")
    for version in job.versions:
        path = _plain(_internal(root, version.path))
        require(path.stat().st_size <= MAX_MEDIA_BYTES and digest(path) == version.sha256, "Version hash mismatch.")
        audio_info(path, version.frames, version.subtype)
        receipt, sha = _read(path.parent / "receipt.json", MAX_RECEIPT_BYTES)
        expected = asdict(version)
        del expected["receipt_sha256"]
        # Listening is mutable state, not part of the immutable creation receipt.
        del expected["listening"]
        require(sha == version.receipt_sha256 and receipt.get("version") == expected
                and receipt.get("job_id") == job.id and receipt.get("source_sha256") == job.source.sha256,
                "Version receipt mismatch.")
    for run in job.runs:
        if run.receipt_sha256 is None:
            continue
        receipt, sha = _read(root / "runs" / run.id / "receipt.json", MAX_RECEIPT_BYTES)
        require(sha == run.receipt_sha256, "Run receipt hash mismatch.")
        if "legacy_evidence" in receipt:
            evidence = receipt["legacy_evidence"]
            require(receipt.get("run") == asdict(replace(run, receipt_sha256=None))
                    and receipt.get("job_id") == job.id
                    and receipt.get("source_sha256") == job.source.sha256
                    and receipt.get("fingerprint") == run.fingerprint
                    and run.execution == "failed" and run.technical == "not_run"
                    and receipt.get("failure_kind") == "legacy_evidence"
                    and receipt.get("verification") is None and receipt.get("version") is None,
                    "Invalid native legacy failure receipt.")
            require(evidence.get("path") == f"runs/{run.id}/legacy-evidence.json", "Invalid legacy evidence path.")
            _hash(evidence.get("sha256"))
            _, evidence_sha = _read(_plain(_internal(root, evidence["path"])), MAX_RECEIPT_BYTES)
            require(evidence_sha == evidence["sha256"], "Legacy evidence hash mismatch.")


def load_job(directory: Path) -> Job:
    """Read every consecutive hash-linked revision, fail closed, verify all registered media."""
    root = _plain(Path(directory), directory=True)
    if (root / "copy-intent.json").exists():
        marker, _ = _read(root / "copy-complete.json", MAX_RECEIPT_BYTES)
        intent, _ = _read(root / "copy-intent.json", MAX_RECEIPT_BYTES)
        require(marker == intent, "Incomplete or mismatched job copy.")
    state = _plain(root / "state", directory=True)
    names = []
    with os.scandir(state) as entries:
        for entry in entries:
            if entry.name.startswith((".pending-", ".probe-")):
                continue
            require(re.fullmatch(r"[0-9]{8}\.json", entry.name) is not None, "Unexpected state entry.")
            names.append(entry.name)
            require(len(names) <= MAX_REVISIONS, "Revision history limit reached.")
    require(bool(names), "Incomplete job: no committed revision.")
    previous = None
    for index, name in enumerate(sorted(names), 1):
        require(name == f"{index:08d}.json", "Missing revision; refusing older state.")
        value, sha = _read(state / name, MAX_METADATA_BYTES)
        job = _decode(value, sha)
        require(job.revision == index and job.previous_revision_sha256 == (previous.revision_sha256 if previous else None),
                "Broken revision hash chain.")
        if previous is not None:
            _transition(previous, job)
        else:
            require(job.current_version == job.versions[0].id, "Initial current version must be canonical.")
        previous = job
    _verify_media(root, job)
    return job


def _transition(before: Job, after: Job) -> None:
    if before.current_version != after.current_version:
        selected = next(v for v in after.versions if v.id == after.current_version)
        require(selected.technical == "passed" and selected.listening == "better",
                "Promotion requires technical pass and explicit better verdict.")
    require((before.id, before.source, before.intent, before.wanted_vocals_may_include_rap) ==
            (after.id, after.source, after.intent, after.wanted_vocals_may_include_rap), "Job/source identity is immutable.")
    require(len(after.versions) >= len(before.versions), "Version history cannot be removed.")
    for old, new in zip(before.versions, after.versions):
        require(replace(old, listening=new.listening) == new, "Registered version provenance is immutable.")
    require(after.feedback[:len(before.feedback)] == before.feedback
            and after.protected_ranges[:len(before.protected_ranges)] == before.protected_ranges,
            "Feedback/protection history cannot be removed or rewritten.")
    require(len(after.runs) >= len(before.runs), "Run history cannot be removed.")
    for old, new in zip(before.runs, after.runs):
        require(replace(old, execution=new.execution, technical=new.technical, process_id=new.process_id,
                        process_identity=new.process_identity, receipt_sha256=new.receipt_sha256) == new,
                "Run intent is immutable.")
        require(old.execution == "running" or old == new, "Terminal run is immutable.")


def commit_revision(directory: Path, job: Job) -> Job:
    """Commit a replacement of a loaded snapshot; stale revision/digest raises RevisionConflict."""
    validate_job(job)
    root = _plain(Path(directory), directory=True)
    before = load_job(root)
    if (job.revision, job.revision_sha256) != (before.revision, before.revision_sha256):
        raise RevisionConflict("Job changed; reload status before acting.")
    _transition(before, job)
    require(before.revision < MAX_REVISIONS, "Revision limit reached; history cannot be pruned.")
    candidate = replace(job, revision=before.revision + 1, previous_revision_sha256=before.revision_sha256,
                        revision_sha256=None)
    validate_job(candidate)
    _verify_media(root, candidate)
    data = _encode(_data(candidate), MAX_METADATA_BYTES)
    try:
        _publish(root / "state" / f"{candidate.revision:08d}.json", data)
    except FileExistsError as error:
        raise RevisionConflict("Concurrent commit won; reload status, do not repeat processing.") from error
    return replace(candidate, revision_sha256=hashlib.sha256(data).hexdigest())


def resolve_version(directory: Path, version_id: str = "current") -> tuple[Version, Path]:
    """Return explicitly identified, hash-verified media; never use mtime or a latest alias."""
    job = load_job(directory)
    identifier = job.current_version if version_id == "current" else version_id
    _id(identifier)
    for version in job.versions:
        if version.id == identifier:
            return version, _internal(_plain(Path(directory), directory=True), version.path)
    raise ValueError("Unknown version ID.")


def mapped_protected_ranges(job: Job, version_id: str) -> tuple[ProtectedRange, ...]:
    """Translate accepted ancestor feedback into a descendant; clip at trim edges.

    Sibling/descendant feedback never flows backwards or sideways. Returned ranges
    retain the original feedback ID and do not imply a whole-version verdict.
    """
    validate_job(job)
    versions = {v.id: v for v in job.versions}
    require(version_id in versions, "Unknown protection target.")
    target = versions[version_id]
    ancestors = set()
    cursor = target
    while True:
        ancestors.add(cursor.id)
        if cursor.parent_id is None:
            break
        cursor = versions[cursor.parent_id]
    result = []
    for item in job.feedback:
        if not item.accepted or item.version_id not in ancestors:
            continue
        origin = versions[item.version_id]
        shift = origin.source_start_frame - target.source_start_frame
        first = max(0, shift + (item.start_frame if item.scope == "interval" else 0))
        last = min(target.frames, shift + (item.end_frame if item.scope == "interval" else origin.frames))
        if first < last:
            result.append(ProtectedRange(target.id, target.sha256, first, last, item.id))
    return tuple(result)


def record_feedback(directory: Path, *, version_id: str, category: str, note: str,
                    scope: Literal["whole", "interval"], start_frame: int | None = None,
                    end_frame: int | None = None, accepted: bool = False,
                    expected_revision: int, expected_revision_sha256: str) -> Job:
    """Persist verbatim wording and explicit scope; acceptance protects, never promotes."""
    job = load_job(directory)
    if (job.revision, job.revision_sha256) != (expected_revision, expected_revision_sha256):
        raise RevisionConflict("Job changed; reload before feedback.")
    version = next((v for v in job.versions if v.id == version_id), None)
    require(version is not None, "Unknown feedback version.")
    item = Feedback(new_id(), version.id, version.sha256, category, note, scope,
                    start_frame, end_frame, accepted)
    candidate = replace(job, feedback=(*job.feedback, item))
    validate_job(candidate)
    if accepted:
        protection = ProtectedRange(version.id, version.sha256,
                                    start_frame if scope == "interval" else 0,
                                    end_frame if scope == "interval" else version.frames, item.id)
        candidate = replace(candidate, protected_ranges=(*job.protected_ranges, protection))
    return commit_revision(directory, candidate)


def _finite_audio(path: Path) -> None:
    # Bounded CPU-only decoding; a valid header alone does not establish finite audio.
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(path) as stream:
        for block in stream.blocks(blocksize=RATE, dtype="float64", always_2d=True):
            require(bool(np.isfinite(block).all()), "Nonfinite decoded audio.")


def _copy(source: Path, destination: Path) -> None:
    source = _plain(source)
    require(0 < source.stat().st_size <= MAX_MEDIA_BYTES, "Invalid/oversized input.")
    before = digest(source)
    with source.open("rb") as src, destination.open("xb") as dst:
        remaining = MAX_MEDIA_BYTES
        while block := src.read(min(1024 * 1024, remaining + 1)):
            remaining -= len(block)
            require(remaining >= 0, "Input grew beyond media limit.")
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    require(digest(destination) == before == digest(source), "Source changed during copy.")


def _version_receipt(root: Path, job_id: str, source_sha256: str, version: Version, details: dict) -> Version:
    value = asdict(version)
    del value["receipt_sha256"]
    del value["listening"]
    receipt = {"schema_version": 1, "job_id": job_id, "source_sha256": source_sha256,
               "version": value, "details": details}
    data = _encode(receipt, MAX_RECEIPT_BYTES)
    _publish(_internal(root, version.path).parent / "receipt.json", data)
    return replace(version, receipt_sha256=hashlib.sha256(data).hexdigest())


def create_job(source: Path, directory: Path, *, intent: Intent = "unknown",
               wanted_vocals_may_include_rap: bool = False,
               policy: ResourcePolicy = ResourcePolicy()) -> Job:
    """Exclusively import original bytes and convert using media.extract; no listening approval.

    Failed imports keep diagnostics but have no committed state and cannot load.
    Existing parent directory is required. No GPU, models or dependency installation.
    """
    require(intent in ("music", "spoken_audio", "unknown") and type(wanted_vocals_may_include_rap) is bool,
            "Invalid recording intent.")
    source = _plain(Path(source))
    require(0 < source.stat().st_size <= MAX_MEDIA_BYTES, "Source exceeds 1 GiB limit.")
    suffix = source.suffix.lower()
    require(re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) is not None, "Source needs a safe suffix.")
    root = Path(os.path.abspath(directory))
    _plain(root.parent, directory=True)
    require(not root.exists() and not root.is_symlink(), "Job destination must be fresh.")
    source_hash = digest(source)
    probe = json.loads(execute(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                               "stream=codec_name,sample_rate,channels:format=duration,format_name", "-of", "json", str(source)], 30))
    require(type(probe.get("streams")) is list and len(probe["streams"]) == 1, "No supported audio stream.")
    stream, format_info = probe["streams"][0], probe["format"]
    require(digest(source) == source_hash, "Source changed during probe.")
    original = Source(f"source/original{suffix}", source_hash, source.stat().st_size,
                      float(format_info["duration"]), int(stream["sample_rate"]), int(stream["channels"]),
                      stream["codec_name"], format_info["format_name"])
    identifier, version_id = new_id(), new_id()
    # Validate all supplied policy/source fields before copying or decoding.
    placeholder = Version(version_id, f"versions/{version_id}/audio.wav", "0" * 64, 1, "FLOAT", "canonical",
                          technical="passed", receipt_sha256="0" * 64)
    tools = {name: execute([name, "-version"], 10).splitlines()[0] for name in ("ffmpeg", "ffprobe")}
    initial = Job(identifier, original, intent, wanted_vocals_may_include_rap, policy,
                  versions=(placeholder,), current_version=version_id)
    validate_job(initial)
    required = original.bytes + MAX_FRAMES * 8 + MAX_METADATA_BYTES
    require(shutil.disk_usage(root.parent).free >= required and policy.max_output_bytes >= required, "Insufficient import space/budget.")
    root.mkdir(exist_ok=False)
    for name in ("state", "source", "versions", "runs"):
        (root / name).mkdir()
    _test_publication(root / "state")
    _publish(root / "incomplete.json", _encode({"schema_version": 1, "operation": "import", "job_id": identifier}, MAX_RECEIPT_BYTES))
    original_path = root / original.path
    _copy(source, original_path)
    require(digest(original_path) == original.sha256 and digest(source) == original.sha256, "Source changed after probe.")
    output = root / placeholder.path
    output.parent.mkdir()
    extract(original_path, output)
    _plain(output)
    info = audio_info(output, subtype="FLOAT")
    _finite_audio(output)
    with output.open("r+b") as file:
        os.fsync(file.fileno())
    version = replace(placeholder, sha256=digest(output), frames=info.frames)
    version = _version_receipt(root, identifier, original.sha256, version,
                               {"operation": "canonical_conversion", "implementation": "songtool.media.extract",
                                "sample_rate": RATE, "channels": 2, "encoding": "pcm_f32le", "tools": tools,
                                "lossless_conversion_claimed": False, "listening_approved": False})
    initial = replace(initial, versions=(version,), revision=1)
    validate_job(initial)
    _verify_media(root, initial)
    data = _encode(_data(initial), MAX_METADATA_BYTES)
    _publish(root / "state" / "00000001.json", data)
    return replace(initial, revision_sha256=hashlib.sha256(data).hexdigest())


def register_version(directory: Path, audio: Path, *, parent_id: str, parent_start_frame: int,
                     role: str = "candidate", technical: Technical = "not_run", run_id: str | None = None,
                     verification: dict | None = None, expected_revision: int,
                     expected_revision_sha256: str) -> Job:
    """Copy canonical WAV and publish receipt before state; never change current/listening.

    Caller explicitly attests a length-preserving or simple-trim timeline via
    parent_start_frame (relative to parent). No cross-file alignment is inferred.
    A passed candidate requires a bounded verification receipt supplied by the
    future workflow validator; this API does not perform perceptual verification.
    On conflict, orphan media/receipt remain diagnostic, not registered success.
    """
    job = load_job(directory)
    if (job.revision, job.revision_sha256) != (expected_revision, expected_revision_sha256):
        raise RevisionConflict("Job changed; reload before registration.")
    require(len(job.versions) < MAX_RECORDS, "Version limit reached; history cannot be pruned.")
    _id(parent_id)
    parent = next((v for v in job.versions if v.id == parent_id), None)
    require(parent is not None, "Unknown parent.")
    _integer(parent_start_frame, 0, parent.frames - 1, "parent offset")
    _text(role, "role")
    require(role != "canonical", "Only import creates canonical root.")
    require(technical in ("not_run", "passed", "failed"), "Invalid technical status.")
    require(run_id is None or any(r.id == run_id for r in job.runs), "Unknown producing run.")
    require(verification is None or type(verification) is dict, "Verification must be structured metadata.")
    require(technical != "passed" or bool(verification), "Technical pass needs explicit verification evidence.")
    details = {"operation": "register", "parent_start_frame": parent_start_frame,
               "timeline": "explicit_translation", "verification": verification, "listening_approved": False}
    _encode(details, MAX_RECEIPT_BYTES // 2)
    audio = _plain(Path(audio))
    require(audio.stat().st_size <= MAX_MEDIA_BYTES, "Oversized candidate.")
    info = audio_info(audio)
    _finite_audio(audio)
    require(parent_start_frame + info.frames <= parent.frames, "Candidate exceeds mapped parent interval.")
    root = _plain(Path(directory), directory=True)
    require(shutil.disk_usage(root).free >= audio.stat().st_size + MAX_METADATA_BYTES
            and audio.stat().st_size <= job.policy.max_output_bytes, "Insufficient candidate space/budget.")
    identifier = new_id()
    version = Version(identifier, f"versions/{identifier}/audio.wav", digest(audio), info.frames, info.subtype, role,
                      parent_id, parent.sha256, parent.source_start_frame + parent_start_frame, run_id, technical)
    path = root / version.path
    _plain(path.parent.parent, directory=True)
    path.parent.mkdir(exist_ok=False)
    _copy(audio, path)
    require(digest(path) == version.sha256, "Candidate changed during registration.")
    version = _version_receipt(root, job.id, job.source.sha256, version, details)
    candidate = replace(job, versions=(*job.versions, version),
                        latest_technically_passed_candidate=version.id if technical == "passed"
                        else job.latest_technically_passed_candidate)
    return commit_revision(root, candidate)


def select_version(directory: Path, version_id: str, *, verdict: str, note: str) -> Job:
    """Record a whole-version listening verdict; only verified better promotes."""
    require(verdict in ("better", "worse", "unconfirmed"), "Invalid selection verdict.")
    job = load_job(directory)
    version = next((v for v in job.versions if v.id == version_id), None)
    require(version is not None, "Unknown selection version.")
    require(verdict != "better" or version.technical == "passed", "Cannot promote an unverified/failed candidate.")
    listening = "uncertain" if verdict == "unconfirmed" else verdict
    # A negative verdict records actual listening status without selecting another file.
    updated = replace(version, listening=listening)
    item = Feedback(new_id(), version.id, version.sha256,
                    "good" if verdict == "better" else "uncertain",
                    f"{verdict}: {note}", "whole")
    _text(note, "listening note", 8000)
    return commit_revision(directory, replace(job,
        versions=tuple(updated if v.id == version.id else v for v in job.versions),
        feedback=(*job.feedback, item),
        current_version=version.id if verdict == "better" else job.current_version))


def recover_run(directory: Path, run_id: str) -> Job:
    """Reconcile an immutable terminal receipt or mark a dead owner interrupted.

    No outputs are deleted or processing restarted. Missing legacy controller
    identity fails closed when no child identity establishes ownership.
    """
    from . import runtime
    root = _plain(Path(directory), directory=True)
    job = load_job(root)
    _id(run_id)
    run = next((r for r in job.runs if r.id == run_id), None)
    require(run is not None, "Unknown run.")
    if run.execution != "running":
        return job
    run_dir = _plain(root / "runs" / run.id, directory=True)
    intent, _ = _read(run_dir / "intent.json", MAX_RECEIPT_BYTES)
    require(intent.get("fingerprint") == run.fingerprint and intent.get("parent_sha256") == run.parent_sha256,
            "Recovery intent mismatch.")
    controller = intent.get("controller")
    if controller is not None:
        require(type(controller) is dict and set(controller) == {"pid", "identity"}, "Invalid controller identity.")
        _integer(controller["pid"], 1, 2**32-1, "controller PID")
        _text(controller["identity"], "controller identity")
        require(not runtime.process_matches(controller["pid"], controller["identity"]), "Run controller is still active.")
    else:
        require(run.process_id is not None, "No recorded ownership identity; recovery fails closed.")
    if run.process_id is not None:
        require(not runtime.process_matches(run.process_id, run.process_identity), "Run child is still active.")
    # Child identity can have been published before its state callback committed.
    identity_path = run_dir / "worker" / "process.json"
    if identity_path.exists():
        identity, _ = _read(identity_path, MAX_RECEIPT_BYTES)
        require(not runtime.process_matches(identity["pid"], identity["identity"]), "Uncommitted child is active.")
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.exists():
        terminal = replace(run, execution="interrupted", technical="not_run", receipt_sha256=None)
        receipt = {"schema_version": 1, "job_id": job.id, "source_sha256": job.source.sha256,
                   "fingerprint": run.fingerprint, "run": asdict(terminal), "version": None,
                   "execution": "interrupted", "technical": "not_run", "listening": "unreviewed",
                   "failure_kind": "operational", "outcome": "interrupted", "rendered_now": False}
        _publish(receipt_path, _encode(receipt, MAX_RECEIPT_BYTES))
    receipt, sha = _read(receipt_path, MAX_RECEIPT_BYTES)
    require(receipt.get("schema_version") == 1 and receipt.get("job_id") == job.id
            and receipt.get("source_sha256") == job.source.sha256 and receipt.get("fingerprint") == run.fingerprint,
            "Foreign recovery receipt.")
    terminal = _record(Run, receipt.get("run"))
    require(terminal.execution != "running" and terminal.receipt_sha256 is None
            and replace(terminal, execution=run.execution, technical=run.technical,
                        receipt_sha256=run.receipt_sha256) == run,
            "Terminal receipt does not match committed run identity.")
    require(receipt.get("execution") == terminal.execution and receipt.get("technical") == terminal.technical,
            "Inconsistent terminal receipt.")
    terminal = replace(terminal, receipt_sha256=sha)
    candidate = _record(Version, receipt["version"]) if receipt.get("version") is not None else None
    if terminal.execution == "completed":
        verification = receipt.get("verification") or {}
        require(terminal.technical == "passed" and verification.get("technical") == "passed",
                "Completed receipt lacks technical verification.")
        require((candidate is not None) == (run.operation != "scan"), "Completed receipt missing/unexpected candidate.")
        if run.operation == "scan":
            _, scan_sha = _read(run_dir / "worker" / "scan.json", MAX_RECEIPT_BYTES)
            require(scan_sha == verification.get("analysis_sha256"), "Recovered scan hash mismatch.")
    if candidate:
        require(terminal.execution == "completed" and terminal.technical == candidate.technical == "passed"
                and candidate.run_id == run.id and candidate.parent_id == run.parent_id
                and candidate.parent_sha256 == run.parent_sha256 and candidate.listening == "unreviewed",
                "Invalid recovered candidate.")
        verification = receipt.get("verification") or {}
        require(verification.get("technical") == "passed" and verification.get("candidate_sha256") == candidate.sha256,
                "Recovered candidate lacks verification.")
    result = replace(job, runs=tuple(terminal if r.id == run.id else r for r in job.runs),
                     versions=(*job.versions, candidate) if candidate else job.versions,
                     latest_technically_passed_candidate=candidate.id if candidate else job.latest_technically_passed_candidate)
    return commit_revision(root, result)


def copy_job(directory: Path, destination: Path) -> Job:
    """Verified bounded snapshot, exclusive destination and report-last completion.

    Includes diagnostic outputs, never follows links or copies publication staging
    aliases. Active jobs must first finish/recover. A failed copy stays incomplete.
    """
    root = _plain(Path(directory), directory=True)
    job = load_job(root)
    require(not any(r.execution == "running" for r in job.runs), "Finish/recover active runs before copying.")
    target = Path(os.path.abspath(destination))
    _plain(target.parent, directory=True)
    require(not target.exists() and not target.is_symlink(), "Copy destination must be fresh.")
    require(not target.is_relative_to(root) and not root.is_relative_to(target), "Overlapping copy paths.")
    for run in job.runs:
        _, sha = _read(root / "runs" / run.id / "receipt.json", MAX_RECEIPT_BYTES)
        require(sha == run.receipt_sha256, "Committed run receipt hash mismatch.")
    files, total, directory_count = [], 0, 0
    def traversal_error(error):
        raise error
    for base, directories, names in os.walk(root, followlinks=False, onerror=traversal_error):
        directory_count += 1
        require(directory_count <= 20000 and len(directories) + len(names) <= 20000,
                "Copy exceeds directory/entry bound.")
        _plain(Path(base), directory=True)
        for name in directories:
            _plain(Path(base) / name, directory=True)
        for name in names:
            if name.startswith((".pending-", ".probe-")) or (Path(base) == root and name in ("copy-intent.json", "copy-complete.json")):
                continue
            path = Path(base) / name
            if path.suffix == ".json":
                _, sha = _read(path, MAX_METADATA_BYTES)
            else:
                _plain(path)
                require(path.stat().st_size <= MAX_MEDIA_BYTES, "Oversized copy file.")
                sha = digest(path)
            size = path.stat().st_size
            total += size
            files.append((path.relative_to(root), size, sha))
            require(len(files) <= 20000 and total <= 64 * 1024**3, "Copy exceeds file/64 GiB bound.")
    require(shutil.disk_usage(target.parent).free >= total + MAX_METADATA_BYTES, "Insufficient copy capacity.")
    target.mkdir(exist_ok=False)
    _test_publication(target)
    marker = {"schema_version": 1, "job_id": job.id, "revision": job.revision,
              "revision_sha256": job.revision_sha256, "files": len(files), "bytes": total}
    _publish(target / "copy-intent.json", _encode(marker, MAX_RECEIPT_BYTES))
    for relative, size, sha in files:
        src, dst = root / relative, target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as source, dst.open("xb") as output:
            remaining = size
            while remaining:
                block = source.read(min(1024**2, remaining))
                require(bool(block), "Copy source truncated.")
                output.write(block)
                remaining -= len(block)
            require(not source.read(1), "Copy source grew.")
            output.flush()
            os.fsync(output.fileno())
        require(digest(dst) == sha == digest(src), "Copy hash mismatch.")
    for name in ("runs", "state", "source", "versions"):
        (target / name).mkdir(exist_ok=True)
    require(load_job(root).revision_sha256 == job.revision_sha256, "Job changed during copy.")
    _verify_media(target, job)
    _publish(target / "copy-complete.json", _encode(marker, MAX_RECEIPT_BYTES))
    return load_job(target)
