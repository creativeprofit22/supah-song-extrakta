"""Explicit, single-attempt audio operations; no automatic listening promotion.

Run receipts are published last, before the terminal state revision. A receipt
contains the complete terminal Run and optional Version for later recovery; a
partial worker directory is never success. Import remains jobs.create_job.
Only the static child entry below is dispatched, never filters or plugins.
"""
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version as package_version
import shutil
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from . import cleanup as c, jobs, runtime

OPERATIONS = {"trim": 1, "scan": 1, "bandit": 1, "gentle-denoise": 1, "local-eq": 1}
# Evidence identity only: deliberately NOT a worker operation or generic-denoise alias.
LEGACY_OPERATION = "legacy-trailer-fixed"


def _publish(path, value, limit=jobs.MAX_RECEIPT_BYTES):
    data = jobs._encode(value, limit)
    jobs._publish(path, data)
    return hashlib.sha256(data).hexdigest()


def _parent(job, identifier):
    identifier = job.current_version if identifier == "current" else identifier
    result = next((v for v in job.versions if v.id == identifier), None)
    c.require(result is not None, "Unknown parent version.")
    return result


def _parameters(operation, parameters, parent):
    p = {} if parameters is None else dict(parameters)
    allowed = {"trim": {"start_frame", "end_frame"}, "scan": {"speech_version_id"},
               "bandit": {"full_song_offset"}, "gentle-denoise": set(),
               "local-eq": {"intervals", "parent_sha256"}}
    c.require(operation in OPERATIONS, "Unsupported static operation.")
    c.require(set(p) <= allowed[operation], "Unknown operation parameters; no custom filters.")
    if operation == "trim":
        jobs._integer(p.get("start_frame"), 0, parent.frames - 1, "trim start")
        jobs._integer(p.get("end_frame"), p["start_frame"] + 1, parent.frames, "trim end")
    if operation == "scan" and "speech_version_id" in p:
        jobs._id(p["speech_version_id"])
    if operation == "bandit":
        p.setdefault("full_song_offset", False)
        c.require(type(p["full_song_offset"]) is bool, "Invalid offset selection.")
    if operation == "local-eq":
        c.require(p.get("parent_sha256") == parent.sha256, "EQ needs an explicit parent hash/map.")
        intervals = p.get("intervals")
        c.require(type(intervals) is list and 0 < len(intervals) <= 256, "Explicit EQ intervals required.")
        end = 0
        for pair in intervals:
            c.require(type(pair) in (list, tuple) and len(pair) == 2, "Invalid EQ interval.")
            first, last = pair
            jobs._integer(first, end, parent.frames - 1, "EQ start")
            jobs._integer(last, first + 2 * c.FADE, parent.frames, "EQ end")
            end = last
        p["intervals"] = [list(pair) for pair in intervals]
    jobs._encode(p, jobs.MAX_RECEIPT_BYTES // 2)
    return p


def summarize_job(directory: Path, version_id: str = "current") -> dict:
    job = jobs.load_job(directory)
    parent = _parent(job, version_id)
    ancestors = {parent.id}
    cursor = parent
    while cursor.parent_id:
        ancestors.add(cursor.parent_id)
        cursor = _parent(job, cursor.parent_id)
    unsupported = [f.category for f in job.feedback if f.version_id in ancestors and
                   f.category in ("wanted_vocal_loss", "warbling_reverse_like_artifact")]
    legacy = []
    for run in job.runs:
        if run.parent_id in ancestors and run.receipt_sha256:
            receipt, sha = jobs._read(Path(directory) / "runs" / run.id / "receipt.json", jobs.MAX_RECEIPT_BYTES)
            c.require(sha == run.receipt_sha256, "Run receipt hash mismatch.")
            if "legacy_evidence" in receipt:
                legacy.append({"run_id": run.id, "parent_id": run.parent_id,
                               "operation": run.operation, "fingerprint": run.fingerprint,
                               "failure_reason": receipt["error"],
                               "attestation": receipt["legacy_evidence"]["attestation"]})
    latest = next((r for r in job.runs if r.id == job.latest_attempt), None)
    return {"job_id": job.id, "current_version": job.current_version,
            "selected_version": parent.id, "legacy_failed_evidence": legacy,
            "legacy_recommendation": "Exact historical treatment is barred; no generic recipe equivalence is implied."
                                     if legacy else None,
            "latest_attempt": asdict(latest) if latest else None,
            "latest_technically_passed_candidate": job.latest_technically_passed_candidate,
            "execution": latest.execution if latest else "not_run",
            "technical": latest.technical if latest else "not_run",
            "listening": parent.listening, "feedback": [asdict(f) for f in job.feedback],
            "repair_status": "no_supported_repair" if unsupported else "explicit_operation_required",
            "explanation": "Wanted vocal loss/reverse-like artifacts need new capability, not denoise."
                           if unsupported else "Numerical checks are not listening approval."}


def describe_run(directory: Path, operation: str, version_id: str = "current", *,
                 parameters: dict | None = None, device: str = "cpu") -> dict:
    job = jobs.load_job(directory)
    parent = _parent(job, version_id)
    if operation == LEGACY_OPERATION:
        p = dict(parameters or {})
        c.require(set(p) == {"recipe_sha256", "start_frame", "end_frame"}, "Explicit legacy recipe hash and frame scope required.")
        jobs._hash(p["recipe_sha256"])
        jobs._integer(p["start_frame"], 0, parent.frames - 1, "legacy start")
        jobs._integer(p["end_frame"], p["start_frame"] + 1, parent.frames, "legacy end")
        c.require(device == "cpu", "Legacy evidence does not request execution or CUDA.")
        details = {"schema_version": 1, "source_sha256": job.source.sha256,
                   "parent_id": parent.id, "parent_sha256": parent.sha256,
                   "source_start_frame": parent.source_start_frame, "operation": operation,
                   "operation_version": 1, "parameters": p, "evidence_only": True}
        return {**details, "fingerprint": hashlib.sha256(jobs._encode(details, jobs.MAX_RECEIPT_BYTES)).hexdigest(),
                "rendered_now": False, "listening_approved": False}
    p = _parameters(operation, parameters, parent)
    c.require(device in ("cpu", "cuda") and (device == "cpu" or operation == "bandit"),
              "Only explicitly selected Bandit may use CUDA.")
    protected = [[r.start_frame, r.end_frame] for r in jobs.mapped_protected_ranges(job, parent.id)]
    tools = {name: package_version(name) for name in ("numpy", "soundfile", "scipy")}
    tools.update(python=sys.version, ffmpeg=c.execute(["ffmpeg", "-version"], 10).splitlines()[0])
    tools["implementation_sha256"] = c.digest(Path(__file__))
    tools["cleanup_sha256"] = c.digest(Path(c.__file__))
    tools["runtime_sha256"] = c.digest(Path(runtime.__file__))
    from . import review
    tools["review_sha256"] = c.digest(Path(review.__file__))
    if operation == "scan" and p.get("speech_version_id"):
        tools["speech_reference_sha256"] = _parent(job, p["speech_version_id"]).sha256
    import soundfile as sf
    tools["libsndfile"] = sf.__libsndfile_version__
    model = None
    if operation == "bandit":
        from .resources import REVISION, WEIGHTS_SHA256
        model = {"revision": REVISION, "checkpoint_sha256": WEIGHTS_SHA256}
        tools["torch"] = package_version("torch")
        from . import separation
        tools["separation_sha256"] = c.digest(Path(separation.__file__))
    details = {"schema_version": 1, "source_sha256": job.source.sha256,
               "parent_id": parent.id, "parent_sha256": parent.sha256,
               "source_start_frame": parent.source_start_frame, "operation": operation,
               "operation_version": OPERATIONS[operation], "parameters": p,
               "protected_intervals": protected, "tools": tools, "model": model, "device": device}
    fingerprint = hashlib.sha256(jobs._encode(details, jobs.MAX_RECEIPT_BYTES)).hexdigest()
    return {**details, "fingerprint": fingerprint, "rendered_now": False,
            "listening_approved": False}


def adopt_failed_evidence(directory: Path, evidence_file: Path, *, expected_evidence_sha256: str,
                          parent_id: str, parent_sha256: str, source_sha256: str,
                          failure_reason: str, operation: str, parameters: dict,
                          evidence_mapping: dict, attestation: str,
                          expected_revision: int, expected_revision_sha256: str) -> dict:
    """Explicit local evidence attestation, not historical technical verification.

    Mapping values are nonempty lists of JSON object keys (never code/commands).
    Required keys: source_sha256, parent_sha256, failure_reason, parameters;
    supported operations additionally require treatment = exact describe_run output.
    Legacy parameters identify a separately hashed fixed recipe and parent-frame scope.
    Missing historical equivalence evidence fails closed; no inferred generic alias.
    """
    root = jobs._plain(Path(directory), directory=True)
    job = jobs.load_job(root)
    if (job.revision, job.revision_sha256) != (expected_revision, expected_revision_sha256):
        raise jobs.RevisionConflict("Job changed; reload before evidence adoption.")
    parent = _parent(job, parent_id)
    c.require(parent.id == parent_id and parent.sha256 == parent_sha256
              and job.source.sha256 == source_sha256, "Legacy evidence source/parent mapping mismatch.")
    jobs._hash(expected_evidence_sha256)
    jobs._text(failure_reason, "original failure wording", 8192)
    jobs._text(attestation, "legacy evidence attestation", 8192)
    selected = jobs._plain(Path(evidence_file))
    evidence, sha = jobs._read(selected, jobs.MAX_RECEIPT_BYTES)
    c.require(sha == expected_evidence_sha256, "Legacy evidence hash mismatch.")
    intent = describe_run(root, operation, parent.id, parameters=parameters)
    expected = {"source_sha256": source_sha256, "parent_sha256": parent_sha256,
                "failure_reason": failure_reason, "parameters": intent["parameters"]}
    if operation != LEGACY_OPERATION:
        expected["treatment"] = intent
    c.require(type(evidence_mapping) is dict and set(evidence_mapping) == set(expected),
              "Explicit evidence mapping required; unsupported recipe equivalence cannot be assumed.")
    for key, value in expected.items():
        path = evidence_mapping[key]
        c.require(type(path) is list and 0 < len(path) <= 16, "Invalid evidence key path.")
        found = evidence
        for component in path:
            jobs._text(component, "evidence key")
            c.require(type(found) is dict and component in found, "Missing evidence mapping field.")
            found = found[component]
        c.require(jobs._encode({"value": found}, jobs.MAX_RECEIPT_BYTES) ==
                  jobs._encode({"value": value}, jobs.MAX_RECEIPT_BYTES),
                  f"Legacy evidence mapping mismatch: {key}.")
    c.require(not any(r.execution == "running" for r in job.runs), "Finish/recover active run before adoption.")
    c.require(len(job.runs) < jobs.MAX_RECORDS, "Run history limit reached.")
    # Duplicate adoption does not rename or weaken an existing terminal failure.
    for previous in job.runs:
        if previous.fingerprint == intent["fingerprint"]:
            c.require(False, "Treatment already recorded; inspect existing receipt, do not re-adopt.")
    run = jobs.Run(jobs.new_id(), operation, intent["fingerprint"], parent.id, parent.sha256,
                   execution="failed", technical="not_run")
    run_dir = root / "runs" / run.id
    legacy = {"path": f"runs/{run.id}/legacy-evidence.json", "sha256": sha,
              "mapping": evidence_mapping, "attestation": attestation,
              "semantics": "User-attested historical failure; measurements and execution unverified.",
              "operation_supported": operation in OPERATIONS}
    receipt = {"schema_version": 1, "job_id": job.id, "source_sha256": job.source.sha256,
               "run": asdict(run), "version": None, "fingerprint": run.fingerprint,
               "treatment": intent, "legacy_evidence": legacy, "failure_kind": "legacy_evidence",
               "error": failure_reason, "verification": None, "runtime": None,
               "execution": "failed", "technical": "not_run", "listening": "unreviewed",
               "outcome": "failed", "rendered_now": False, "listening_approved": False}
    jobs._encode(receipt, jobs.MAX_RECEIPT_BYTES)
    run_dir.mkdir(exist_ok=False)
    jobs._copy(selected, root / legacy["path"])
    c.require(c.digest(root / legacy["path"]) == sha, "Evidence changed during adoption.")
    run = replace(run, receipt_sha256=_publish(run_dir / "receipt.json", receipt))
    # One terminal commit: a crash before this leaves only unregistered diagnostics.
    # latest_attempt denotes actual job execution, not retrospective evidence import.
    jobs.commit_revision(root, replace(job, runs=(*job.runs, run)))
    return receipt


def _preserve(original, processed, intervals):
    """Raised-cosine transitions entirely outside accepted half-open intervals."""
    import numpy as np
    merged = []
    for first, last in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(last, merged[-1][1])
        else:
            merged.append([first, last])
    weight = np.ones(len(original))
    for index, (first, last) in enumerate(merged):
        left = merged[index - 1][1] if index else 0
        right = merged[index + 1][0] if index + 1 < len(merged) else len(original)
        c.require((first == 0 or first - left >= c.FADE) and
                  (last == len(original) or right - last >= c.FADE),
                  "No legal outside-only protection transition area.")
        weight[first:last] = 0
        fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, c.FADE))
        if first:
            weight[first-c.FADE:first] = np.minimum(weight[first-c.FADE:first], fade[::-1])
        if last < len(original):
            weight[last:last+c.FADE] = np.minimum(weight[last:last+c.FADE], fade)
    result = original + (processed - original) * weight[:, None]
    for first, last in merged:
        result[first:last] = original[first:last]
    return result, merged


def validate_result(parent: Path, candidate: Path, *, operation: str,
                    parameters: dict | None = None, protected_intervals=()) -> dict:
    """Decoded DOUBLE guards. Generic cleanup limits match the legacy verifier.

    Trim is exact translated copying; Bandit is not expected to equal its mix.
    Both still require finite, peak-safe output and exact accepted samples.
    """
    import numpy as np
    import soundfile as sf
    c.require(operation in OPERATIONS and operation != "scan", "Not a rendering operation.")
    parent_frames = c.audio_info(parent).frames
    parent_sha256 = c.digest(parent)
    p = _parameters(operation, parameters, SimpleNamespace(frames=parent_frames, sha256=parent_sha256))
    protected_intervals = [list(pair) for pair in protected_intervals]
    start = p["start_frame"] if operation == "trim" else 0
    end = p["end_frame"] if operation == "trim" else parent_frames
    jobs._integer(start, 0, parent_frames - 1, "result parent start")
    jobs._integer(end, start + 1, parent_frames, "result parent end")
    for first, last in protected_intervals:
        jobs._integer(first, 0, parent_frames - 1, "protected start")
        jobs._integer(last, first + 1, parent_frames, "protected end")
    c.audio_info(candidate, end - start, "DOUBLE")
    a = sf.read(parent, start=start, stop=end, dtype="float64", always_2d=True)[0]
    b = sf.read(candidate, dtype="float64", always_2d=True)[0]
    c.finite_audio(a)
    c.finite_audio(b)
    failures = []
    def check(name, passed):
        if not passed:
            failures.append(name)
    for first, last in protected_intervals:
        first, last = max(first, start), min(last, end)
        if first < last:
            check("protected_samples", np.array_equal(a[first-start:last-start], b[first-start:last-start]))
    check("sample_peak", float(np.max(np.abs(b))) < 1)
    loud = c.loudness(candidate, "ffmpeg", c.rms(b) == 0)
    check("true_peak", loud["true_peak_dbtp"] is None or loud["true_peak_dbtp"] <= -1)
    if operation == "trim":
        check("exact_trim", np.array_equal(a, b))
    if operation in ("gentle-denoise", "local-eq"):
        before, after = c.analyze_windows(parent), c.analyze_windows(candidate)
        baseline_loud = c.loudness(parent, "ffmpeg", c.rms(a) == 0)
        check("baseline_sample_peak", float(np.max(np.abs(a))) < 1)
        check("baseline_true_peak", baseline_loud["true_peak_dbtp"] is None or baseline_loud["true_peak_dbtp"] <= -1)
        x, y = c.overall(before), c.overall(after)
        delta = c.db(y / x) if x and y else None
        check("overall_rms", abs(delta) <= 0.5 if delta is not None else x == y == 0)
        for x, y in zip(before, after, strict=True):
            change = c.db(y["rms"] / x["rms"]) if x["rms"] else None
            check(f"half_second_{x['start_frame']}",
                  change is not None and abs(change) <= 1 if x["state"] == "active" else y["rms"] <= 0.001)
        for first in range(0, len(a), 5 * jobs.RATE):
            x, y = a[first:first+5*jobs.RATE], b[first:first+5*jobs.RATE]
            level, difference = c.rms(x), c.rms(y-x)
            relative = c.db(difference / level) if level else None
            check(f"difference_{first}", relative is None or relative < -20 if level > 0.001 else c.rms(y) <= 0.001)
        for row in c.correlations(parent, candidate):
            if row["state"] == "measured":
                check("alignment", row["lag"] == 0 and row["correlation"] >= 0.99)
    # Guard every protection transition, including ones outside sampled alignment windows.
    for first, last in protected_intervals:
        for lo, hi in ((max(0, first-c.FADE), first), (last, min(end, last+c.FADE))):
            lo, hi = max(lo, start)-start, min(hi, end)-start
            if lo < hi:
                x, y = a[lo:hi], b[lo:hi]
                level, difference = c.rms(x), c.rms(y-x)
                relative = c.db(difference/level) if level else None
                check("protection_transition", (relative is None or relative < -20) if level > 0.001 else c.rms(y) <= 0.001)
    return {"technical": "failed" if failures else "passed", "failures": failures,
            "encoding": "DOUBLE", "decoded_protection_exact": "protected_samples" not in failures,
            "candidate_sha256": c.digest(candidate), "frames": len(b), "loudness": loud,
            "parent_sha256": parent_sha256, "parent_frames": parent_frames,
            "parent_start_frame": start, "operation": operation, "parameters": p,
            "protected_intervals": protected_intervals, "listening_approved": False}


def _worker(directory, run_id):
    """Private module entry: immutable intent selects only application-owned code."""
    import numpy as np
    import soundfile as sf
    from .review import _write_audio
    root = jobs._plain(Path(directory), directory=True)
    jobs._id(run_id)
    run_dir = root / "runs" / run_id
    intent, _ = jobs._read(run_dir / "intent.json", jobs.MAX_RECEIPT_BYTES)
    job = jobs.load_job(root)
    run = next(r for r in job.runs if r.id == run_id)
    c.require(run.execution == "running" and run.fingerprint == intent["fingerprint"], "Stale worker intent.")
    parent = _parent(job, run.parent_id)
    op, p = intent["operation"], _parameters(intent["operation"], intent["parameters"], parent)
    source, output = root / parent.path, run_dir / "worker"
    try:
        expected = describe_run(root, run.operation, run.parent_id, parameters=p, device=run.device)
        c.require({k: v for k, v in intent.items() if k not in ("retry_reason", "controller")} == expected
                  and expected["fingerprint"] == run.fingerprint,
                  "Worker intent/protection/tools no longer match the committed fingerprint.")
        if op == "scan":
            from .review import scan_version
            report = scan_version(root, parent.id, speech_version_id=p.get("speech_version_id"), _output=output)
            verification = {"technical": "passed", "analysis_sha256": c.digest(report)}
        else:
            original = sf.read(source, dtype="float64", always_2d=True)[0]
            c.finite_audio(original)
            if op == "trim":
                processed = original[p["start_frame"]:p["end_frame"]]
            elif op == "bandit":
                from . import separation
                import torch
                model = separation._load_model(run.device)
                torch.set_num_threads(job.policy.cpu_threads)
                runtime.mark_inference_started()
                processed = separation._infer_grid(original.astype("float32"), model, run.device,
                                                   jobs.RATE if p["full_song_offset"] else 0)[1].T.astype("float64")
            else:
                if op == "gentle-denoise":
                    c.verify_delay(output, "ffmpeg")
                chain = c.denoise_filter(parent.frames) if op == "gentle-denoise" else c.EQ
                c.filter_audio(source, output / "filtered.wav", chain, "ffmpeg")
                c.audio_info(output / "filtered.wav", parent.frames, "DOUBLE")
                processed = sf.read(output / "filtered.wav", dtype="float64", always_2d=True)[0]
                if op == "local-eq":
                    wet_energy, dry_energy = float(np.sum(processed**2)), float(np.sum(original**2))
                    gain = c.db((wet_energy/dry_energy)**0.5) if dry_energy else None
                    c.require(gain <= 1.5 if gain is not None else wet_energy == 0, "EQ energy guard failed.")
                    processed = original + (processed-original) * c.eq_mask(0, len(original), tuple(map(tuple, p["intervals"])))
            if op != "trim":
                processed, _ = _preserve(original, processed, intent["protected_intervals"])
            _write_audio(output / "candidate.wav", processed)
            verification = validate_result(source, output / "candidate.wav", operation=op,
                                          parameters=p, protected_intervals=intent["protected_intervals"])
        c.require(c.digest(source) == parent.sha256, "Parent changed during work.")
        _publish(output / "verification.json", verification)
    except ValueError as error:
        # Numerical, mapping and preservation guards are never retryable operational failures.
        _publish(output / "verification.json", {"technical": "failed", "failures": [str(error)[:2000]]})
        raise


def run_operation(directory: Path, operation: str, version_id: str = "current", *,
                  parameters: dict | None = None, device: str = "cpu", retry_reason: str | None = None,
                  timeout: int | None = None, workspace: Path | None = None, cancel=None) -> dict:
    root = jobs._plain(Path(directory), directory=True)
    intent = describe_run(root, operation, version_id, parameters=parameters, device=device)
    job = jobs.load_job(root)
    parent = _parent(job, intent["parent_id"])
    c.require(operation == LEGACY_OPERATION or intent["protected_intervals"] ==
              [[r.start_frame, r.end_frame] for r in jobs.mapped_protected_ranges(job, parent.id)],
              "Protection changed; describe again.")
    c.require(not any(r.execution == "running" for r in job.runs), "One active run per job; recover dead runs explicitly.")
    if operation == "gentle-denoise":
        ancestors = {parent.id}
        cursor = parent
        while cursor.parent_id:
            ancestors.add(cursor.parent_id)
            cursor = _parent(job, cursor.parent_id)
        c.require(not any(f.version_id in ancestors and f.category in
                         ("wanted_vocal_loss", "warbling_reverse_like_artifact") for f in job.feedback),
                  "no_supported_repair: denoise cannot restore wanted vocals or reverse-like artifacts.")
    if retry_reason is not None:
        jobs._text(retry_reason, "operational retry reason", 2000)
    reusable = None
    for previous in reversed(job.runs):
        if previous.fingerprint != intent["fingerprint"]:
            continue
        receipt, sha = jobs._read(root / "runs" / previous.id / "receipt.json", jobs.MAX_RECEIPT_BYTES)
        c.require(sha == previous.receipt_sha256, "Run receipt hash mismatch.")
        rejected = any(v.run_id == previous.id and v.listening == "worse" for v in job.versions)
        c.require(not rejected, "Identical listening-rejected run is blocked.")
        if previous.execution == "completed":
            reusable = receipt if reusable is None else reusable
        else:
            c.require(receipt.get("failure_kind") == "operational" and
                      (retry_reason is not None or reusable is not None),
                      "Identical failed attempt blocked; only operational failures permit a reasoned retry.")
    c.require(operation in OPERATIONS, "Unsupported legacy evidence-only operation; no renderer exists.")
    if reusable is not None:
        return {**reusable, "outcome": "reused_result", "rendered_now": False}
    c.require(len(job.runs) < jobs.MAX_RECORDS and len(job.versions) < jobs.MAX_RECORDS,
              "History limit reached before work.")
    policy = job.policy
    if timeout is not None:
        jobs._integer(timeout, 1, policy.wall_time_seconds, "timeout")
        policy = replace(policy, wall_time_seconds=timeout)
    run = jobs.Run(jobs.new_id(), operation, intent["fingerprint"], parent.id, parent.sha256, device=device)
    run_dir = root / "runs" / run.id
    run_dir.mkdir(exist_ok=False)
    controller = {"pid": os.getpid(), "identity": runtime.process_identity(os.getpid())}
    c.require(controller["identity"] is not None, "Controller identity unavailable.")
    _publish(run_dir / "intent.json", {**intent, "retry_reason": retry_reason, "controller": controller})
    _publish(run_dir / "incomplete.json", {"job_id": job.id, "run_id": run.id})
    job = jobs.commit_revision(root, replace(job, runs=(*job.runs, run), latest_attempt=run.id))
    def started(pid, identity):
        nonlocal job, run
        run = replace(run, process_id=pid, process_identity=identity)
        job = jobs.commit_revision(root, replace(job, runs=tuple(run if r.id == run.id else r for r in job.runs)))
    accounting, error, verification, candidate = None, None, None, None
    try:
        accounting = runtime.run_owned([sys.executable, "-B", "-m", "songtool.workflow", str(root), run.id],
                                      run_dir / "worker", workspace=workspace or root.parent,
                                      policy=policy, device=device, cancel=cancel, on_started=started)
        if (run_dir / "worker" / "verification.json").exists():
            verification, _ = jobs._read(run_dir / "worker" / "verification.json", jobs.MAX_RECEIPT_BYTES)
        if accounting.status != "completed":
            error = accounting.status
        elif not verification:
            error = "Missing technical verification"
    except BaseException as exception:
        error = f"{type(exception).__name__}: {str(exception)[:2000]}"
    passed = error is None and verification is not None and verification["technical"] == "passed"
    failure_kind = None if passed else "guard" if verification and verification["technical"] == "failed" else "operational"
    # Reload for concurrent feedback; intent/protection changes cannot be silently ignored.
    current = jobs.load_job(root)
    if current.feedback != job.feedback:
        passed, failure_kind, error = False, "guard", "Feedback changed during work; candidate requires a new intent."
    try:
        if passed and operation != "scan":
            audio = jobs._plain(run_dir / "worker" / "candidate.wav")
            c.require(c.digest(audio) == verification["candidate_sha256"], "Verified output changed.")
            c.require(shutil.disk_usage(root).free >= audio.stat().st_size + jobs.MAX_METADATA_BYTES
                      and runtime._bytes(run_dir) + audio.stat().st_size <= policy.max_output_bytes,
                      "Candidate publication exceeds output space/budget.")
            identifier = jobs.new_id()
            offset = intent["parameters"]["start_frame"] if operation == "trim" else 0
            candidate = jobs.Version(identifier, f"versions/{identifier}/audio.wav", c.digest(audio),
                                     verification["frames"], "DOUBLE", "candidate", parent.id, parent.sha256,
                                     parent.source_start_frame + offset, run.id, "passed")
            destination = root / candidate.path
            destination.parent.mkdir(exist_ok=False)
            jobs._copy(audio, destination)
            candidate = jobs._version_receipt(root, job.id, job.source.sha256, candidate,
                                              {"verification": verification, "fingerprint": run.fingerprint})
    except Exception as exception:
        passed, candidate = False, None
        failure_kind = "guard" if isinstance(exception, ValueError) else "operational"
        error = f"Candidate publication failed: {str(exception)[:2000]}"
    run = replace(run, execution="completed" if passed else "failed",
                  technical="passed" if passed else "failed" if failure_kind == "guard" else "not_run")
    if accounting is not None and operation != "bandit":
        accounting = replace(accounting, inference_ran=False)
    receipt = {"schema_version": 1, "job_id": job.id, "source_sha256": job.source.sha256,
               "run": asdict(run), "version": asdict(candidate) if candidate else None,
               "fingerprint": run.fingerprint, "retry_reason": retry_reason, "failure_kind": failure_kind,
               "error": error, "verification": verification, "runtime": asdict(accounting) if accounting else None,
               "output_bytes_before_receipt": runtime._bytes(run_dir) +
                   ((root / candidate.path).stat().st_size if candidate else 0),
               "execution": run.execution, "technical": run.technical, "listening": "unreviewed",
               "outcome": "analyzed_only" if passed and operation == "scan" else "rendered_new" if passed else "failed",
               "rendered_now": bool(passed and operation != "scan"), "listening_approved": False}
    sha = _publish(run_dir / "receipt.json", receipt)
    run = replace(run, receipt_sha256=sha)
    jobs.commit_revision(root, replace(current, runs=tuple(run if r.id == run.id else r for r in current.runs),
                         versions=(*current.versions, candidate) if candidate else current.versions,
                         latest_technically_passed_candidate=candidate.id if candidate else current.latest_technically_passed_candidate))
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Private static workflow worker requires job directory and run ID.")
    _worker(Path(sys.argv[1]), sys.argv[2])
