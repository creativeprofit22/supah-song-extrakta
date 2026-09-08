# Explicit local audio jobs

This is a private-local workflow for explicitly selected files, not an autonomous
optimizer, folder watcher, hostile-upload sandbox, or promise of better separation.
Use trusted local media and a local filesystem supporting exclusive hard links and
locking. Keep job directories private to cooperating users/processes; path/hash
checks are not a security boundary against an adversary changing files concurrently.
No installation, legacy adoption, GPU use, or current-song processing is automatic.

## Vocabulary and decisions

- **Source:** preserved original bytes plus hash and probe metadata.
- **Canonical:** separately converted stereo 48 kHz float WAV. Conversion is recorded;
  resampling/downmixing is not claimed lossless and conveys no listening approval.
- **Working baseline:** the exact version chosen as the next operation's parent.
  It may be raw/unmastered and need not be the preferred listening master.
- **Candidate:** a new render, never automatically selected. A trim has the selected
  interval's length; other rendering operations produce one whole-parent-length candidate.
- **Current/preferred:** `current_version` is a stored ID, initially the canonical
  import (unreviewed). Subsequently `choose --verdict better` selects a technically
  passed version after whole-version listening. No mutable `current.wav` exists.
- **Technical verification:** numerical/format/provenance checks, not better sound.
- **Listening approval:** human judgment with explicit scope. Accepted excerpts do
  not imply a whole-song verdict. Latest attempt, latest passed candidate, and current
  version are separate fields.

## Practical CLI

Run from the repository root with the existing Python 3.12 environment and FFmpeg /
ffprobe on PATH. Examples use `python`; on this Windows workstation substitute
`.venv/Scripts/python.exe`. `JOB` must be explicit, its parent directory must exist,
and create/copy destinations must be fresh. IDs below come from JSON/status, not filenames
or newest timestamps. Examples are alternatives, not instructions to run every treatment.

```text
python -B -m songtool job create SOURCE JOB --intent music --wanted-vocals-may-include-rap
python -B -m songtool job status JOB --json
python -B -m songtool job open JOB --version current
python -B -m songtool job scan JOB --version VERSION_ID
python -B -m songtool job scan JOB --version VERSION_ID --clips
python -B -m songtool job feedback JOB --clip CLIP_ID --version VERSION_ID --verdict good --accepted --note "Exact user wording"
python -B -m songtool job run JOB --operation gentle-denoise --version VERSION_ID --device cpu --timeout 600
python -B -m songtool job open JOB --version CANDIDATE_ID
python -B -m songtool job choose JOB --version CANDIDATE_ID --verdict better --note "Whole-version listening judgment"
python -B -m songtool job recover JOB --run RUN_ID
python -B -m songtool job copy JOB FRESH_DESTINATION
```

`--intent` is required: `music`, `spoken_audio`, or `unknown`; rap is an optional
intent flag, not a license to mix speech back into music. Unknown/spoken recordings
can be tracked and scanned; explicitly choosing Bandit is an intentional music-separation
request, not an inferred treatment. Quality for other recording types is not promised.

Canonical API summaries and CLI status (JSON or labelled text) expose the recorded
`intent`, `wanted_vocals_may_include_rap`, and explanatory `recommendations`.
Music guidance requires explicit operation selection, CPU by default, and explicit
CUDA selection for Bandit. Spoken/unknown guidance warns that music-specific separation
requires explicit selection and may remove wanted speech. When the rap flag is true,
status cautions to preserve wanted vocals and not use speech-stem reinsertion or
denoise as restoration of missing vocals; restoration needs new capability.
These are advisory messages, not operation bans or treatment selections. A false rap
flag is not evidence that speech is unwanted. Intent changes no samples, thresholds,
repair eligibility, or listening approval; `repair_status` remains feedback-based.
Status reads and verifies recorded history: it starts no operation worker or model.
Historical receipt verification may invoke FFmpeg for numerical checks, without
rendering a new candidate or changing listening approval.

All scan/run/open version selectors accept `current`; prefer a resolved ID after status.
`choose` requires an actual ID. `status` without `--json` prints labelled fields.
`open` verifies hashes and prints resolved path, ID, technical/listening status and
`opened_existing` before invoking the desktop player. It does not render anything.

| Operation | Options and meaning |
| --- | --- |
| `trim` | Required `--start-frame N --end-frame N`, relative to parent, half-open interval |
| `scan` | CPU analysis only; optional `--speech-version-id ID` for an explicitly aligned registered estimate |
| `bandit` | CPU default; explicit `--device cuda` opts into GPU for this run; optional `--full-song-offset` selects the one-second offset grid |
| `gentle-denoise` | Fixed gentle denoise only; no custom filter options or trailer EQ times |
| `local-eq` | Required `--parent-sha256 HASH` and repeatable `--eq-interval START END`; sorted nonoverlapping parent-frame ranges, each at least 9,600 frames; fixed +1.5 dB / 3.5 kHz / Q 0.7 EQ |

`job scan` is the convenient scan command; only it has `--clips`. Both scan and run
accept `--timeout SECONDS` and `--retry-reason TEXT`. Options belonging to a different
operation are rejected. Bandit needs the existing pinned model resources; job commands
do not authorize fetching new dependencies/models. Legacy top-level `separate` still
has its historical CUDA default: the **job** CPU policy does not retrofit legacy commands.
Legacy `cleanup-preview` remains a separately labelled, fixed trailer-only recipe.

## Optional review and exact protection

A whole-song request normally means **one authorized whole-song candidate once**, then
listening—not compulsory repeated clip reviews. Scans rank uncertain spectral/tonal
hints, not confirmed defects or exhaustive detection. Without an aligned speech reference,
`speech_similarity_hint` is unavailable, not proof of clean audio. Tonal-dip hints are
not user-reported muffling. Optional export provides at most eight five-second clips.

Copy both `id` and `version_id` from the same clip in the scan or exported clips report,
not displayed clip numbers. Pass them as `--clip CLIP_ID --version VERSION_ID`.
The optional feedback `--version` requires an exact registered ID in that job, not `current`.
Without it, feedback still works when the clip matches only one version; ambiguous
matches fail and require `--version ID`. Identical audio on the same map can share a
stable clip ID across versions. Feedback and acceptance apply only to the selected
version (protection can follow its descendants), never unrelated matching versions.
Existing clip IDs and immutable receipts do not change. Each clip identifies
its parent hash, `[start_frame, end_frame)` and source map. Time is integer 48 kHz frames:
source frame = version `source_start_frame` + local frame. Only length-preserving
translations and simple trims are supported; equal duration alone proves no alignment.
CLI feedback categories are `good`, `residual_dialogue`, `wanted_vocal_loss`, `muffling`,
`warbling_reverse_like_artifact`, `noise`, and `uncertain`. Preserve the user's wording
in `--note`. Only explicit `good --accepted` protects that interval; acceptance does
not promote the version. `choose` accepts `better`, `worse`, or `unconfirmed` (stored
as `uncertain`) and records a whole-version verdict. Failed/unverified candidates
cannot be promoted; diagnostic preferences can instead be scoped feedback via the API.

Protection follows validated ancestor maps, not unrelated equal-length files. New
candidates restore exact parent samples inside accepted intervals, with 100 ms
transitions outside. Impossible transition space stops the operation. Generic output
uses DOUBLE WAV and verifies decoded equality and transition guards; that is not the
legacy PCM24 encoding guard. No speech-stem reinsertion or automatic repair loop.

## Status-first agent rule and retries

Before acting, read status, resolve the exact current and intended parent IDs/hashes,
consult feedback and failed receipts, then state operation, device, and limits. Run
one approved operation. Report `opened_existing`, `analyzed_only`, `rendered_new`,
`reused_result`, or failure honestly, with execution, technical, and scoped listening
states separate. Failed work may leave partial diagnostics; it is not a candidate
success even if audio exists. New run receipts include `render_acknowledgement`: null
until a worker has completed its exclusive, flushed, decoded-equality-checked export
and published an immutable `worker/render-complete.json` (at most 4 KiB). The controller
verifies its job/run/fingerprint/parent identity and diagnostic WAV hash, DOUBLE encoding,
stereo 48 kHz format and expected frame count. File existence alone is not completion.
`rendered_now=true` means that run completed a diagnostic render, even when numerical
guards, concurrent feedback or candidate publication fail. Such failures retain their
failed outcome and no published candidate version; they never change preference or
confer listening approval. Missing/invalid acknowledgements cannot establish completion.
Analysis-only, pre-render failure, reuse and open report `rendered_now=false`.
Reuse retains the original acknowledgement as historical evidence, not new activity.
Recovery preserves a terminal receipt byte-for-byte; without one, it records verified
render activity from the interrupted run, not work performed by recovery itself.
The recovery CLI still reports no new render. Historical receipts without the
`render_acknowledgement` field retain the older promotion-dependent boolean semantics:
a false value cannot rule out a completed failed diagnostic. They are never rewritten.

Wanted-vocal loss and warbling/reverse-like feedback on the selected version or its
ancestors produce `no_supported_repair` only when their scope overlaps surviving
audio. Interval feedback is translated to source frames and intersected with the
selected version's half-open source interval, including through nested trims;
disjoint intervals and exact boundary contact do not block denoise. Whole-version
feedback covers the ancestor's entire interval and remains applicable to descendants.
Sibling/descendant feedback does not flow sideways or backwards, regardless of equal
lengths. Status and denoise preflight use the same scope rule without changing original
wording, scope, or history. Overlapping wanted-vocal loss/reverse-like artifacts still
need new capability, not denoising. Other hints likewise do not promise an available cure.

Fingerprints include source/parent hashes, mapping, operation/version/parameters,
protected intervals, device, pinned model/checkpoint and relevant tool/code versions.
Identical completed operations reuse receipts. Identical guard failures and
listening-rejected runs block retry; only operational failures allow an explicit
recorded retry reason. Never change parameters/names merely to evade failed history.
Thresholds remain unchanged: sample peak below full scale, true peak at most −1 dBTP;
generic cleanup overall RMS change at most 0.5 dB, active half-second change at most
1 dB, active five-second difference below −20 dB relative, quiet ceiling −60 dBFS,
plus alignment and protection checks. A retry cannot relax these or erase history.

## Resources, persistence, and restore

Defaults: two numerical CPU threads, one active operation per job, 1,800-second wall
time, 2 GiB output budget. `--timeout` can only lower that job ceiling. API creation
accepts `jobs.ResourcePolicy`; there are no CLI thread/budget flags. Sources/media
are bounded to 1 GiB each and ten minutes; canonical processing is stereo 48 kHz.
Import checks space for the original plus maximum canonical audio/metadata; owned
workers reserve output budget plus bounded logs (default 1 MiB, excess discarded).
Import validates policy and capacity before launching work. One CPU-owned worker
covers probing, copying, conversion and verification under a single remaining
wall-time budget; numerical thread limits apply before worker imports. Import
metadata capture and diagnostic logs are each bounded to 1 MiB. The job root's
`runtime.json` records execution accounting; only controller-observed success can
publish the initial state. `incomplete.json`, `import-ready.json` and partial media
alone never constitute a healthy job. Output usage is sampled, not a filesystem
quota; final publication also checks space used including the runtime receipt.
These are bounded application budgets, not hard OS RAM/thermal quotas.

Workers receive thread limits before numerical imports, with owned process identity,
timeout/cancellation and process-tree teardown. CUDA requires explicit selection;
the lease coordinates only jobs sharing a workspace (CLI: the job's parent directory).
It cannot control unrelated applications or other workspaces. Receipts record device,
elapsed time, child identity, output accounting and inference evidence; telemetry is
`unavailable` when absent, never a guessed temperature or a thermal safety guarantee.
No GPU power settings, precision, or overlap policy is changed.

A separate guardian owns the Windows Job Object/Linux process group and GPU lease
before the controller can release real work. A private liveness pipe triggers exact-tree
teardown after abrupt controller death, as well as normal timeout/cancellation. Linux
keeps the group leader unreaped until teardown and uses a subreaper for orphaned
children. The guardian retains ownership and the lease until descendants have stopped,
even if the kernel delays termination. No process-name killing is used.

Recovery checks controller, guardian and worker PID/creation identities. A started
guardian-owned tree also needs its stop acknowledgement before recovery can publish
interruption. Missing acknowledgement fails closed; do not remove ownership evidence
to free a job. Legacy Linux worker receipts also block recovery while their process
group remains active. These guarantees cover this tool's owned, non-detaching workers,
not independently daemonized external applications.

`source/`, `versions/`, `runs/` and `state/00000001.json` onward hold immutable media,
receipts and consecutive hash-linked snapshots. Publication is exclusive, report-last;
missing/corrupt state fails closed, without falling back to an older revision. Limits:
256 versions, 256 runs, 2,000 feedback records, 4,096 revisions, 1 MiB per state JSON,
64 KiB per receipt (registration details have a smaller bound). Limits refuse further
mutation, never silently prune. Before publishing a running intent, scan/render runs
reserve revisions for intent, optional child identity and terminal/recovery state,
plus worst-case snapshot bytes and a render candidate slot. Every commit retaining
an active run preserves that headroom; feedback and policy changes may be refused
before the absolute limits. Insufficient headroom refuses work before launch.
Existing snapshots remain readable without retroactively requiring this reservation.
Do not edit committed files or hard-linked staging files.

`recover` checks the recorded process identity before marking dead work interrupted,
or reconciles matching terminal receipts. It never reruns or deletes artifacts.
`copy` rejects overlapping/existing destinations and link traversal, checks capacity,
copies and verifies a complete snapshot, and publishes completion last. Committed
state, receipts, committed scan reports and import/copy completion contracts remain strict
JSON. Uncommitted diagnostics (including partial scan/verification reports) and unpublished
staging files are copied as bounded opaque bytes with source/destination hash verification,
not repaired or treated as success. JSON diagnostics and staging files retain the 1 MiB
ceiling. An opaque JSON report may retain exactly one same-directory `.pending-*` hard-link
alias; other diagnostic hard links are refused. Linked publication aliases are omitted,
not independent partial evidence. Incomplete copies cannot load as healthy
jobs. Restore drill: copy to a fresh directory, run
`job status FRESH_DESTINATION --json` (loads/verifies source and version hashes), and
compare IDs, hashes and feedback with the original. Use that fresh job directly; do
not overwrite state to roll back. Same-disk snapshots/copies are **not disaster backups**.
Off-device backup requires a user-selected destination; none is automatically configured.

## Optional legacy-trailer adoption: explicit validated API

**Do not perform adoption automatically.** No `job import-legacy`, receipt-import, or
scan-map-import CLI exists. Inventory and validation below require explicit approval
and a reviewed local Python procedure, not a new full-song render. Leave all existing
media, names, receipts and the older preferred master untouched.

1. Inventory the actual original, raw full-song separation, current trimmed raw working
   baseline, relevant separation/trim parents, older preferred master, scan manifests,
   clip maps, and failed-run receipts. Validate SHA-256 against original receipts using
   `songtool.cleanup.digest(Path(...))`; inspect content with `audio_info`/ffprobe and
   verify decoded finite audio and exact frame counts. Validate every trim offset and
   source map from evidence, including decoded equality where a trim claims copying.
   Do not invent a 59-second offset or infer alignment from durations. Preserve failed
   markers even when a report looks successful. Record missing evidence as missing.
2. With approval, `jobs.create_job(source, fresh_directory, intent="music",
   wanted_vocals_may_include_rap=True)` copies the chosen original bytes and makes a
   canonical WAV. This conversion is new media, not rerunning separation. Record which
   original was chosen (container versus decoded soundtrack), and validate its mapping.
3. Register existing canonical-format WAV parents in dependency order with
   `jobs.register_version(directory, audio, parent_id=..., parent_start_frame=...,
   role="legacy_parent", technical="not_run", verification={...},
   expected_revision=job.revision, expected_revision_sha256=job.revision_sha256)`.
   Use `Path` arguments; reload with `jobs.load_job(directory)` before each mutation,
   then use the returned job and new `job.versions[-1].id`. The API copies bytes and
   validates format/finite samples, bounds and hashes; caller must attest the exact
   translation. Register the trim as `role="legacy_working_baseline"` and older master
   as a distinct `role="legacy_preferred_master"`, using their real parents/offsets.
   Keep historical receipt hashes, paths, mapping rationale and bounded evidence in
   `verification`; this is evidence metadata, not automatic guard verification. Leave
   `technical="not_run"` for historical evidence alone. A new `technical="passed"`
   registration requires a `workflow.validate_result` report using the exact registered
   parent, candidate, supported operation/parameters and `jobs.mapped_protected_ranges`
   intervals. Registration independently reruns the fixed numerical checks; stale,
   contradictory or mismatched reports are rejected. An optional `run_id` must identify
   a passed, completed run whose immutable intent and receipt match that exact result.
   Loading rechecks passed receipts
   against their creation-time protection, not feedback added later. Older legitimate
   worker receipts use their immutable run intent and are not rewritten.
   A failed diagnostic WAV may be registered `technical="failed"`; this is **not** a
   failed workflow Run and does not block fingerprint retries.
4. Preserve scan manifests as hash-verified exclusive copies in a separately managed
   private evidence directory; retain its inventory/hashes in registration evidence.
   Arbitrary scan attachments still require separate verified backup. Failed receipts
   adopted through step 6 are copied inside the job and included in verified `job copy`.
   Do not fabricate stable clip IDs or manually insert native Run records.
   For verified legacy maps use `jobs.record_feedback(directory, version_id=...,
   category=..., note=original_wording, scope="interval", start_frame=...,
   end_frame=..., accepted=..., expected_revision=job.revision,
   expected_revision_sha256=job.revision_sha256)` after each reload. This interval API
   avoids rerunning a scan merely to get CLI IDs. Record source/clip mapping evidence
   alongside the original wording; stop if frame boundaries are unknown.
5. Preserve actual listening scope: **opening trim approved; clips 1 and 6 accepted;
   clip 8 accepted with minor reservation; clips 2–5 problematic; clip 7 lower-priority
   muffling; no whole-song approval**. Use accepted `good` intervals for 1/6/8, retaining
   8's reservation verbatim; classify problems only as supported by actual wording.
   Opening-trim approval is an edit decision, not acceptance of every sample in the
   retained song: record scoped explanatory feedback without whole-song protection.
   Do not call `select_version(..., verdict="better")` to invent approval. Registered
   raw trim remains an explicit parent ID, distinct from both canonical import and
   older preferred master; registration does not change `current_version`.
6. Use `workflow.adopt_failed_evidence` only after explicitly reviewing the selected
   historical JSON receipt (64 KiB maximum), its exact registered source/parent hashes,
   original failure wording and recipe identity. The public API below copies those
   original bytes exclusively into a generated run directory, publishes a native terminal
   failed receipt and commits an immutable Run. It records `technical="not_run"`, null
   verification/runtime and a user attestation: no prior measurements or execution are
   authenticated, and no audio is rendered or promoted. `latest_attempt` is unchanged.

   ```python
   from pathlib import Path
   from songtool import jobs, workflow

   job = jobs.load_job(directory)  # read status first; never infer a parent
   receipt = workflow.adopt_failed_evidence(
       directory, Path(selected_historical_receipt),
       expected_evidence_sha256=reviewed_receipt_sha256,
       parent_id=registered_parent_id, parent_sha256=reviewed_parent_sha256,
       source_sha256=reviewed_source_sha256,
       failure_reason=original_failure_wording,
       operation="legacy-trailer-fixed",
       parameters={"recipe_sha256": reviewed_fixed_recipe_sha256,
                   "start_frame": exact_start, "end_frame": exact_end},
       evidence_mapping={"source_sha256": ["source_sha256"],
                         "parent_sha256": ["parent_sha256"],
                         "failure_reason": ["failure_reason"],
                         "parameters": ["parameters"]},
       attestation=original_mapping_and_recipe_attestation,
       expected_revision=job.revision,
       expected_revision_sha256=job.revision_sha256)
   ```

   Variables above are explicitly reviewed values, not commands read from JSON. Each
   mapping value is a list of nested JSON object keys in the **original** receipt;
   adapt the paths to its real structure. Mapped values must equal the supplied hashes,
   wording and static parameters exactly. Recipe hash identifies the reviewed complete
   fixed recipe, not merely the generic denoise filter; frames are half-open parent
   48 kHz coordinates. Missing fields, mismatches or uncertain recipe identity must stop
   adoption: do not rewrite the historical receipt or synthesize evidence to pass validation.

   `legacy-trailer-fixed` is separately labelled, unsupported for rendering, and its
   exact source/parent/recipe/scope fingerprint is barred even with an operational retry
   reason. It never aliases or globally blacklists `gentle-denoise`. For a supported
   operation, static parameters are validated normally and an additional `"treatment"`
   mapping must match the complete `workflow.describe_run(...)` result, including tool,
   model and protection identity. Without that evidence, equivalence is unsupported.
   Historical trailer evidence must not be relabelled generic to circumvent this check.

   Reload with `jobs.load_job`, inspect `workflow.summarize_job(directory, parent_id)`
   for selected-version/ancestor legacy evidence, and perform `jobs.copy_job` to a fresh
   destination. Reload verifies the native receipt and original evidence hash; a changed
   or missing evidence file fails closed. No committed history is edited. A crash before
   the single state commit leaves unregistered diagnostics, not a successful adoption.
   Keep an unapproved baseline as an explicit parent ID; never fake current preference.
   Actual trailer adoption and the adequacy of its historical fields remain unverified
   until separately authorized; unsupported/missing evidence remains manually barred.

The repeated failed fixed denoise history must prevent an identical retry on this
recording, not blacklist denoise on unrelated jobs. No adoption step authorizes
renaming, removal, regeneration of legacy audio, promotion, or another full-song trial.

## Verification scope

`python -B scripts/verify_audio_jobs.py` is the portable synthetic-file harness; it
needs installed project dependencies and FFmpeg/ffprobe, not trailer fixtures, a model
download or GPU. CLI help is safe: `python -B -m songtool job run --help` and
`python -B -m songtool cleanup-preview --help`. `uv build` checks packaging. CI wiring
is maintained in `.github/workflows/ci.yml`; neither synthetic checks nor a green build
establish listening quality. The separate ignored cleanup harness requires original
private fixtures. Do not run a current-song trial as routine verification.
