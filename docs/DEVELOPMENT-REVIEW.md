# Development review operator guide

Open the local review surface at [http://127.0.0.1:8787/review](http://127.0.0.1:8787/review).
It is a development-only review tool for the existing 16 opaque clips. Those
clips were already used in development work; they are not fresh held-out data
and are not a Gate D panel.

## Start a session

Choose the available review set, enter your own reviewer ID, then choose one
kind:

- `human_self_reported` means the person entering the ID self-reports as human.
  The service does not verify identity, independence, or qualification.
- `model_assisted` records a model-assisted draft. It is never human evidence.

The UI does not invent reviewer IDs, assignments, ratings, or a registry. If
there is no set or no session, it shows no rating form.

The review media and packet IDs are opaque. The page exposes task/rubric and
review guidance, but not source clip IDs, source lineages, cohort labels, model
votes, checkpoints, or hashes.

## Drafts, completion, and resuming

Use **Save draft** whenever a partial judgement is useful. Drafts are private
and may contain only the fields you actually enter. A worksheet is displayed as
complete only after integrity, collision, progress, completion evidence,
nonempty evidence-frame indices, and a nonempty observable reason are present;
the progress/completion consistency rules still apply.

Completion is a UI state, not a human-quality assertion, annotation export, or
Gate result.

Session metadata (session ID, set ID, entered reviewer ID, and kind) lives in
`sessionStorage`; the secret is an HttpOnly cookie and is not readable by page
JavaScript. A reload in the same tab resumes the saved session through the
cookie and deferred resume request. Closing the tab removes its sessionStorage
metadata, but does **not** delete server-side drafts. An expired or rejected
session clears local metadata and cannot automatically recover an identity;
start a new session explicitly.

Sessions are bound to the review set and use an 8-hour TTL. The browser cookie
is local, HttpOnly, SameSite=Strict, and scoped to `/api/development-review`.

## Data safeguards and recovery

Production draft storage is a private SQLite database outside the served
`data/live-integrated` tree (the default is the sibling
`data/live-integrated-development-review-private/development-review.sqlite3`).
It is created only after an explicit session POST. The private directory must
be mode 0700; database, WAL, and SHM files are mode 0600. Symlink components,
served-root paths, and unsafe origins are rejected. Write requests require the
same loopback origin, scheme, host, and port as the local request.

The production review database is currently absent. Do not fabricate ratings to
test it. When real reviews begin, preserve the database and its WAL/SHM files:
do not delete, reset, move into the served artifact tree, or replace them to
recover from an expired browser session. Investigate access or storage issues
without discarding existing drafts.

There is deliberately no formal annotation export, calibration importer, or
Gate bridge from this tool. Development reviews remain isolated until a future,
separately approved protocol defines an appropriate use.
