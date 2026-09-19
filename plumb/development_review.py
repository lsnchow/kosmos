"""Private, local-only persistence for explicitly entered development reviews.

This module is intentionally not a Gate D workflow.  It exposes only opaque
review packets, accepts labels only from a browser POST made by a real user, and
keeps drafts in a private SQLite database.  It has no export, calibration,
annotation, teacher-output, or source-metadata integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response

from .annotation import PROGRESS_SCALE_LABELS
from .records import canonical_json, utc_now


COOKIE_NAME = "plumb_development_review"
_SET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
_REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
REVIEWER_KINDS = {"human_self_reported", "model_assisted"}
INTEGRITY_VALUES = {"intact", "artifact", "uncertain"}
COLLISION_VALUES = {"none_visible", "visible", "uncertain"}
COMPLETION_VALUES = {"met", "not_met", "uncertain"}


class DevelopmentReviewError(ValueError):
    """A packet, session, or entered review fails the local-only contract."""


class ReviewSessionError(DevelopmentReviewError):
    """A cookie/session pair cannot read or write this review scope."""


@dataclass(frozen=True)
class ReviewSet:
    set_id: str
    worksheet_path: Path
    clips: tuple[Mapping[str, Any], ...]


def _safe_http_url(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DevelopmentReviewError("%s must be an opaque HTTP(S) URL" % field)
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise DevelopmentReviewError("%s must be an opaque HTTP(S) URL" % field)
    return value


def _review_set_id(value: str) -> str:
    if not _SET_ID_RE.fullmatch(value):
        raise DevelopmentReviewError("invalid review set identifier")
    return value


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_expiry(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class DevelopmentReviewStore:
    """Packet reader plus a private local SQLite draft store."""

    def __init__(
        self,
        repo_root: Path,
        review_root: Path,
        served_root: Path,
        private_root: Path,
        session_ttl_seconds: int = 8 * 60 * 60,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.review_root = Path(review_root).resolve()
        self.served_root = Path(served_root).resolve()
        self.private_root = Path(private_root).absolute()
        if isinstance(session_ttl_seconds, bool) or not isinstance(session_ttl_seconds, int) or not 60 <= session_ttl_seconds <= 12 * 60 * 60:
            raise DevelopmentReviewError("development-review session TTL must be an integer from 60 through 43200 seconds")
        self.session_ttl_seconds = session_ttl_seconds
        self.db_path = self.private_root / "development-review.sqlite3"
        self._lock = threading.RLock()
        self._assert_private_location()

    def _connect(self) -> sqlite3.Connection:
        # The storage root can be swapped after construction.  Re-check before
        # every database open rather than trusting the one-time constructor
        # validation, and keep this one-way (the assertion never connects).
        self._assert_private_location()
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _is_within(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise DevelopmentReviewError("private review storage cannot use symlink components")

    def _assert_private_location(self) -> None:
        self._reject_symlink_components(self.private_root)
        canonical = self.private_root.resolve(strict=False)
        if self._is_within(canonical, self.served_root):
            raise DevelopmentReviewError("private review storage must be outside the served data root")
        if not self.private_root.parent.is_dir():
            raise DevelopmentReviewError("private review storage parent must already exist")
        if self.private_root.exists() and self.private_root.stat().st_mode & 0o077:
            raise DevelopmentReviewError("existing private review storage directory must be mode 0700")
        for candidate in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal"), self.db_path.with_name(self.db_path.name + "-shm")):
            if candidate.is_symlink():
                raise DevelopmentReviewError("private review database paths cannot be symlinks")
            if candidate.exists():
                resolved = candidate.resolve()
                if not self._is_within(resolved, canonical) or self._is_within(resolved, self.served_root):
                    raise DevelopmentReviewError("private review database must remain outside the served data root")

    def _initialise(self) -> None:
        """Create private storage only after an explicit session POST."""

        self._assert_private_location()
        created = not self.private_root.exists()
        self.private_root.mkdir(mode=0o700, exist_ok=True)
        if created:
            os.chmod(self.private_root, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  token_sha256 TEXT NOT NULL UNIQUE,
                  review_set_id TEXT NOT NULL,
                  reviewer_id TEXT NOT NULL,
                  reviewer_kind TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  last_seen_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ratings (
                  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                  opaque_clip_id TEXT NOT NULL,
                  integrity TEXT,
                  collision TEXT,
                  progress INTEGER,
                  completion_evidence TEXT,
                  evidence_frame_indices_json TEXT,
                  observable_reason TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (session_id, opaque_clip_id)
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
            if "expires_at" not in columns:
                raise DevelopmentReviewError("existing development-review database lacks session expiry; remove it before use")
        if not self.db_path.exists():
            raise DevelopmentReviewError("private review database was not created")
        for candidate in (
            self.db_path,
            self.db_path.with_name(self.db_path.name + "-wal"),
            self.db_path.with_name(self.db_path.name + "-shm"),
        ):
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def _load_set(self, set_id: str) -> ReviewSet:
        set_id = _review_set_id(set_id)
        root = self.review_root.resolve()
        path = (root / set_id / "blank-worksheets.jsonl").resolve()
        if root not in path.parents or not path.is_file():
            raise KeyError("review set not found")
        clips = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    clips.append(self._validate_packet(value))
        except (OSError, ValueError, TypeError) as error:
            raise DevelopmentReviewError("review packet file is invalid") from error
        if not clips:
            raise DevelopmentReviewError("review set has no packets")
        return ReviewSet(set_id=set_id, worksheet_path=path, clips=tuple(clips))

    @staticmethod
    def _validate_packet(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise DevelopmentReviewError("review packet must be an object")
        if value.get("schema") != "plumb-development-review-packets-v1":
            raise DevelopmentReviewError("review packet schema is not supported")
        if value.get("purpose") != "development_review_only" or value.get("gate_d_eligible") is not False:
            raise DevelopmentReviewError("review packet is not development-review-only")
        clip_id = value.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.startswith("review-clip-"):
            raise DevelopmentReviewError("review packet lacks an opaque clip identifier")
        task = value.get("task")
        instruction = value.get("task_instruction")
        rubric = value.get("task_rubric")
        if not all(isinstance(item, str) and item for item in (task, instruction, rubric)):
            raise DevelopmentReviewError("review packet task fields are invalid")
        video = _safe_http_url(value.get("raw_video_url"), "raw_video_url")
        guidance = value.get("review_guidance")
        if not isinstance(guidance, Mapping):
            raise DevelopmentReviewError("review packet lacks reviewer guidance")
        integrity_guidance = guidance.get("integrity")
        collision_guidance = guidance.get("collision")
        completion_guidance = guidance.get("completion")
        if (
            not isinstance(integrity_guidance, Mapping)
            or not isinstance(collision_guidance, Mapping)
            or not isinstance(completion_guidance, str)
            or set(integrity_guidance) != INTEGRITY_VALUES
            or set(collision_guidance) != COLLISION_VALUES
            or any(not isinstance(item, str) or not item for item in integrity_guidance.values())
            or any(not isinstance(item, str) or not item for item in collision_guidance.values())
            or not completion_guidance
        ):
            raise DevelopmentReviewError("review packet guidance is invalid")
        frames = value.get("sampled_frames")
        if not isinstance(frames, list) or len(frames) != 16:
            raise DevelopmentReviewError("review packet must contain exactly 16 sampled frames")
        safe_frames = []
        seen = set()
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise DevelopmentReviewError("review frame must be an object")
            index = frame.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index > 15 or index in seen:
                raise DevelopmentReviewError("review frame index is invalid")
            seen.add(index)
            timestamp = frame.get("timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
                raise DevelopmentReviewError("review frame timestamp is invalid")
            safe_frames.append({"index": index, "url": _safe_http_url(frame.get("url"), "frame.url"), "timestamp": timestamp})
        if seen != set(range(16)):
            raise DevelopmentReviewError("review packet frame indices must be exactly 0 through 15")
        safe_frames.sort(key=lambda frame: frame["index"])
        if any(
            float(current["timestamp"]) <= float(previous["timestamp"])
            for previous, current in zip(safe_frames, safe_frames[1:])
        ):
            raise DevelopmentReviewError("review packet frame timestamps must be strictly increasing by frame index")
        return {
            "opaque_clip_id": clip_id,
            "task": {
                "id": task,
                "instruction": instruction,
                "rubric": rubric,
                "progress_definitions": list(PROGRESS_SCALE_LABELS),
                "definitions": {
                    "integrity": dict(integrity_guidance),
                    "collision": dict(collision_guidance),
                    "completion": completion_guidance,
                },
            },
            "media": {"video_url": video, "frames": safe_frames},
        }

    def list_sets(self) -> list[dict[str, Any]]:
        root = self.review_root
        if not root.is_dir():
            return []
        values = []
        for candidate in sorted(root.iterdir(), key=lambda item: item.name):
            if candidate.is_symlink() or not candidate.is_dir() or not _SET_ID_RE.fullmatch(candidate.name):
                continue
            try:
                review_set = self._load_set(candidate.name)
            except (DevelopmentReviewError, KeyError):
                continue
            values.append(
                {
                    "set_id": review_set.set_id,
                    "title": review_set.set_id,
                    "clip_count": len(review_set.clips),
                    "status": "development_review_only",
                }
            )
        return values

    def create_session(self, set_id: str, reviewer_id: str, reviewer_kind: str) -> tuple[dict[str, str], str]:
        review_set = self._load_set(set_id)
        if not isinstance(reviewer_id, str) or not _REVIEWER_ID_RE.fullmatch(reviewer_id):
            raise DevelopmentReviewError("reviewer_id must be an explicitly entered identifier of at most 128 characters")
        if reviewer_kind not in REVIEWER_KINDS:
            raise DevelopmentReviewError("reviewer_kind must be human_self_reported or model_assisted")
        session_id = "devreview-" + uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = _session_expiry(self.session_ttl_seconds)
        self._initialise()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO sessions(
                     session_id, token_sha256, review_set_id, reviewer_id, reviewer_kind, created_at, last_seen_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, _token_digest(token), review_set.set_id, reviewer_id, reviewer_kind, now, now, expires_at),
            )
        return {"session_id": session_id, "reviewer_id": reviewer_id, "reviewer_kind": reviewer_kind}, token

    def _session(self, session_id: str, token: Optional[str], set_id: Optional[str] = None) -> Mapping[str, Any]:
        if not isinstance(session_id, str) or not session_id.startswith("devreview-") or not token:
            raise ReviewSessionError("review session is missing or invalid")
        self._assert_private_location()
        if not self.db_path.is_file():
            raise ReviewSessionError("review session is not authorized")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND token_sha256 = ? AND expires_at > ?",
                (session_id, _token_digest(token), utc_now()),
            ).fetchone()
            if row is None or (set_id is not None and row["review_set_id"] != set_id):
                raise ReviewSessionError("review session is not authorized for this set")
            connection.execute("UPDATE sessions SET last_seen_at = ? WHERE session_id = ?", (utc_now(), session_id))
            return dict(row)

    @staticmethod
    def _rating_payload(value: Mapping[str, Any], valid_indexes: Iterable[int]) -> dict[str, Any]:
        allowed_fields = {
            "integrity",
            "collision",
            "progress",
            "completion_evidence",
            "evidence_frame_indices",
            "observable_reason",
        }
        if not value or set(value) - allowed_fields:
            raise DevelopmentReviewError("rating must contain one or more known development-review fields")
        payload: dict[str, Any] = {}
        for name, allowed in (
            ("integrity", INTEGRITY_VALUES),
            ("collision", COLLISION_VALUES),
            ("completion_evidence", COMPLETION_VALUES),
        ):
            if name in value:
                if value[name] is None:
                    payload[name] = None
                    continue
                if not isinstance(value[name], str) or value[name] not in allowed:
                    raise DevelopmentReviewError("rating enum value is invalid")
                payload[name] = value[name]
        if "progress" in value:
            progress = value["progress"]
            if progress is None:
                payload["progress"] = None
            elif isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 5:
                raise DevelopmentReviewError("progress must be an integer from 0 through 5")
            else:
                payload["progress"] = progress
        if "evidence_frame_indices" in value:
            frames = value["evidence_frame_indices"]
            if not isinstance(frames, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in frames):
                raise DevelopmentReviewError("evidence_frame_indices must be an integer list")
            allowed = set(valid_indexes)
            if len(frames) != len(set(frames)) or any(item not in allowed for item in frames):
                raise DevelopmentReviewError("evidence_frame_indices names an unavailable frame")
            payload["evidence_frame_indices"] = list(frames)
        if "observable_reason" in value:
            reason = value["observable_reason"]
            if not isinstance(reason, str) or len(reason) > 2000:
                raise DevelopmentReviewError("observable_reason must be at most 2000 characters")
            payload["observable_reason"] = reason
        return payload

    def _draft(self, session_id: str, clip_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ratings WHERE session_id = ? AND opaque_clip_id = ?", (session_id, clip_id)
            ).fetchone()
        if row is None:
            return None
        draft: dict[str, Any] = {}
        for name in ("integrity", "collision", "completion_evidence", "observable_reason"):
            if row[name] is not None:
                draft[name] = row[name]
        if row["progress"] is not None:
            draft["progress"] = int(row["progress"])
        if row["evidence_frame_indices_json"] is not None:
            draft["evidence_frame_indices"] = json.loads(row["evidence_frame_indices_json"])
        return draft

    @staticmethod
    def _validate_rating_consistency(draft: Mapping[str, Any]) -> None:
        completion = draft.get("completion_evidence")
        progress = draft.get("progress")
        if completion == "met" and progress != 5:
            raise DevelopmentReviewError("completion_evidence='met' requires progress 5")
        if completion == "not_met" and progress is not None and progress not in {0, 1, 2, 3, 4}:
            raise DevelopmentReviewError("completion_evidence='not_met' requires progress from 0 through 4")

    @classmethod
    def _is_completed_draft(cls, draft: Mapping[str, Any]) -> bool:
        required = {
            "integrity",
            "collision",
            "progress",
            "completion_evidence",
            "evidence_frame_indices",
            "observable_reason",
        }
        if not required.issubset(draft):
            return False
        try:
            cls._validate_rating_consistency(draft)
        except DevelopmentReviewError:
            return False
        evidence = draft.get("evidence_frame_indices")
        reason = draft.get("observable_reason")
        if not isinstance(evidence, list) or not evidence:
            return False
        if not isinstance(reason, str) or not reason.strip():
            return False
        return True

    def clips(self, set_id: str, session_id: Optional[str] = None, token: Optional[str] = None) -> dict[str, Any]:
        review_set = self._load_set(set_id)
        session = self._session(session_id, token, review_set.set_id) if session_id else None
        values = []
        saved_drafts = 0
        completed = 0
        for clip in review_set.clips:
            item = dict(clip)
            if session is not None:
                draft = self._draft(str(session["session_id"]), str(clip["opaque_clip_id"]))
                if draft is not None:
                    item["draft"] = draft
                    saved_drafts += 1
                    if self._is_completed_draft(draft):
                        completed += 1
            values.append(item)
        return {
            "set_id": review_set.set_id,
            "clips": values,
            "progress": {"saved_drafts": saved_drafts, "completed": completed, "total": len(values)},
        }

    def rating(self, set_id: str, session_id: str, token: Optional[str], clip_id: str) -> Optional[dict[str, Any]]:
        review_set = self._load_set(set_id)
        self._session(session_id, token, review_set.set_id)
        if clip_id not in {clip["opaque_clip_id"] for clip in review_set.clips}:
            raise KeyError("opaque clip not found")
        return self._draft(session_id, clip_id)

    def save_rating(
        self, set_id: str, session_id: str, token: Optional[str], clip_id: str, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        review_set = self._load_set(set_id)
        self._session(session_id, token, review_set.set_id)
        clip = next((item for item in review_set.clips if item["opaque_clip_id"] == clip_id), None)
        if clip is None:
            raise KeyError("opaque clip not found")
        payload = self._rating_payload(value, (frame["index"] for frame in clip["media"]["frames"]))
        current = self._draft(session_id, clip_id) or {}
        merged = dict(current)
        for name, item in payload.items():
            if item is None:
                merged.pop(name, None)
            else:
                merged[name] = item
        self._validate_rating_consistency(merged)
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO ratings(
                       session_id, opaque_clip_id, integrity, collision, progress, completion_evidence,
                       evidence_frame_indices_json, observable_reason, created_at, updated_at
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(session_id, opaque_clip_id) DO UPDATE SET
                       integrity = excluded.integrity, collision = excluded.collision, progress = excluded.progress,
                       completion_evidence = excluded.completion_evidence,
                       evidence_frame_indices_json = excluded.evidence_frame_indices_json,
                       observable_reason = excluded.observable_reason,
                       updated_at = excluded.updated_at""",
                (
                    session_id,
                    clip_id,
                    merged.get("integrity"),
                    merged.get("collision"),
                    merged.get("progress"),
                    merged.get("completion_evidence"),
                    canonical_json(merged["evidence_frame_indices"]) if "evidence_frame_indices" in merged else None,
                    merged.get("observable_reason"),
                    now,
                    now,
                ),
            )
            connection.commit()
        saved = self._draft(session_id, clip_id)
        assert saved is not None
        return saved


def _safe_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise HTTPException(403, "Unsafe development-review origin")
    try:
        request_host = request.url.hostname
        request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise HTTPException(403, "Unsafe development-review origin")
    if (
        parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or request_host != parsed.hostname
        or request.url.scheme != parsed.scheme
        or request_port != origin_port
    ):
        raise HTTPException(403, "Unsafe development-review origin")


def register_development_review_routes(app: FastAPI, store: DevelopmentReviewStore) -> None:
    """Register only private draft endpoints; no annotation/calibration bridge."""

    def token(request: Request) -> Optional[str]:
        return request.cookies.get(COOKIE_NAME)

    @app.get("/api/development-review/sets")
    def list_sets() -> dict[str, Any]:
        return {"sets": store.list_sets()}

    @app.get("/api/development-review/sets/{set_id}/clips")
    def list_clips(set_id: str, request: Request, session_id: Optional[str] = None) -> dict[str, Any]:
        try:
            return store.clips(set_id, session_id=session_id, token=token(request))
        except KeyError:
            raise HTTPException(404, "Development review set not found")
        except ReviewSessionError:
            raise HTTPException(403, "Development review session is not authorized")
        except DevelopmentReviewError as error:
            raise HTTPException(422, str(error))

    @app.post("/api/development-review/sessions", status_code=201)
    async def create_session(request: Request) -> Response:
        _safe_origin(request)
        try:
            try:
                value = await request.json()
            except json.JSONDecodeError as error:
                raise DevelopmentReviewError("session body must be valid JSON") from error
            if not isinstance(value, Mapping) or set(value) != {"set_id", "reviewer_id", "reviewer_kind"}:
                raise DevelopmentReviewError("session requires set_id, reviewer_id, and reviewer_kind")
            payload, session_token = store.create_session(
                str(value["set_id"]), str(value["reviewer_id"]), str(value["reviewer_kind"])
            )
        except KeyError:
            raise HTTPException(404, "Development review set not found")
        except DevelopmentReviewError as error:
            raise HTTPException(422, str(error))
        response = Response(content=canonical_json(payload), media_type="application/json", status_code=201)
        response.set_cookie(
            COOKIE_NAME,
            session_token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/api/development-review",
            max_age=store.session_ttl_seconds,
        )
        return response

    @app.get("/api/development-review/sessions/{session_id}/ratings/{opaque_clip_id}")
    def get_rating(session_id: str, opaque_clip_id: str, set_id: str, request: Request) -> dict[str, Any]:
        try:
            value = store.rating(set_id, session_id, token(request), opaque_clip_id)
        except KeyError:
            raise HTTPException(404, "Development review set or opaque clip not found")
        except ReviewSessionError:
            raise HTTPException(403, "Development review session is not authorized")
        except DevelopmentReviewError as error:
            raise HTTPException(422, str(error))
        return {"session_id": session_id, "opaque_clip_id": opaque_clip_id, "draft": value}

    @app.put("/api/development-review/sessions/{session_id}/ratings/{opaque_clip_id}")
    async def put_rating(session_id: str, opaque_clip_id: str, set_id: str, request: Request) -> dict[str, Any]:
        _safe_origin(request)
        try:
            try:
                value = await request.json()
            except json.JSONDecodeError as error:
                raise DevelopmentReviewError("rating body must be valid JSON") from error
            if not isinstance(value, Mapping):
                raise DevelopmentReviewError("rating must be an object")
            draft = store.save_rating(set_id, session_id, token(request), opaque_clip_id, value)
        except KeyError:
            raise HTTPException(404, "Development review set or opaque clip not found")
        except ReviewSessionError:
            raise HTTPException(403, "Development review session is not authorized")
        except DevelopmentReviewError as error:
            raise HTTPException(422, str(error))
        return {"session_id": session_id, "opaque_clip_id": opaque_clip_id, "draft": draft}


__all__ = [
    "COOKIE_NAME",
    "DevelopmentReviewError",
    "DevelopmentReviewStore",
    "ReviewSessionError",
    "register_development_review_routes",
]
