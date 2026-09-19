"""Durable application-side delivery for Baseten Chain async requests.

This module has deliberately narrow authority.  It makes exactly one POST for
each durable logical submission, stores a signed webhook before handing it to
an artifact writer, and exposes an explicitly injected reconciler for a
*verified* request-management route.  It does not infer a Chain status/cancel
URL from the submission URL: Baseten's public async status/cancel examples are
model-ID routes, while the documented Chain async invocation page only defines
the Chain submission routes.  An account-tested controller is therefore a
deployment prerequisite.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from .ledger import Ledger
from .platform import (
    AmbiguousSubmissionError,
    AsyncChainRequestOptions,
    AuthenticatedCallback,
    BasetenChainClient,
    CallbackAssociation,
    PlatformError,
    PlatformHTTPError,
)


class RemoteRequestController(Protocol):
    """An account-tested lifecycle adapter for submitted Chain request IDs.

    The adapter must return a documented lifecycle value such as ``QUEUED``,
    ``IN_PROGRESS``, ``SUCCEEDED``, ``FAILED``, ``EXPIRED``, ``CANCELED``, or
    ``WEBHOOK_FAILED``.  It is injected rather than manufactured from a Chain
    URL because the available vendor documentation does not establish that
    route for Chains.
    """

    async def get_status(self, request_id: str) -> str:
        """Return the remote lifecycle state without claiming to retrieve output."""

    async def cancel_queued(self, request_id: str) -> str:
        """Attempt one cancellation and return its resulting lifecycle state."""


@dataclass(frozen=True)
class DispatchResult:
    outbox_id: str
    disposition: str
    request_id: Optional[str] = None
    error: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CallbackReceipt:
    callback: AuthenticatedCallback
    duplicate: bool
    associated: bool


@dataclass(frozen=True)
class ReconciliationReport:
    recovered_dispatches: int
    recovered_callback_claims: int
    recovered_cancellations: int
    observed: Tuple[str, ...]
    cancellation_attempted: Tuple[str, ...]
    unavailable: Tuple[str, ...]
    errors: Tuple[Mapping[str, str], ...]


class BasetenOutbox:
    """Coordinates the durable ledger with ``BasetenChainClient``.

    Artifact persistence and mapping a Chain's result to PLUMB's normalized
    logical result happen outside this class.  The caller obtains a callback
    finalization lease, writes the artifact, and then calls
    :meth:`finalise_callback`.  That ordering ensures a terminal logical row
    never points at an unwritten final artifact.
    """

    @staticmethod
    def _positive_finite(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError("%s must be a finite positive value" % label)
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("%s must be a finite positive value" % label) from error
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("%s must be a finite positive value" % label)
        return normalized

    def __init__(
        self,
        ledger: Ledger,
        client: BasetenChainClient,
        dispatch_lease_seconds: float = 30.0,
        callback_lease_seconds: float = 60.0,
        cancellation_lease_seconds: float = 30.0,
    ) -> None:
        self.ledger = ledger
        self.client = client
        self.dispatch_lease_seconds = self._positive_finite(dispatch_lease_seconds, "dispatch_lease_seconds")
        self.callback_lease_seconds = self._positive_finite(callback_lease_seconds, "callback_lease_seconds")
        self.cancellation_lease_seconds = self._positive_finite(
            cancellation_lease_seconds, "cancellation_lease_seconds"
        )

    def enqueue(
        self,
        outbox_id: str,
        run_id: str,
        episode_id: str,
        entrypoint_input: Mapping[str, Any],
        options: AsyncChainRequestOptions,
        deadline_at: str,
    ) -> Dict[str, Any]:
        """Durably queue one Chain entrypoint request without sending it."""

        return self.ledger.enqueue_baseten_submission(
            outbox_id,
            run_id,
            episode_id,
            entrypoint_input,
            options.as_payload_fields(),
            deadline_at,
        )

    @staticmethod
    def _options_from_record(value: Mapping[str, Any]) -> AsyncChainRequestOptions:
        webhook_endpoint = value.get("webhook_endpoint")
        if not isinstance(webhook_endpoint, str):
            raise ValueError("durable outbox options are missing webhook_endpoint")
        return AsyncChainRequestOptions(
            webhook_endpoint=webhook_endpoint,
            priority=value.get("priority"),
            max_time_in_queue_seconds=value.get("max_time_in_queue_seconds"),
        )

    @staticmethod
    def _error_record(error: BaseException) -> Dict[str, Any]:
        status_code = getattr(error, "status_code", None)
        record: Dict[str, Any] = {"kind": type(error).__name__, "message": str(error)}
        if isinstance(status_code, int):
            record["status_code"] = status_code
        return record

    async def dispatch(self, outbox_id: str, dispatch_token: Optional[str] = None) -> DispatchResult:
        """Make the one allowed POST after committing the dispatch boundary.

        A transport error, 408, 5xx, unreadable success body, unexpected
        client failure, or a crash after the pre-POST transaction is all
        treated as ambiguous.  None of those paths sends a second POST.
        """

        token = dispatch_token or uuid.uuid4().hex
        claimed = self.ledger.claim_baseten_dispatch(outbox_id, token, self.dispatch_lease_seconds)
        if claimed is None:
            return DispatchResult(outbox_id=outbox_id, disposition="not_dispatchable")
        try:
            receipt = await self.client.submit_async(
                claimed["payload"], self._options_from_record(claimed["options"])
            )
        except AmbiguousSubmissionError as error:
            self.ledger.record_baseten_ambiguity(outbox_id, token, self._error_record(error.cause))
            return DispatchResult(outbox_id=outbox_id, disposition="ambiguous", error=self._error_record(error.cause))
        except PlatformHTTPError as error:
            # The client reserves this exception for proved non-ambiguous
            # rejections (ambiguous statuses become AmbiguousSubmissionError).
            self.ledger.record_baseten_known_rejection(outbox_id, token, self._error_record(error))
            return DispatchResult(outbox_id=outbox_id, disposition="rejected", error=self._error_record(error))
        except PlatformError as error:
            self.ledger.record_baseten_ambiguity(outbox_id, token, self._error_record(error))
            return DispatchResult(outbox_id=outbox_id, disposition="ambiguous", error=self._error_record(error))
        except Exception as error:  # a client interruption may happen after bytes left the process
            self.ledger.record_baseten_ambiguity(outbox_id, token, self._error_record(error))
            return DispatchResult(outbox_id=outbox_id, disposition="ambiguous", error=self._error_record(error))
        try:
            persisted = self.ledger.record_baseten_acceptance(
                outbox_id, token, receipt.request_id, receipt.response
            )
        except Exception as error:
            # We have observed a request ID but failed to make it durable.  Do
            # not re-POST; recovery must use a reviewed manual reconciliation.
            return DispatchResult(
                outbox_id=outbox_id,
                disposition="acceptance_persistence_failed",
                request_id=receipt.request_id,
                error=self._error_record(error),
            )
        if not persisted:
            return DispatchResult(
                outbox_id=outbox_id,
                disposition="acceptance_lost_race",
                request_id=receipt.request_id,
            )
        return DispatchResult(outbox_id=outbox_id, disposition="accepted", request_id=receipt.request_id)

    def ingest_callback(self, raw_body: bytes, headers: Mapping[str, str]) -> CallbackReceipt:
        """Authenticate and persist a webhook before assigning it to an episode.

        It is valid for a callback to reach the application before a process
        has stored the POST response.  Such a receipt remains unassociated in
        the database and is atomically attached when its request ID is later
        acknowledged.
        """

        callback = self.client.authenticate_callback(raw_body, headers)
        signature_header = next(
            (value for key, value in headers.items() if key.lower() == "x-baseten-signature"), None
        )
        request_id_header = next(
            (value for key, value in headers.items() if key.lower() == "x-baseten-request-id"), None
        )
        stored = self.ledger.record_baseten_callback(
            callback.request_id,
            raw_body,
            callback.raw,
            callback.event_type,
            signature_header=signature_header,
            request_id_header=request_id_header,
        )
        outbox_id = stored["outbox_id"]
        if outbox_id is None:
            return CallbackReceipt(callback=callback, duplicate=bool(stored["duplicate"]), associated=False)
        outbox = self.ledger.get_baseten_submission(str(outbox_id))
        association = CallbackAssociation(
            request_id=callback.request_id,
            run_id=str(outbox["run_id"]),
            episode_id=str(outbox["episode_id"]),
        )
        return CallbackReceipt(
            callback=replace(callback, association=association),
            duplicate=bool(stored["duplicate"]),
            associated=True,
        )

    def claim_callback_finalization(
        self, outbox_id: str, processor_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Claim the saved callback before writing its final artifact."""

        return self.ledger.claim_baseten_callback_finalization(
            outbox_id, processor_token or uuid.uuid4().hex, self.callback_lease_seconds
        )

    def finalise_callback(
        self,
        outbox_id: str,
        processor_token: str,
        terminal_status: str,
        artifact_refs: Mapping[str, Any],
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
        cancellation_reason: Optional[str] = None,
    ) -> bool:
        return self.ledger.finalise_baseten_callback(
            outbox_id,
            processor_token,
            terminal_status,
            artifact_refs,
            result=result,
            error=error,
            cancellation_reason=cancellation_reason,
        )

    def request_cancellation(self, outbox_id: str) -> Dict[str, Any]:
        """Record local cancellation intent; remote cancellation remains reconciled."""

        return self.ledger.request_baseten_cancellation(outbox_id)

    def finalise_pre_dispatch_cancellation(
        self, outbox_id: str, artifact_refs: Mapping[str, Any], reason: str = "cancelled_before_submission"
    ) -> bool:
        return self.ledger.finalise_baseten_pre_dispatch_cancellation(outbox_id, artifact_refs, reason)

    def finalise_remote_terminal(
        self,
        outbox_id: str,
        terminal_status: str,
        artifact_refs: Mapping[str, Any],
        error: Optional[Mapping[str, Any]] = None,
        cancellation_reason: Optional[str] = None,
    ) -> bool:
        """Record a callback-less failed/expired/cancelled remote attempt."""

        return self.ledger.finalise_baseten_remote_terminal(
            outbox_id,
            terminal_status,
            artifact_refs,
            error=error,
            cancellation_reason=cancellation_reason,
        )

    async def reconcile(
        self, controller: Optional[RemoteRequestController] = None
    ) -> ReconciliationReport:
        """Recover local leases and use a verified remote controller when supplied.

        If no controller is supplied, the report identifies request records
        that need an account-tested lifecycle route.  This is intentionally not
        an approximation from queue depth or a guessed Chain URL.
        """

        recovered_dispatches = self.ledger.recover_expired_baseten_dispatches()
        recovered_callbacks = self.ledger.recover_expired_baseten_callback_claims()
        recovered_cancellations = self.ledger.recover_expired_baseten_remote_cancellations()
        observed = []
        cancellation_attempted = []
        unavailable = []
        errors = []
        candidates = self.ledger.list_baseten_submissions(
            states=(
                "accepted",
                "ambiguous",
                "cancellation_requested",
                "cancellation_ambiguous",
                "callback_missing",
                "remote_terminal_without_callback",
            )
        )
        for record in candidates:
            outbox_id = str(record["outbox_id"])
            request_id = record.get("platform_request_id")
            if not request_id:
                # A missing request ID is the definition of an ambiguous POST.
                unavailable.append(outbox_id)
                continue
            if controller is None:
                unavailable.append(outbox_id)
                continue
            try:
                status = (await controller.get_status(str(request_id))).upper()
            except Exception as error:
                errors.append({"outbox_id": outbox_id, "operation": "status", "error": str(error)})
                continue
            self.ledger.record_baseten_remote_status(outbox_id, status)
            observed.append(outbox_id)
            refreshed = self.ledger.get_baseten_submission(outbox_id)
            # Baseten documents that only queued requests can be cancelled.
            # Commit the cancel boundary before the injected controller issues
            # its one DELETE.  A delete error becomes reconciliation work, not
            # an automatic second DELETE.
            if refreshed["cancellation_requested"] and status == "QUEUED":
                token = uuid.uuid4().hex
                claimed = self.ledger.claim_baseten_remote_cancellation(
                    outbox_id, token, self.cancellation_lease_seconds
                )
                if claimed is None:
                    continue
                try:
                    cancellation_status = (await controller.cancel_queued(str(request_id))).upper()
                except Exception as error:
                    self.ledger.record_baseten_remote_cancellation_ambiguity(
                        outbox_id, token, self._error_record(error)
                    )
                    errors.append({"outbox_id": outbox_id, "operation": "cancel", "error": str(error)})
                    continue
                self.ledger.record_baseten_remote_cancellation_status(
                    outbox_id, token, cancellation_status
                )
                cancellation_attempted.append(outbox_id)
        return ReconciliationReport(
            recovered_dispatches=recovered_dispatches,
            recovered_callback_claims=recovered_callbacks,
            recovered_cancellations=recovered_cancellations,
            observed=tuple(observed),
            cancellation_attempted=tuple(cancellation_attempted),
            unavailable=tuple(unavailable),
            errors=tuple(errors),
        )
