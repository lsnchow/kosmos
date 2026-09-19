"""Dependency-light policy contracts and safe lazy-loader primitives.

Nothing in :mod:`plumb.policies` imports a model framework at module import
time.  That is intentional: a CPU control-plane host should be able to read
the declared native contracts without accidentally importing Torch, JAX, or
Transformers, and should never cause a checkpoint download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation


OPENVLA_SOURCE_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"
AUTOEVAL_POLICY_SOURCE_COMMIT = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
OPEN_PI_ZERO_SOURCE_COMMIT = "c3df7fb062175c16f69d7ca4ce042958ea238fb7"


class PolicyLoadError(RuntimeError):
    """A requested local policy cannot safely be loaded in this runtime."""


class PolicyContractError(ValueError):
    """A caller supplied an observation or action outside a native contract."""


class NativePolicyCallable(Protocol):
    """Minimal protocol for an explicitly supplied, separately certified wrapper."""

    def __call__(self, observation: PolicyObservation) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class ExternalPolicyProfile:
    """Identity for a policy that has no bundled implementation yet.

    ``loader_hook`` is deliberately an injection point rather than an import
    of a guessed third-party API.  An operator may attach a source-reviewed
    wrapper in its own isolated image.  Doing so changes neither this adapter's
    unqualified status nor its need for fixture certification.
    """

    profile_id: str
    local_model_path: Optional[str] = None
    checkpoint_revision: Optional[str] = None
    loader_revision: Optional[str] = None
    local_files_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExternalPolicyHook:
    """Honest placeholder for incompatible native policy environments.

    It never synthesizes an action.  A callable can be injected only by the
    deployment which owns a reviewed native wrapper; tests can use the same
    hook without importing an ML stack.
    """

    def __init__(
        self,
        contract: PolicyContract,
        profile: ExternalPolicyProfile,
        *,
        loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None,
        source_urls: Tuple[str, ...] = (),
    ) -> None:
        self.contract = contract
        self.profile = profile
        self._loader_hook = loader_hook
        self._policy: Optional[NativePolicyCallable] = None
        self._source_urls = source_urls

    def capability(self) -> CapabilityResult:
        if self._loader_hook is None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason=(
                    "%s has a declared native contract but no source-reviewed local loader in PLUMB. "
                    "No inferred or placeholder actions will be emitted."
                )
                % self.contract.name,
                source_verified=bool(self._source_urls),
                evidence_uris=self._source_urls,
                details={
                    "profile_id": self.profile.profile_id,
                    "native_proposal_horizon": self.contract.native_proposal_horizon,
                    "certified_execute_prefix": self.contract.certified_execute_prefix,
                    "local_files_only": self.profile.local_files_only,
                },
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "%s has an explicitly injected local native wrapper. Fixture validation and Gate B are "
                "still required; this adapter cannot mark it qualified."
            )
            % self.contract.name,
            source_verified=bool(self._source_urls),
            evidence_uris=self._source_urls,
            details={"profile_id": self.profile.profile_id, "loader_revision": self.profile.loader_revision},
        )

    def _ensure_policy(self) -> NativePolicyCallable:
        if self._policy is not None:
            return self._policy
        if self._loader_hook is None:
            raise PolicyLoadError(
                "%s is intentionally unimplemented until a pinned native wrapper and fixture evidence are supplied."
                % self.contract.name
            )
        policy = self._loader_hook(self.profile)
        if not callable(policy):
            raise PolicyLoadError("The injected %s loader did not return a callable native policy." % self.contract.name)
        self._policy = policy
        return policy

    def predict(self, observation: PolicyObservation) -> Tuple[float, ...]:
        """Call an injected wrapper without filling or extending its proposal."""

        action = self._ensure_policy()(observation)
        if isinstance(action, (str, bytes)):
            raise PolicyContractError("Native policy action must be a numeric sequence, not text.")
        values = tuple(float(value) for value in action)
        if len(values) != 7:
            raise PolicyContractError(
                "%s injected wrapper returned %d values; PLUMB canonical physical actions are 7-D."
                % (self.contract.name, len(values))
            )
        return values
