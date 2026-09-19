"""Legacy compatibility hooks for policy stacks with isolated native adapters.

Octo's source-backed loader lives in :mod:`plumb.policies.octo` and currently
certifies only its pinned v0.1 normalized-action API; the AutoEval server's
newer ``unnormalization_statistics`` call is a distinct unverified profile.
MiniVLA and SuSIE have distinct source-adapter APIs in
:mod:`plumb.policies.minivla` and :mod:`plumb.policies.susie`. The historic
``*Policy`` classes here retain an explicit injection-only compatibility
boundary. They do not become live loaders merely because a separately named
adapter exists, and never invent an action on a planning host.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from plumb.adapters.contracts import CapabilityStatus, PolicyContract

from .contracts import AUTOEVAL_POLICY_SOURCE_COMMIT, OPEN_PI_ZERO_SOURCE_COMMIT, ExternalPolicyHook, ExternalPolicyProfile, NativePolicyCallable
from .octo import (
    OCTO_BASE_V1_CONTRACT,
    OCTO_SMALL_V1_CONTRACT,
    OctoBaseV1Policy,
    OctoSmallV1Policy,
)


AUTOEVAL_POLICY_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/auto_eval/robot/policy.py"
)
OPEN_PI_ZERO_SOURCE = (
    "https://github.com/allenzren/open-pi-zero/tree/" + OPEN_PI_ZERO_SOURCE_COMMIT
)
MINIVLA_SOURCE = "https://github.com/Stanford-ILIAD/openvla-mini"
SUSIE_SOURCE = "https://github.com/kvablack/susie"


MINIVLA_CONTRACT = PolicyContract(
    name="MiniVLA",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=7,
    certified_execute_prefix=None,
    temporal_ensembling=False,
    preprocessing="Released openvla-mini preprocessing with use_extra=True must be fixture-certified.",
    normalization="Checkpoint-native action normalization and gripper conversion are unresolved.",
    reset_rule="Released wrapper reset behavior must be preserved.",
    rng_rule="Pinned Torch RNG stream required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation=(
        "Legacy compatibility hook only. The distinct source adapter remains blocked until the VQ asset's "
        "license is resolved and the released one-action VQ boundary is reconciled with the claimed seven-action proposal."
    ),
)

OPEN_PI_ZERO_CONTRACT = PolicyContract(
    name="OpenPiZero",
    required_observation_history=1,
    requires_proprio=True,
    native_proposal_horizon=4,
    certified_execute_prefix=None,
    temporal_ensembling=False,
    preprocessing="Pinned official PaliGemma/OpenPiZero image preprocessing.",
    normalization="Native action_normalization_type='bounds'; not a universal adapter setting.",
    reset_rule="Refresh proprioception at every native feedback boundary.",
    rng_rule="Pinned Torch and wrapper RNG streams required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation="Checkpoint state coverage, PaliGemma provenance, and wrapper fixtures remain blocking evidence.",
)

SUSIE_CONTRACT = PolicyContract(
    name="SuSIE",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=None,
    temporal_ensembling=False,
    preprocessing="Goal image plus current image; subgoal cadence must come from a pinned released wrapper.",
    normalization="Separate subgoal and low-level component normalization is unresolved.",
    reset_rule="Reset diffusion and low-level policy state under the released configuration.",
    rng_rule="Pinned JAX/Flax and diffusion RNG streams required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation=(
        "Legacy compatibility hook only. The distinct source adapter keeps the AutoEval gc_bc replication arm "
        "separate from the blocked upstream gc_ddpm_bc sensitivity arm; assets/runtime/cadence still need fixtures."
    ),
)

SUSIE_LL_CONTRACT = PolicyContract(
    name="SuSIE_LL",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=None,
    temporal_ensembling=False,
    preprocessing="Goal image plus current image under the released low-level wrapper.",
    normalization="Goal-conditioned low-level action normalization is unresolved.",
    reset_rule="Reset released low-level policy state.",
    rng_rule="Pinned JAX/Flax RNG stream required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation=(
        "Legacy compatibility hook only. The separately named low-level source adapter remains fixture-blocked "
        "until its artifacts and JAX/Flax runtime are locally verified."
    ),
)


class MiniVLAPolicy(ExternalPolicyHook):
    """Legacy injection-only hook; use ``MiniVLAPolicyAdapter`` for source loading."""
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(MINIVLA_CONTRACT, profile, loader_hook=loader_hook, source_urls=(MINIVLA_SOURCE, AUTOEVAL_POLICY_SOURCE))


class OpenPiZeroPolicy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(OPEN_PI_ZERO_CONTRACT, profile, loader_hook=loader_hook, source_urls=(OPEN_PI_ZERO_SOURCE, AUTOEVAL_POLICY_SOURCE))


class SuSIEPolicy(ExternalPolicyHook):
    """Legacy injection-only hook; use ``SuSIEPolicyAdapter`` for source loading."""
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(SUSIE_CONTRACT, profile, loader_hook=loader_hook, source_urls=(SUSIE_SOURCE, AUTOEVAL_POLICY_SOURCE))


class SuSIELowLevelPolicy(ExternalPolicyHook):
    """Legacy injection-only hook; use ``SuSIELowLevelPolicyAdapter`` for source loading."""
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(SUSIE_LL_CONTRACT, profile, loader_hook=loader_hook, source_urls=(SUSIE_SOURCE, AUTOEVAL_POLICY_SOURCE))
