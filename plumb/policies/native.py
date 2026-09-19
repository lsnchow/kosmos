"""Declared native contracts for the five non-OpenVLA benchmark policies.

This module holds the *uncertified* baseline contract for each policy: the
native observation history, native proposal horizon, temporal-ensembling flag,
and normalization family that its released wrapper declares.  Every one of them
keeps ``certified_execute_prefix=None`` and ``CapabilityStatus.BLOCKED``,
because the spec is explicit that the executed prefix is unresolved until a
golden action fixture is replayed against the pinned wrapper.

The real loaders live beside this module in :mod:`plumb.policies.octo`,
:mod:`plumb.policies.minivla`, :mod:`plumb.policies.openpizero`, and
:mod:`plumb.policies.susie`.  They attach a :class:`PolicyCertification` to the
contracts below; nothing else may change ``certified_execute_prefix``.

:class:`ExternalPolicyHook` subclasses remain for the case where an operator
owns a reviewed third-party wrapper in its own isolated image and wants to
attach it without PLUMB's loader.  They still never synthesize an action.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from plumb.adapters.contracts import CapabilityStatus, PolicyContract

from .contracts import (
    AUTOEVAL_POLICY_SOURCE_COMMIT,
    OPEN_PI_ZERO_SOURCE_COMMIT,
    ExternalPolicyHook,
    ExternalPolicyProfile,
    NativePolicyCallable,
)


AUTOEVAL_POLICY_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/auto_eval/robot/policy.py"
)
AUTOEVAL_EVAL_CONFIG_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/scripts/configs/eval_config.py"
)
OPEN_PI_ZERO_SOURCE = (
    "https://github.com/allenzren/open-pi-zero/tree/" + OPEN_PI_ZERO_SOURCE_COMMIT
)
MINIVLA_SOURCE = "https://github.com/Stanford-ILIAD/openvla-mini"
SUSIE_SOURCE = "https://github.com/kvablack/susie"
OCTO_SOURCE = "https://github.com/octo-models/octo"


OCTO_SMALL_V1_CONTRACT = PolicyContract(
    name="Octo-Small v1.0",
    required_observation_history=2,
    requires_proprio=False,
    native_proposal_horizon=4,
    certified_execute_prefix=None,
    temporal_ensembling=True,
    preprocessing="Released Octo v1.0 Bridge observation wrapper; do not substitute octo-small-1.5.",
    normalization="Native Octo action statistics; fixture verification required.",
    reset_rule="Reset two-image history using released wrapper semantics.",
    rng_rule="Pinned JAX/Octo RNG stream required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation="Native temporal ensembling and executed prefix are unresolved until wrapper fixtures pass.",
)

OCTO_BASE_V1_CONTRACT = PolicyContract(
    name="Octo-Base v1.0 diagnostic",
    required_observation_history=2,
    requires_proprio=False,
    native_proposal_horizon=4,
    certified_execute_prefix=None,
    temporal_ensembling=True,
    preprocessing="Released Octo v1.0 Bridge observation wrapper; diagnostic only, not a primary-matrix policy.",
    normalization="Native Octo action statistics; fixture verification required.",
    reset_rule="Reset two-image history using released wrapper semantics.",
    rng_rule="Pinned JAX/Octo RNG stream required.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation="Diagnostic contract only; certified prefix requires its own golden action fixture.",
)

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
    limitation="Action chunk is declared, executed prefix is not certified; pretrain_vq license is unresolved.",
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
    limitation="Requires separate subgoal/low-level models and fixture-tested cadence; AutoEval gc_bc arm differs from upstream.",
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
    limitation="Executes the low-level goal-conditioned policy directly; exact native execution remains fixture-blocked.",
)


NATIVE_POLICY_CONTRACTS: Mapping[str, PolicyContract] = {
    "Octo-Small v1.0": OCTO_SMALL_V1_CONTRACT,
    "Octo-Base v1.0 diagnostic": OCTO_BASE_V1_CONTRACT,
    "MiniVLA": MINIVLA_CONTRACT,
    "OpenPiZero": OPEN_PI_ZERO_CONTRACT,
    "SuSIE": SUSIE_CONTRACT,
    "SuSIE_LL": SUSIE_LL_CONTRACT,
}


def uncertified_contracts() -> Tuple[PolicyContract, ...]:
    """Every declared native contract, all of them still uncertified."""

    return tuple(NATIVE_POLICY_CONTRACTS.values())


def native_adapter_types() -> Dict[str, Any]:
    """Map contract name to its real loader class.

    Imported lazily so this module stays free of an import cycle and so the
    declarative contracts can be read without touching the loaders.
    """

    from .minivla import MiniVLAPolicyAdapter
    from .octo import OctoBaseV1PolicyAdapter, OctoSmallV1PolicyAdapter
    from .openpizero import OpenPiZeroPolicyAdapter
    from .susie import SuSIELowLevelPolicyAdapter, SuSIEPolicyAdapter

    return {
        "Octo-Small v1.0": OctoSmallV1PolicyAdapter,
        "Octo-Base v1.0 diagnostic": OctoBaseV1PolicyAdapter,
        "MiniVLA": MiniVLAPolicyAdapter,
        "OpenPiZero": OpenPiZeroPolicyAdapter,
        "SuSIE": SuSIEPolicyAdapter,
        "SuSIE_LL": SuSIELowLevelPolicyAdapter,
    }


class OctoSmallV1Policy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(OCTO_SMALL_V1_CONTRACT, profile, loader_hook=loader_hook, source_urls=(OCTO_SOURCE, AUTOEVAL_POLICY_SOURCE))


class OctoBaseV1Policy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(OCTO_BASE_V1_CONTRACT, profile, loader_hook=loader_hook, source_urls=(OCTO_SOURCE, AUTOEVAL_POLICY_SOURCE))


class MiniVLAPolicy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(MINIVLA_CONTRACT, profile, loader_hook=loader_hook, source_urls=(MINIVLA_SOURCE, AUTOEVAL_POLICY_SOURCE))


class OpenPiZeroPolicy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(OPEN_PI_ZERO_CONTRACT, profile, loader_hook=loader_hook, source_urls=(OPEN_PI_ZERO_SOURCE, AUTOEVAL_POLICY_SOURCE))


class SuSIEPolicy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(SUSIE_CONTRACT, profile, loader_hook=loader_hook, source_urls=(SUSIE_SOURCE, AUTOEVAL_POLICY_SOURCE))


class SuSIELowLevelPolicy(ExternalPolicyHook):
    def __init__(self, profile: ExternalPolicyProfile, *, loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None) -> None:
        super().__init__(SUSIE_LL_CONTRACT, profile, loader_hook=loader_hook, source_urls=(SUSIE_SOURCE, AUTOEVAL_POLICY_SOURCE))
