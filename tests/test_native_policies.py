"""Native multi-action policy adapter contracts, using injected fakes only.

No torch, no jax, no transformers, no network, no checkpoints.  Every adapter is
exercised through its dependency-injection seams so the whole native contract —
horizon, history, staleness, proprio freshness, normalizer application, gripper
units, certification gating, and resume — is testable on a CPU planning host.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from plumb.adapters.bridge import BridgeState
from plumb.adapters.contracts import (
    CapabilityStatus,
    FeedbackMode,
    PolicyObservation,
    ServerTiming,
    WorldResult,
)
from plumb.policies.contracts import (
    BRIDGE_ACTION_GRIPPER_HIGH,
    BRIDGE_STATE_GRIPPER_OPEN_LIMIT,
    UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    ActionNormalizationType,
    GripperActionValue,
    GripperPolarityConvention,
    GripperStateValue,
    NativeWrapperEntryPoint,
    PolicyActionNormalizer,
    PolicyActionNormalizerStatistics,
    PolicyCertification,
    PolicyCertificationError,
    PolicyContractError,
    PolicyExecutionMode,
    mint_certification_payload,
)
from plumb.policies.minivla import (
    ConvertedWeightArtifact,
    MiniVLAPolicyAdapter,
    MiniVLAPolicyProfile,
    MiniVLAUnavailableError,
    _MiniVLARuntime,
)
from plumb.policies.native import OPEN_PI_ZERO_SOURCE_COMMIT, uncertified_contracts
from plumb.policies.octo import (
    OCTO_BASE_CHECKPOINT_STEP,
    OCTO_BASE_MODEL_ID,
    OCTO_SMALL_CHECKPOINT_STEP,
    OCTO_SMALL_MODEL_ID,
    OctoBaseV1PolicyAdapter,
    OctoEnsembleConfig,
    OctoPolicyProfile,
    OctoSmallV1PolicyAdapter,
    OctoTemporalEnsembler,
    _OctoRuntime,
)
from plumb.policies.openpizero import (
    BRIDGE_7D_DROP_STRUCTURAL_ZERO_LAYOUT,
    BRIDGE_8D_PASSTHROUGH_LAYOUT,
    PALIGEMMA_CANDIDATE_MIRROR,
    PALIGEMMA_REQUIRED_FILES,
    OpenPiZeroPolicyAdapter,
    OpenPiZeroPolicyProfile,
    PaliGemmaSupportFiles,
    _OpenPiZeroRuntime,
)
from plumb.policies.susie import (
    SUSIE_AUTOEVAL_RELEASED_ARM,
    SUSIE_UPSTREAM_CORRECTED_ARM,
    SuSIEArmConfiguration,
    SuSIELowLevelPolicyAdapter,
    SuSIEPolicyAdapter,
    SuSIEPolicyProfile,
    _SuSIERuntime,
)


REVISION = "a" * 40
DIGEST = "sha256:" + "c" * 64
JAX_PIN = "0.4.20"


# --------------------------------------------------------------------------- fakes


class FakeFrame:
    """A frame with stable pixel bytes and a Pillow-compatible ``convert``."""

    def __init__(self, tag: bytes) -> None:
        self.tag = tag
        self.converted_to = None

    def tobytes(self) -> bytes:
        return self.tag

    @property
    def shape(self):
        return (2, 2, 3)

    @property
    def dtype(self):
        return "uint8"

    def convert(self, mode):
        self.converted_to = mode
        return self


def frames(*tags: bytes):
    return tuple(FakeFrame(tag) for tag in tags)


class FakeOctoModel:
    def __init__(self, chunks=None):
        self.chunks = list(chunks) if chunks is not None else None
        self.task_calls = []
        self.sample_calls = []

    def create_tasks(self, texts):
        self.task_calls.append(tuple(texts))
        return {"language_instruction": tuple(texts)}

    def sample_actions(self, observations, tasks, unnormalization_statistics=None, rng=None):
        self.sample_calls.append(
            {
                "observations": observations,
                "tasks": tasks,
                "unnormalization_statistics": unnormalization_statistics,
                "rng": rng,
            }
        )
        if self.chunks:
            chunk = self.chunks.pop(0)
        else:
            index = len(self.sample_calls)
            chunk = [[0.01 * index * (row + 1)] * 7 for row in range(4)]
        return [chunk]


def octo_runtime(version=JAX_PIN):
    return _OctoRuntime(
        jax_version=version,
        array_factory=lambda value, dtype: {"dtype": dtype, "value": value},
        prng_key=lambda seed: ("prng", seed),
        fold_in=lambda key, step: (key, "fold", step),
        model_loader=lambda path, step: pytest.fail("real Octo loader must not run in tests"),
    )


def octo_normalizer(revision="octo-bridge-stats-v1"):
    return PolicyActionNormalizer(
        PolicyActionNormalizerStatistics(
            normalizer_revision=revision,
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 6 + (False,),
            mean=(0.002, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0),
            std=(0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 1.0),
            source_uri="fixture://octo/dataset_statistics.json",
            statistics_sha256=DIGEST,
        )
    )


def octo_profile(**changes):
    values = {
        "profile_id": "octo-small-fixture",
        "local_model_path": "/does/not/exist",
        "model_id": OCTO_SMALL_MODEL_ID,
        "checkpoint_revision": REVISION,
        "checkpoint_step": OCTO_SMALL_CHECKPOINT_STEP,
    }
    values.update(changes)
    return OctoPolicyProfile(**values)


def octo_ensemble(exponential_weight=0.0):
    return OctoEnsembleConfig(
        config_revision="octo-temporal-ensemble-v1", exponential_weight=exponential_weight, horizon=4
    )


def octo_adapter(
    *,
    certification=None,
    mode=PolicyExecutionMode.CERTIFIED,
    model=None,
    profile=None,
    normalizer=None,
    exponential_weight=0.0,
    runtime=None,
    cls=OctoSmallV1PolicyAdapter,
):
    model = model if model is not None else FakeOctoModel()
    adapter = cls(
        profile if profile is not None else octo_profile(),
        normalizer=normalizer if normalizer is not None else octo_normalizer(),
        ensemble=octo_ensemble(exponential_weight),
        certification=certification,
        execution_mode=mode,
        runtime_factory=runtime if runtime is not None else octo_runtime,
        model_factory=lambda profile, runtime: model,
    )
    adapter.fake_model = model
    return adapter


class FakeMiniVLAModel:
    def __init__(self, rows=None):
        self.rows = rows
        self.calls = []

    def eval(self):
        return self

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        if self.rows is not None:
            return self.rows
        return [[0.01 * (row + 1)] * 7 for row in range(7)]


def minivla_runtime(version="4.40.1"):
    return _MiniVLARuntime(
        torch=FakeTorch,
        transformers_version=version,
        image_fromarray=FakeFrame,
        model_loader=lambda profile: pytest.fail("real MiniVLA loader must not run in tests"),
    )


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext

        return nullcontext()


def converted_artifact(label, path, **changes):
    values = {
        "label": label,
        "path": path,
        "sha256": DIGEST,
        "source_pickle_sha256": DIGEST,
        "conversion_report_sha256": DIGEST,
        "license": "MIT",
        "license_status": "advertised_unverified",
        "redistribution": "local_use_only",
    }
    values.update(changes)
    return ConvertedWeightArtifact(**values)


def minivla_profile(tmp_path, **changes):
    checkpoint = converted_artifact(
        "minivla-vq-bridge-prismatic checkpoint",
        str(tmp_path / "minivla.safetensors"),
        license=None,
        license_status="unresolved",
        redistribution="prohibited_pending_resolution",
    )
    vq = converted_artifact(
        "Stanford-ILIAD/pretrain_vq tokenizer",
        str(tmp_path / "pretrain_vq.safetensors"),
        license=None,
        license_status="absent_cardData_null",
        redistribution="prohibited_pending_resolution",
    )
    values = {
        "profile_id": "minivla-fixture",
        "converted_checkpoint": checkpoint,
        "converted_vq": vq,
        "checkpoint_revision": REVISION,
        "vq_revision": "b" * 40,
        "loader_revision": "d" * 40,
        "vq_license_acknowledged": True,
    }
    values.update(changes)
    return MiniVLAPolicyProfile(**values)


def minivla_normalizer(revision="minivla-bridge-stats-v1"):
    return PolicyActionNormalizer(
        PolicyActionNormalizerStatistics(
            normalizer_revision=revision,
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 6 + (False,),
            mean=(0.0,) * 6 + (0.0,),
            std=(0.015, 0.015, 0.015, 0.08, 0.08, 0.08, 1.0),
            statistics_sha256=DIGEST,
        )
    )


def minivla_adapter(tmp_path, *, certification=None, mode=PolicyExecutionMode.CERTIFIED, model=None, profile=None):
    model = model if model is not None else FakeMiniVLAModel()
    adapter = MiniVLAPolicyAdapter(
        profile if profile is not None else minivla_profile(tmp_path),
        normalizer=minivla_normalizer(),
        certification=certification,
        execution_mode=mode,
        runtime_factory=minivla_runtime,
        model_factory=lambda profile, runtime: model,
    )
    adapter.fake_model = model
    return adapter


class FakeOpenPiZeroWrapper:
    def __init__(self, rows=None):
        self.rows = rows
        self.calls = []

    def eval(self):
        return self

    def infer_action(self, *, image, text, proprio):
        self.calls.append({"image": image, "text": text, "proprio": tuple(proprio)})
        if self.rows is not None:
            return self.rows
        return [[0.1 * (row + 1)] * 7 for row in range(4)]


def openpizero_runtime():
    return _OpenPiZeroRuntime(
        torch=FakeTorch,
        torch_version="2.4.0",
        image_fromarray=FakeFrame,
        wrapper_factory=lambda profile: pytest.fail("real OpenPiZero loader must not run in tests"),
    )


def openpizero_normalizer(revision="open-pi-zero-bounds-v1", clip=True):
    return PolicyActionNormalizer(
        PolicyActionNormalizerStatistics(
            normalizer_revision=revision,
            normalization_type=ActionNormalizationType.BOUNDS,
            mask=(True,) * 6 + (False,),
            low=(-0.015, -0.015, -0.015, -0.08, -0.08, -0.08, 0.0),
            high=(0.015, 0.015, 0.015, 0.08, 0.08, 0.08, BRIDGE_ACTION_GRIPPER_HIGH),
            clip_normalized_to_unit=clip,
            statistics_sha256=DIGEST,
        )
    )


def paligemma(tmp_path, *, repo_id=None, create=True, **changes):
    root = tmp_path / "paligemma"
    if create:
        root.mkdir(parents=True, exist_ok=True)
        for name in PALIGEMMA_REQUIRED_FILES:
            (root / name).write_text("{}")
    values = {"local_path": str(root), "revision": REVISION, "terms_accepted": True}
    if repo_id is not None:
        values["repo_id"] = repo_id
    values.update(changes)
    return PaliGemmaSupportFiles(**values)


def openpizero_profile(tmp_path, **changes):
    values = {
        "profile_id": "open-pi-zero-fixture",
        "converted_checkpoint_path": str(tmp_path / "open-pi-zero.safetensors"),
        "converted_checkpoint_sha256": DIGEST,
        "source_pickle_sha256": DIGEST,
        "conversion_report_sha256": DIGEST,
        "checkpoint_revision": REVISION,
        "paligemma": paligemma(tmp_path),
        "proprio_layout": BRIDGE_8D_PASSTHROUGH_LAYOUT,
        "entry_point": NativeWrapperEntryPoint(
            module="open_pi_zero_fake",
            attribute="build_bridge_wrapper",
            loader_revision=OPEN_PI_ZERO_SOURCE_COMMIT,
        ),
        "application_cache_dir": str(tmp_path / "cache"),
        "state_dict_coverage_verified": True,
    }
    values.update(changes)
    return OpenPiZeroPolicyProfile(**values)


def openpizero_adapter(
    tmp_path, *, certification=None, mode=PolicyExecutionMode.CERTIFIED, wrapper=None, profile=None, normalizer=None
):
    wrapper = wrapper if wrapper is not None else FakeOpenPiZeroWrapper()
    adapter = OpenPiZeroPolicyAdapter(
        profile if profile is not None else openpizero_profile(tmp_path),
        normalizer=normalizer if normalizer is not None else openpizero_normalizer(),
        certification=certification,
        execution_mode=mode,
        runtime_factory=openpizero_runtime,
        wrapper_factory=lambda profile, runtime: wrapper,
    )
    adapter.fake_wrapper = wrapper
    return adapter


PROPRIO = (0.31, -0.02, 0.07, 0.01, -0.02, 0.03, 0.0, 0.2)


class FakeSubgoalModel:
    def __init__(self):
        self.calls = []

    def sample_subgoal(self, *, image, goal_image, prompt, rng):
        self.calls.append({"image": image, "goal_image": goal_image, "prompt": prompt, "rng": rng})
        return FakeFrame(b"subgoal-%d" % len(self.calls))


class FakeLowLevelModel:
    def __init__(self, rows=None):
        self.rows = rows
        self.calls = []

    def sample_actions(self, *, image, goal_image, rng):
        self.calls.append({"image": image, "goal_image": goal_image, "rng": rng})
        return self.rows if self.rows is not None else [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]


def susie_runtime(version=JAX_PIN):
    return _SuSIERuntime(
        jax_version=version,
        prng_key=lambda seed: ("prng", seed),
        fold_in=lambda key, value: (key, "fold", value),
        low_level_factory=lambda profile: pytest.fail("real SuSIE loader must not run in tests"),
        subgoal_factory=None,
    )


def susie_normalizer(revision="susie-gcbc-stats-v1"):
    return PolicyActionNormalizer(
        PolicyActionNormalizerStatistics(
            normalizer_revision=revision,
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 6 + (False,),
            mean=(0.0,) * 7,
            std=(0.015, 0.015, 0.015, 0.08, 0.08, 0.08, 1.0),
            statistics_sha256=DIGEST,
        )
    )


def susie_profile(tmp_path, *, subgoal=True, **changes):
    values = {
        "profile_id": "susie-fixture",
        "low_level_model_path": str(tmp_path / "gcbc-bridge"),
        "low_level_revision": REVISION,
        "arm": SUSIE_AUTOEVAL_RELEASED_ARM,
        "low_level_entry_point": NativeWrapperEntryPoint(
            module="susie_fake", attribute="build_low_level", loader_revision="e" * 40
        ),
        "goal_image_source": "scenario-goal-panel-v1",
        "jax_version": JAX_PIN,
        "low_level_license": "MIT",
    }
    if subgoal:
        values.update(
            {
                "subgoal_model_path": str(tmp_path / "susie"),
                "subgoal_revision": "f" * 40,
                "subgoal_entry_point": NativeWrapperEntryPoint(
                    module="susie_fake", attribute="build_subgoal", loader_revision="e" * 40
                ),
                "subgoal_cadence": 3,
            }
        )
    values.update(changes)
    return SuSIEPolicyProfile(**values)


def susie_adapter(
    tmp_path,
    *,
    certification=None,
    mode=PolicyExecutionMode.CERTIFIED,
    subgoal_model=None,
    low_level=None,
    profile=None,
    cls=SuSIEPolicyAdapter,
):
    wants_subgoal = cls is SuSIEPolicyAdapter
    subgoal_model = subgoal_model if subgoal_model is not None else (FakeSubgoalModel() if wants_subgoal else None)
    low_level = low_level if low_level is not None else FakeLowLevelModel()
    adapter = cls(
        profile if profile is not None else susie_profile(tmp_path, subgoal=wants_subgoal),
        normalizer=susie_normalizer(),
        certification=certification,
        execution_mode=mode,
        runtime_factory=susie_runtime,
        low_level_factory=lambda profile, runtime: low_level,
        subgoal_factory=(None if not wants_subgoal else (lambda profile, runtime: subgoal_model)),
    )
    adapter.fake_subgoal = subgoal_model
    adapter.fake_low_level = low_level
    return adapter


def mint(adapter, observation, tmp_path, *, prefix, ensembling, name="certification.json", **changes):
    """Mint a golden fixture from one real native call and load it back."""

    proposal = adapter.propose(observation)
    payload = mint_certification_payload(
        proposal,
        certified_execute_prefix=prefix,
        temporal_ensembling=ensembling,
        recorded_at="2026-09-19T00:00:00Z",
        evidence_uris=("fixture://golden-action",),
        **changes,
    )
    path = Path(tmp_path) / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PolicyCertification.from_json_file(path)


# ------------------------------------------------------- normalizer and gripper units


def test_bounds_normalizer_is_invertible_and_applied_exactly_once():
    normalizer = openpizero_normalizer()
    normalized = (0.5, -0.25, 0.0, 1.0, -1.0, 0.125, 0.4)

    physical = normalizer.denormalize(normalized)
    recovered = normalizer.normalize(physical)

    assert physical[0] == pytest.approx(0.015 * 0.5)
    assert physical[3] == pytest.approx(0.08)
    assert physical[4] == pytest.approx(-0.08)
    # The gripper channel is unmasked, so it passes through in its own units.
    assert physical[6] == pytest.approx(0.4)
    assert recovered == pytest.approx(normalized)
    assert normalizer.counters() == {"denormalize_calls": 1, "normalize_calls": 1, "clipped_components": 0}
    # A doubled affine transform is detectable: it is not idempotent.
    assert normalizer.denormalize(physical) != pytest.approx(physical)


def test_normal_normalizer_is_invertible_and_leaves_masked_channels_untouched():
    normalizer = octo_normalizer()
    normalized = (1.0, -2.0, 0.5, 0.25, -0.75, 0.125, 0.996)

    physical = normalizer.denormalize(normalized)
    assert physical[0] == pytest.approx(1.0 * 0.01 + 0.002)
    assert physical[2] == pytest.approx(0.5 * 0.01 - 0.001)
    assert physical[6] == pytest.approx(0.996)
    assert normalizer.normalize(physical) == pytest.approx(normalized)


def test_bounds_clipping_is_counted_and_is_not_a_normal_family_behaviour():
    normalizer = openpizero_normalizer()
    physical = normalizer.denormalize((4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.5))

    assert physical[0] == pytest.approx(0.015)
    assert physical[1] == pytest.approx(-0.015)
    assert normalizer.counters()["clipped_components"] == 2

    unclipped = PolicyActionNormalizer(
        replace(normalizer.statistics, clip_normalized_to_unit=False, normalizer_revision="opz-bounds-noclip")
    )
    # Without clipping, an out-of-range normalized value escapes the trained
    # bounds entirely (0.5 * (4 + 1) * 0.03 - 0.015), which is exactly why the
    # clipping setting has to be declared per adapter rather than assumed.
    assert unclipped.denormalize((4.0,) + (0.0,) * 6)[0] == pytest.approx(0.06)
    assert unclipped.counters()["clipped_components"] == 0
    with pytest.raises(PolicyContractError, match="bounds-normalization behaviour"):
        PolicyActionNormalizerStatistics(
            normalizer_revision="bad",
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 7,
            mean=(0.0,) * 7,
            std=(1.0,) * 7,
            clip_normalized_to_unit=True,
        )


def test_normalizer_statistics_refuse_mixed_families_widths_and_unrevisioned_use():
    with pytest.raises(PolicyContractError, match="normalizer_revision is required"):
        PolicyActionNormalizerStatistics(
            normalizer_revision="",
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 7,
            mean=(0.0,) * 7,
            std=(1.0,) * 7,
        )
    with pytest.raises(PolicyContractError, match="must not also declare mean/std"):
        PolicyActionNormalizerStatistics(
            normalizer_revision="mixed",
            normalization_type=ActionNormalizationType.BOUNDS,
            mask=(True,) * 7,
            low=(-1.0,) * 7,
            high=(1.0,) * 7,
            mean=(0.0,) * 7,
            std=(1.0,) * 7,
        )
    with pytest.raises(PolicyContractError, match="seven channels"):
        PolicyActionNormalizerStatistics(
            normalizer_revision="short",
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 7,
            mean=(0.0,) * 6,
            std=(1.0,) * 6,
        )
    with pytest.raises(PolicyContractError, match="std > 0"):
        PolicyActionNormalizerStatistics(
            normalizer_revision="degenerate",
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 7,
            mean=(0.0,) * 7,
            std=(1.0,) * 6 + (0.0,),
        )
    normalizer = octo_normalizer()
    with pytest.raises(PolicyContractError, match="exactly seven channels"):
        normalizer.denormalize((0.0,) * 6)


def test_state_and_action_gripper_units_are_distinct_with_closed_at_zero_state_polarity():
    closed = GripperStateValue(0.0)
    open_limit = GripperStateValue(BRIDGE_STATE_GRIPPER_OPEN_LIMIT)

    assert closed.is_fully_closed is True
    assert closed.open_fraction() == pytest.approx(0.0)
    assert open_limit.is_fully_closed is False
    assert open_limit.open_fraction() == pytest.approx(1.0)
    # An action gripper value is outside the state range and must not be accepted
    # as one: state units and action gripper units are not interchangeable.
    with pytest.raises(PolicyContractError, match="is not a state value"):
        GripperStateValue(0.996)
    with pytest.raises(PolicyContractError, match="outside the observed"):
        GripperActionValue(1.5)
    assert GripperActionValue(BRIDGE_ACTION_GRIPPER_HIGH).value == pytest.approx(0.996)

    resolved = GripperPolarityConvention(
        convention_id="fixture-high-is-open", source_revision=REVISION, action_high_means_open=True
    )
    inverted = replace(resolved, convention_id="fixture-high-is-closed", action_high_means_open=False)
    assert resolved.action_open_fraction(GripperActionValue(0.996)) == pytest.approx(1.0)
    assert inverted.action_open_fraction(GripperActionValue(0.996)) == pytest.approx(0.0)
    assert resolved.action_from_state(closed).value == pytest.approx(0.0)
    assert inverted.action_from_state(closed).value == pytest.approx(0.996)
    assert resolved.state_from_action(GripperActionValue(0.0)).is_fully_closed is True
    with pytest.raises(PolicyContractError, match="not interchangeable"):
        resolved.action_from_state(GripperActionValue(0.5))
    with pytest.raises(PolicyContractError, match="not interchangeable"):
        resolved.action_open_fraction(GripperStateValue(0.2))


def test_unresolved_gripper_convention_refuses_every_conversion():
    convention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION
    assert convention.resolved is False
    with pytest.raises(PolicyContractError, match="has not resolved whether a high action value means open"):
        convention.action_open_fraction(GripperActionValue(0.5))
    with pytest.raises(PolicyContractError, match="has not resolved"):
        convention.action_from_state(GripperStateValue(0.2))
    with pytest.raises(PolicyContractError, match="has not resolved"):
        convention.state_from_action(GripperActionValue(0.5))


# ------------------------------------------------------------------- certification


def test_certification_requires_its_own_artifact_hash_and_a_matching_body(tmp_path):
    adapter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    adapter.reset(11)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(adapter, golden, tmp_path, prefix=1, ensembling=True)

    assert certification.errors() == ()
    assert certification.fixture_sha256 is not None
    assert certification.certification_sha256 == certification.canonical_sha256()

    tampered = replace(certification, expected_actions=((0.0,) * 7,) * 4)
    assert "certification_sha256 does not match the canonical certification body" in tampered.errors()
    unhashed = replace(certification, fixture_sha256=None)
    assert any("fixture_sha256" in problem for problem in unhashed.errors())

    payload = json.loads((tmp_path / "certification.json").read_text(encoding="utf-8"))
    payload.pop("certification_sha256")
    orphan = tmp_path / "orphan.json"
    orphan.write_text(json.dumps(payload), encoding="utf-8")
    assert any("certification_sha256 is required" in problem for problem in PolicyCertification.from_json_file(orphan).errors())

    bad_kind = tmp_path / "bad.json"
    bad_kind.write_text(json.dumps({"kind": "something_else", "schema_version": 1}), encoding="utf-8")
    with pytest.raises(PolicyCertificationError, match="kind must be"):
        PolicyCertification.from_json_file(bad_kind)


def test_certification_refuses_a_multi_step_prefix_under_temporal_ensembling(tmp_path):
    adapter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    adapter.reset(11)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(adapter, golden, tmp_path, prefix=4, ensembling=True)

    assert any("execute prefix must be 1" in problem for problem in certification.errors())

    honest = mint(adapter, replace(golden, image_history=frames(b"g1", b"g1"), timestamp=1.0),
                  tmp_path, prefix=1, ensembling=True, name="honest.json")
    assert honest.errors() == ()


def test_certification_binds_the_golden_observation_normalizer_and_source_revision(tmp_path):
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(3)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=1, ensembling=True)

    adapter = octo_adapter(certification=certification)
    adapter.reset(3)
    assert adapter.certify(golden).passed is True

    other = octo_adapter(certification=certification)
    other.reset(3)
    check = other.certify(replace(golden, image_history=frames(b"different", b"different")))
    assert check.passed is False
    assert "golden_observation_hash_mismatch" in check.reasons

    mismatched = octo_adapter(certification=certification, normalizer=octo_normalizer("other-revision"))
    assert any("normalizer_revision_mismatch" in problem for problem in mismatched.certification_errors())
    assert mismatched.capability().status is CapabilityStatus.BLOCKED

    wrong_policy = OctoBaseV1PolicyAdapter(
        octo_profile(model_id=OCTO_BASE_MODEL_ID, checkpoint_step=OCTO_BASE_CHECKPOINT_STEP),
        normalizer=octo_normalizer(),
        ensemble=octo_ensemble(),
        certification=certification,
        runtime_factory=octo_runtime,
        model_factory=lambda profile, runtime: FakeOctoModel(),
    )
    assert any("policy_name_mismatch" in problem for problem in wrong_policy.certification_errors())


def test_minting_refuses_an_unstable_observation_identity_and_an_unpinned_revision(tmp_path):
    adapter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    adapter.reset(5)
    opaque = object()
    proposal = adapter.propose(
        PolicyObservation(image_history=(opaque, opaque), prompt="Open the drawer", timestamp=0.0)
    )
    assert proposal.observation_identity_stable is False
    with pytest.raises(PolicyCertificationError, match="no stable pixel identity"):
        mint_certification_payload(proposal, certified_execute_prefix=1, temporal_ensembling=True)

    adapter.reset(5)
    stable = adapter.propose(
        PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    )
    with pytest.raises(PolicyCertificationError, match="source revision"):
        mint_certification_payload(
            replace(stable, source_revision=None), certified_execute_prefix=1, temporal_ensembling=True
        )


# ---------------------------------------------------------------------------- Octo


def test_octo_uses_two_image_history_four_action_proposal_and_one_ensembled_execution(tmp_path):
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(21)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=1, ensembling=True)

    model = FakeOctoModel(chunks=[[[0.0] * 7, [1.0] * 7, [2.0] * 7, [3.0] * 7]])
    adapter = octo_adapter(certification=certification, model=model)
    adapter.reset(21)
    plan = adapter.plan_control(
        PolicyObservation(image_history=frames(b"f0", b"f0"), prompt="Open the drawer", timestamp=0.0), 70
    )

    assert adapter.contract.required_observation_history == 2
    assert adapter.contract.native_proposal_horizon == 4
    assert adapter.contract.temporal_ensembling is True
    assert adapter.contract.certified_execute_prefix == 1
    # The plan exposes only the ensembled row: the un-ensembled tail of a
    # temporally ensembled proposal is never executable.
    assert len(plan.actions) == 1
    assert plan.verified_prefix == 1
    assert plan.history_length == 2
    assert plan.metadata["native_proposal_rows"] == 4
    assert plan.metadata["temporal_ensembling"] is True
    assert plan.metadata["qualified"] is False
    assert len(model.sample_calls) == 1
    call = model.sample_calls[0]
    # PLUMB owns denormalization, so the native sampler is asked for normalized
    # actions and the revisioned normalizer is applied exactly once per row.
    assert call["unnormalization_statistics"] is None
    assert call["observations"]["image_primary"]["dtype"] == "uint8"
    assert call["observations"]["pad_mask"]["value"] == [[False, True]]
    assert model.task_calls == [("Open the drawer",)]
    assert adapter.normalizer.counters()["denormalize_calls"] == 4


def test_octo_temporal_ensembler_matches_the_released_ordering(tmp_path):
    uniform = OctoTemporalEnsembler(octo_ensemble(0.0))
    first = tuple((float(index),) * 7 for index in (10.0, 20.0, 30.0, 40.0))
    second = tuple((float(index),) * 7 for index in (50.0, 60.0, 70.0, 80.0))

    assert uniform.ensemble(first)[0] == pytest.approx(10.0)
    # Depth two pairs the older prediction's index 1 with the newer index 0 and,
    # at exp_weight=0, averages them uniformly.
    assert uniform.ensemble(second)[0] == pytest.approx((20.0 + 50.0) / 2.0)
    assert uniform.depth == 2

    weighted = OctoTemporalEnsembler(octo_ensemble(1.0))
    weighted.ensemble(first)
    blended = weighted.ensemble(second)[0]
    # The released wrapper gives more recent predictions exponentially *less*
    # weight, so the older prediction dominates.
    assert blended < (20.0 + 50.0) / 2.0
    assert blended == pytest.approx((1.0 * 20.0 + (2.718281828459045 ** -1.0) * 50.0) / (1.0 + 2.718281828459045 ** -1.0))

    deep = OctoTemporalEnsembler(octo_ensemble(0.0))
    for _ in range(6):
        deep.ensemble(first)
    assert deep.depth == 4
    with pytest.raises(PolicyContractError, match="will not pad the chunk"):
        deep.ensemble(first[:2])


def test_octo_refuses_the_1_5_line_the_wrong_step_and_a_root_checkpoint_path(tmp_path):
    blocked = octo_adapter(profile=octo_profile(model_id="rail-berkeley/octo-small-1.5"))
    result = blocked.capability()
    assert result.status is CapabilityStatus.BLOCKED
    assert "octo-small-1.5 is explicitly excluded" in result.reason

    wrong_step = octo_adapter(profile=octo_profile(checkpoint_step=300000))
    assert "pins checkpoint step 270000" in wrong_step.capability().reason

    root = tmp_path / "octo-small"
    root.mkdir()
    absent = OctoSmallV1PolicyAdapter(
        octo_profile(local_model_path=str(root)),
        normalizer=octo_normalizer(),
        ensemble=octo_ensemble(),
        certification=None,
        runtime_factory=octo_runtime,
    )
    # Uncertified adapters report BLOCKED first; the checkpoint-path trap shows
    # up in the availability check.
    assert absent.capability().status is CapabilityStatus.BLOCKED
    assert absent.profile.checkpoint_relative_path == "270000/default/checkpoint"
    assert "not at the repository root" in (absent._availability_error() or "")
    (root / "270000" / "default").mkdir(parents=True)
    (root / "270000" / "default" / "checkpoint").write_text("weights")
    assert absent._availability_error() is None


def test_octo_rejects_a_stale_repeated_frame_a_wrong_history_and_a_non_advancing_tick(tmp_path):
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(4)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=1, ensembling=True)
    adapter = octo_adapter(certification=certification)
    adapter.reset(4)

    first = FakeFrame(b"f0")
    adapter.plan_control(
        PolicyObservation(image_history=(first, first), prompt="Open the drawer", timestamp=0.0), 70
    )
    with pytest.raises(PolicyContractError, match="requerying a policy against a stale image"):
        adapter.plan_control(
            PolicyObservation(image_history=(first, first), prompt="Open the drawer", timestamp=0.2), 69
        )
    with pytest.raises(PolicyContractError, match="exactly 2 observation frames"):
        adapter.plan_control(
            PolicyObservation(image_history=frames(b"f1"), prompt="Open the drawer", timestamp=0.4), 69
        )
    with pytest.raises(PolicyContractError, match="does not advance past"):
        adapter.plan_control(
            PolicyObservation(image_history=frames(b"f0", b"f1"), prompt="Open the drawer", timestamp=0.0), 69
        )
    with pytest.raises(PolicyContractError, match="must not receive proprioception"):
        adapter.plan_control(
            PolicyObservation(
                image_history=frames(b"f0", b"f2"), prompt="Open the drawer", proprio=PROPRIO, timestamp=0.4
            ),
            69,
        )
    with pytest.raises(PolicyContractError, match="nonempty exact task instruction"):
        adapter.plan_control(PolicyObservation(image_history=frames(b"f0", b"f3"), prompt="  ", timestamp=0.6), 69)


def test_octo_requires_a_reset_seed_and_refuses_a_wrong_jax_pin(tmp_path):
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(9)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=1, ensembling=True)

    unseeded = octo_adapter(certification=certification)
    with pytest.raises(PolicyContractError, match="requires reset\\(seed\\)"):
        unseeded.propose(PolicyObservation(image_history=frames(b"f0", b"f0"), prompt="Open the drawer", timestamp=0.0))
    with pytest.raises(PolicyContractError, match="integer seed"):
        unseeded.reset(1.5)

    from plumb.policies.octo import OctoUnavailableError

    wrong_jax = octo_adapter(certification=certification, runtime=lambda: octo_runtime("0.5.0"))
    wrong_jax.reset(9)
    with pytest.raises(OctoUnavailableError, match="JAX"):
        wrong_jax.propose(PolicyObservation(image_history=frames(b"f0", b"f0"), prompt="Open the drawer", timestamp=0.0))


def test_octo_snapshot_restores_its_ensemble_buffer_and_refuses_a_foreign_snapshot(tmp_path):
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(31)
    golden = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=1, ensembling=True)

    adapter = octo_adapter(certification=certification)
    adapter.reset(31)
    for index, tag in enumerate((b"f0", b"f1", b"f2")):
        adapter.plan_control(
            PolicyObservation(
                image_history=frames(b"prev", tag), prompt="Open the drawer", timestamp=float(index) * 0.2
            ),
            70 - index,
        )
    snapshot = json.loads(json.dumps(adapter.snapshot_state()))
    assert snapshot["step"] == 3
    assert len(snapshot["extra"]["ensembler"]["history"]) == 3

    resumed = octo_adapter(certification=certification)
    resumed.restore_state(snapshot)
    assert resumed.snapshot_state() == snapshot

    with pytest.raises(PolicyContractError, match="refusing a cross-policy resume"):
        resumed.restore_state({**snapshot, "policy": "MiniVLA"})
    with pytest.raises(PolicyContractError, match="normalizer revision"):
        resumed.restore_state({**snapshot, "normalizer_revision": "other"})
    with pytest.raises(PolicyContractError, match="Snapshot certification"):
        resumed.restore_state({**snapshot, "certification_sha256": "sha256:" + "0" * 64})
    with pytest.raises(PolicyContractError, match="Snapshot kind"):
        resumed.restore_state({"kind": "other", "schema_version": 1})


# ------------------------------------------------------------------------- MiniVLA


def test_minivla_proposes_seven_actions_from_one_fresh_image(tmp_path):
    minter = minivla_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(2)
    golden = PolicyObservation(image_history=frames(b"g0"), prompt="Close the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=7, ensembling=False)

    model = FakeMiniVLAModel()
    adapter = minivla_adapter(tmp_path, certification=certification, model=model)
    adapter.reset(2)
    plan = adapter.plan_control(
        PolicyObservation(image_history=frames(b"f0"), prompt="Close the drawer", timestamp=0.0), 70
    )

    assert adapter.contract.native_proposal_horizon == 7
    assert adapter.contract.required_observation_history == 1
    assert adapter.contract.certified_execute_prefix == 7
    assert len(plan.actions) == 7 and plan.verified_prefix == 7
    assert len(model.calls) == 1
    assert model.calls[0]["unnorm_key"] is None
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["instruction"] == "Close the drawer"
    assert model.calls[0]["image"].converted_to == "RGB"
    assert adapter.normalizer.counters()["denormalize_calls"] == 7
    # Physical rows land in the observed Bridge ranges once, not twice.
    assert plan.actions[0][0] == pytest.approx(0.01 * 0.015)
    assert plan.actions[6][3] == pytest.approx(0.07 * 0.08)


def test_minivla_refuses_a_legacy_pickle_an_unhashed_conversion_and_a_wrong_chunk(tmp_path):
    legacy = minivla_profile(
        tmp_path,
        converted_checkpoint=converted_artifact(
            "minivla checkpoint",
            str(tmp_path / "checkpoints/step-362500-epoch-21-loss=0.2259.pt"),
            license=None,
            license_status="unresolved",
            redistribution="prohibited_pending_resolution",
        ),
    )
    adapter = minivla_adapter(tmp_path, profile=legacy)
    assert "legacy .pt pickle" in adapter.capability().reason

    unhashed = minivla_profile(
        tmp_path,
        converted_vq=converted_artifact(
            "pretrain_vq",
            str(tmp_path / "vq.safetensors"),
            conversion_report_sha256=None,
            license=None,
            license_status="absent_cardData_null",
            redistribution="prohibited_pending_resolution",
        ),
    )
    assert "isolated conversion report hash" in minivla_adapter(tmp_path, profile=unhashed).capability().reason

    assert "7-action chunk" in minivla_adapter(tmp_path, profile=minivla_profile(tmp_path, action_chunk=4)).capability().reason
    assert "use_extra=True" in minivla_adapter(tmp_path, profile=minivla_profile(tmp_path, use_extra=False)).capability().reason
    assert "Transformers 4.40.1" in minivla_adapter(
        tmp_path, profile=minivla_profile(tmp_path, transformers_version="5.0.0")
    ).capability().reason
    assert "openvla-mini commit" in minivla_adapter(
        tmp_path, profile=minivla_profile(tmp_path, loader_revision=None)
    ).capability().reason

    with pytest.raises(MiniVLAUnavailableError, match="legacy .pt pickle"):
        minivla_adapter(tmp_path, profile=legacy, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION).propose(
            PolicyObservation(image_history=frames(b"f0"), prompt="Close the drawer", timestamp=0.0)
        )


def test_minivla_pretrain_vq_null_license_is_surfaced_and_must_be_acknowledged(tmp_path):
    unacknowledged = minivla_profile(tmp_path, vq_license_acknowledged=False)
    adapter = minivla_adapter(tmp_path, profile=unacknowledged)
    result = adapter.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert "declares no license" in result.reason
    assert "does not mirror or redistribute it" in result.reason
    assert result.details["vq_license"] is None
    assert result.details["vq_license_status"] == "absent_cardData_null"
    assert result.details["vq_redistribution"] == "prohibited_pending_resolution"

    fabricated = minivla_profile(
        tmp_path,
        converted_vq=converted_artifact("pretrain_vq", str(tmp_path / "vq.safetensors"), license="MIT"),
    )
    assert "would be a fabrication" in minivla_adapter(tmp_path, profile=fabricated).capability().reason


def test_minivla_reports_unavailable_only_once_its_converted_artifacts_exist(tmp_path):
    minter = minivla_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    golden = PolicyObservation(image_history=frames(b"g0"), prompt="Close the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=7, ensembling=False)

    profile = minivla_profile(tmp_path)
    adapter = MiniVLAPolicyAdapter(
        profile,
        normalizer=minivla_normalizer(),
        certification=certification,
        runtime_factory=minivla_runtime,
    )
    result = adapter.capability()
    assert result.status is CapabilityStatus.UNAVAILABLE
    assert "converted artifacts are absent" in result.reason

    Path(profile.converted_checkpoint.path).write_text("weights")
    Path(profile.converted_vq.path).write_text("codebook")
    assert adapter.capability().status is CapabilityStatus.READY_UNQUALIFIED


def test_minivla_snapshot_round_trips_and_rejects_a_sampling_change(tmp_path):
    minter = minivla_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(8)
    golden = PolicyObservation(image_history=frames(b"g0"), prompt="Close the drawer", timestamp=0.0)
    certification = mint(minter, golden, tmp_path, prefix=7, ensembling=False)

    adapter = minivla_adapter(tmp_path, certification=certification)
    adapter.reset(8)
    adapter.plan_control(PolicyObservation(image_history=frames(b"f0"), prompt="Close the drawer", timestamp=0.0), 70)
    snapshot = json.loads(json.dumps(adapter.snapshot_state()))

    resumed = minivla_adapter(tmp_path, certification=certification)
    resumed.restore_state(snapshot)
    assert resumed.snapshot_state() == snapshot
    assert resumed.snapshot_state()["extra"]["sampling"] == {"do_sample": False, "action_chunk": 7}

    with pytest.raises(PolicyContractError, match="certified path is deterministic"):
        resumed.restore_state(
            {**snapshot, "extra": {**snapshot["extra"], "sampling": {"do_sample": True, "action_chunk": 7}}}
        )


# ---------------------------------------------------------------------- OpenPiZero


def openpizero_observation(tag=b"f0", *, proprio=PROPRIO, timestamp=0.0):
    return PolicyObservation(
        image_history=frames(tag), prompt="Put the eggplant in the blue sink", proprio=proprio, timestamp=timestamp
    )


def test_openpizero_proposes_four_actions_from_refreshed_eight_dimensional_proprio(tmp_path):
    minter = openpizero_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(6)
    golden = openpizero_observation(b"g0")
    certification = mint(minter, golden, tmp_path, prefix=4, ensembling=False)

    wrapper = FakeOpenPiZeroWrapper()
    adapter = openpizero_adapter(tmp_path, certification=certification, wrapper=wrapper)
    adapter.reset(6)
    plan = adapter.plan_control(openpizero_observation(b"f0"), 100)

    assert adapter.contract.requires_proprio is True
    assert adapter.contract.native_proposal_horizon == 4
    assert adapter.contract.certified_execute_prefix == 4
    assert len(plan.actions) == 4 and plan.verified_prefix == 4
    assert plan.metadata["action_normalization_type"] == "bounds"
    assert plan.metadata["proprio_refreshed_at_timestamp"] == 0.0
    assert plan.metadata["proprio_width"] == 8
    assert plan.metadata["proprio_gripper_units"] == "bridge_state"
    assert len(wrapper.calls) == 1
    assert wrapper.calls[0]["proprio"] == PROPRIO
    assert wrapper.calls[0]["text"] == "Put the eggplant in the blue sink"
    assert adapter.normalizer.counters()["denormalize_calls"] == 4

    dropped = openpizero_adapter(
        tmp_path,
        certification=certification,
        profile=openpizero_profile(tmp_path, proprio_layout=BRIDGE_7D_DROP_STRUCTURAL_ZERO_LAYOUT),
    )
    dropped.reset(6)
    dropped.plan_control(openpizero_observation(b"f9"), 100)
    assert dropped.fake_wrapper.calls[0]["proprio"] == PROPRIO[:6] + (PROPRIO[7],)


def test_openpizero_refuses_missing_zero_filled_and_malformed_proprio(tmp_path):
    minter = openpizero_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(6)
    certification = mint(minter, openpizero_observation(b"g0"), tmp_path, prefix=4, ensembling=False)
    adapter = openpizero_adapter(tmp_path, certification=certification)
    adapter.reset(6)

    with pytest.raises(PolicyContractError, match="never zero-filled"):
        adapter.plan_control(openpizero_observation(b"f0", proprio=None), 100)
    with pytest.raises(PolicyContractError, match="zero-fill, not a measurement"):
        adapter.plan_control(openpizero_observation(b"f1", proprio=(0.0,) * 8), 100)
    with pytest.raises(PolicyContractError, match="canonical 8-D Bridge state"):
        adapter.plan_control(openpizero_observation(b"f2", proprio=PROPRIO[:7]), 100)
    with pytest.raises(PolicyContractError, match="structural zero"):
        adapter.plan_control(
            openpizero_observation(b"f3", proprio=PROPRIO[:6] + (0.5, PROPRIO[7])), 100
        )
    with pytest.raises(PolicyContractError, match="finite real numbers"):
        adapter.plan_control(openpizero_observation(b"f4", proprio=PROPRIO[:7] + (float("nan"),)), 100)
    # An action-unit gripper value in the state channel is caught by the unit type.
    with pytest.raises(PolicyContractError, match="is not a state value"):
        adapter.plan_control(openpizero_observation(b"f5", proprio=PROPRIO[:7] + (0.996,)), 100)
    assert adapter.fake_wrapper.calls == []

    adapter.plan_control(openpizero_observation(b"f6", timestamp=0.0), 100)
    with pytest.raises(PolicyContractError, match="cannot establish that the state was refreshed"):
        adapter.plan_control(openpizero_observation(b"f7", timestamp=None), 100)


def test_openpizero_requires_the_bounds_family_and_a_resolved_gripper_conversion(tmp_path):
    with pytest.raises(PolicyContractError, match="declares action_normalization_type='bounds'"):
        OpenPiZeroPolicyAdapter(
            openpizero_profile(tmp_path), normalizer=octo_normalizer(), wrapper_factory=lambda p, r: None
        )

    action_units = openpizero_profile(
        tmp_path,
        proprio_layout=replace(BRIDGE_8D_PASSTHROUGH_LAYOUT, gripper_units="action_gripper"),
    )
    blocked = openpizero_adapter(tmp_path, profile=action_units)
    assert "resolved gripper polarity convention" in blocked.capability().reason

    resolved = OpenPiZeroPolicyAdapter(
        action_units,
        normalizer=openpizero_normalizer(),
        gripper_convention=GripperPolarityConvention(
            convention_id="fixture-high-is-open", source_revision=REVISION, action_high_means_open=True
        ),
        execution_mode=PolicyExecutionMode.FIXTURE_CERTIFICATION,
        runtime_factory=openpizero_runtime,
        wrapper_factory=lambda profile, runtime: FakeOpenPiZeroWrapper(),
    )
    assert resolved._profile_errors() == ()
    resolved.reset(1)
    proposal = resolved.propose(openpizero_observation(b"g1"))
    assert proposal.metadata["proprio_gripper_units"] == "action_gripper"


def test_openpizero_accepts_official_paligemma_and_a_compared_mirror_only(tmp_path):
    official = openpizero_adapter(tmp_path)
    assert official.capability().details["paligemma"]["official"] is True
    assert official.capability().details["paligemma"]["byte_identity_claimed"] is False
    assert official._profile_errors() == ()

    unaccepted = openpizero_adapter(
        tmp_path, profile=openpizero_profile(tmp_path, paligemma=paligemma(tmp_path, terms_accepted=False))
    )
    assert "explicitly accepted access terms" in unaccepted.capability().reason

    mirror = openpizero_adapter(
        tmp_path,
        profile=openpizero_profile(tmp_path, paligemma=paligemma(tmp_path, repo_id=PALIGEMMA_CANDIDATE_MIRROR)),
    )
    assert "candidate mirror only" in mirror.capability().reason
    assert "byte identity may not be claimed" in mirror.capability().reason

    compared = openpizero_adapter(
        tmp_path,
        profile=openpizero_profile(
            tmp_path,
            paligemma=paligemma(
                tmp_path, repo_id=PALIGEMMA_CANDIDATE_MIRROR, mirror_comparison_report_sha256=DIGEST
            ),
        ),
    )
    assert compared._profile_errors() == ()

    third_party = openpizero_adapter(
        tmp_path, profile=openpizero_profile(tmp_path, paligemma=paligemma(tmp_path, repo_id="someone/paligemma"))
    )
    assert "must come from the official" in third_party.capability().reason


def test_openpizero_reports_missing_paligemma_files_and_unverified_state_dict_coverage(tmp_path):
    minter = openpizero_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(6)
    certification = mint(minter, openpizero_observation(b"g0"), tmp_path, prefix=4, ensembling=False)

    profile = openpizero_profile(tmp_path, paligemma=paligemma(tmp_path / "absent", create=False))
    adapter = OpenPiZeroPolicyAdapter(
        profile,
        normalizer=openpizero_normalizer(),
        certification=certification,
        runtime_factory=openpizero_runtime,
    )
    result = adapter.capability()
    assert result.status is CapabilityStatus.UNAVAILABLE
    assert "reviewed converted checkpoint" in result.reason

    Path(profile.converted_checkpoint_path).write_text("weights")
    result = adapter.capability()
    assert result.status is CapabilityStatus.UNAVAILABLE
    assert "PaliGemma support files are missing" in result.reason
    assert "tokenizer.model" in result.reason

    unverified = openpizero_adapter(tmp_path, profile=openpizero_profile(tmp_path, state_dict_coverage_verified=False))
    assert "strict=True checks keys, not provenance" in unverified.capability().reason


def test_openpizero_snapshot_round_trips_its_last_consumed_proprio(tmp_path):
    minter = openpizero_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(6)
    certification = mint(minter, openpizero_observation(b"g0"), tmp_path, prefix=4, ensembling=False)

    adapter = openpizero_adapter(tmp_path, certification=certification)
    adapter.reset(6)
    adapter.plan_control(openpizero_observation(b"f0"), 100)
    snapshot = json.loads(json.dumps(adapter.snapshot_state()))
    assert snapshot["extra"]["last_proprio"] == list(PROPRIO)

    resumed = openpizero_adapter(tmp_path, certification=certification)
    resumed.restore_state(snapshot)
    assert resumed.snapshot_state() == snapshot

    with pytest.raises(PolicyContractError, match="proprio layout"):
        resumed.restore_state(
            {**snapshot, "extra": {**snapshot["extra"], "proprio_layout_id": "something-else"}}
        )


# --------------------------------------------------------------------------- SuSIE


def susie_observation(tag=b"f0", *, goal=b"goal", timestamp=0.0, prompt="fold the cloth from top right to bottom left"):
    return PolicyObservation(
        image_history=frames(tag),
        prompt=prompt,
        goal_image=None if goal is None else FakeFrame(goal),
        timestamp=timestamp,
    )


def test_susie_without_a_goal_image_source_is_blocked_with_a_named_reason(tmp_path):
    for cls, subgoal in ((SuSIEPolicyAdapter, True), (SuSIELowLevelPolicyAdapter, False)):
        adapter = susie_adapter(
            tmp_path, cls=cls, profile=susie_profile(tmp_path, subgoal=subgoal, goal_image_source=None)
        )
        result = adapter.capability()
        assert result.status is CapabilityStatus.BLOCKED
        assert "susie_goal_image_source_unavailable" in result.reason
        assert "will not substitute the start frame for a goal" in result.reason
        assert result.details["goal_image_required"] is True
        assert result.details["start_frame_substituted_for_goal"] is False


def test_susie_refuses_an_absent_goal_or_one_identical_to_the_start_frame(tmp_path):
    minter = susie_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(15)
    certification = mint(minter, susie_observation(b"g0"), tmp_path, prefix=1, ensembling=False)
    adapter = susie_adapter(tmp_path, certification=certification)
    adapter.reset(15)

    with pytest.raises(PolicyContractError, match="requires an explicit goal image"):
        adapter.plan_control(susie_observation(b"f0", goal=None), 80)
    with pytest.raises(PolicyContractError, match="identical to its current observation frame"):
        adapter.plan_control(susie_observation(b"same", goal=b"same"), 80)
    assert adapter.fake_low_level.calls == []

    adapter.plan_control(susie_observation(b"start", goal=b"goal", timestamp=0.0), 80)
    with pytest.raises(PolicyContractError, match="identical to this episode's start frame"):
        adapter.plan_control(susie_observation(b"next", goal=b"start", timestamp=0.2), 79)
    with pytest.raises(PolicyContractError, match="goal image changed mid-episode"):
        adapter.plan_control(susie_observation(b"next", goal=b"other-goal", timestamp=0.2), 79)


def test_susie_regenerates_a_subgoal_on_its_declared_cadence_and_caches_between(tmp_path):
    minter = susie_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(15)
    certification = mint(minter, susie_observation(b"g0"), tmp_path, prefix=1, ensembling=False)

    adapter = susie_adapter(tmp_path, certification=certification)
    adapter.reset(15)
    for index in range(7):
        adapter.plan_control(susie_observation(b"f%d" % index, timestamp=index * 0.2), 80 - index)

    subgoal = adapter.fake_subgoal
    low_level = adapter.fake_low_level
    assert adapter.profile.subgoal_cadence == 3
    # Steps 0, 3 and 6 refresh; the rest reuse the cached subgoal image.
    assert len(subgoal.calls) == 3
    assert len(low_level.calls) == 7
    conditioning = [call["goal_image"].tag for call in low_level.calls]
    assert conditioning == [b"subgoal-1"] * 3 + [b"subgoal-2"] * 3 + [b"subgoal-3"]
    assert adapter.snapshot_state()["extra"]["subgoal"]["refreshes"] == 3
    assert adapter.last_proposal.metadata["conditioning"] == "generated_subgoal"
    assert adapter.last_proposal.backend_calls == 2


def test_susie_resume_inside_a_subgoal_interval_refuses_rather_than_regenerating(tmp_path):
    minter = susie_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(15)
    certification = mint(minter, susie_observation(b"g0"), tmp_path, prefix=1, ensembling=False)

    adapter = susie_adapter(tmp_path, certification=certification)
    adapter.reset(15)
    for index in range(2):
        adapter.plan_control(susie_observation(b"f%d" % index, timestamp=index * 0.2), 80 - index)
    mid = json.loads(json.dumps(adapter.snapshot_state()))
    assert mid["extra"]["subgoal"]["pixels_available"] is True

    resumed = susie_adapter(tmp_path, certification=certification)
    resumed.restore_state(mid)
    degraded = resumed.snapshot_state()
    # Exactly one documented field degrades: the diffusion subgoal's pixels are
    # not restorable from a JSON checkpoint, though its hash is retained.
    assert degraded["extra"]["subgoal"]["pixels_available"] is False
    assert degraded["extra"]["subgoal"]["identity"] == mid["extra"]["subgoal"]["identity"]
    assert {**degraded, "extra": mid["extra"]} == {**mid, "extra": mid["extra"]}
    resumed.restore_state(degraded)
    assert resumed.snapshot_state() == degraded

    with pytest.raises(PolicyContractError, match="resume only on a subgoal cadence boundary"):
        resumed.plan_control(susie_observation(b"f2", timestamp=0.4), 78)

    boundary = susie_adapter(tmp_path, certification=certification)
    boundary.restore_state({**mid, "step": 3})
    boundary.plan_control(susie_observation(b"f3", timestamp=0.6), 77)
    assert len(boundary.fake_subgoal.calls) == 1


def test_susie_ll_executes_the_low_level_policy_directly_against_the_task_goal(tmp_path):
    minter = susie_adapter(tmp_path, cls=SuSIELowLevelPolicyAdapter, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(19)
    certification = mint(minter, susie_observation(b"g0"), tmp_path, prefix=1, ensembling=False)

    adapter = susie_adapter(tmp_path, cls=SuSIELowLevelPolicyAdapter, certification=certification)
    adapter.reset(19)
    plan = adapter.plan_control(susie_observation(b"f0"), 80)

    assert adapter.base_contract.name == "SuSIE_LL"
    assert adapter.requires_subgoal is False
    assert adapter.fake_subgoal is None
    assert plan.metadata["conditioning"] == "task_goal_image"
    assert plan.metadata["subgoal_identity"] is None
    assert adapter.fake_low_level.calls[0]["goal_image"].tag == b"goal"
    assert adapter.last_proposal.backend_calls == 1

    misconfigured = susie_adapter(
        tmp_path,
        cls=SuSIELowLevelPolicyAdapter,
        profile=susie_profile(tmp_path, subgoal=True),
    )
    assert "no subgoal diffusion stage" in misconfigured.capability().reason

    without_subgoal = susie_adapter(tmp_path, profile=susie_profile(tmp_path, subgoal=False))
    assert "separate subgoal diffusion checkpoint" in without_subgoal.capability().reason


def test_susie_released_and_corrected_arms_are_separate_policy_identities(tmp_path):
    released = susie_adapter(tmp_path)
    assert released.capability().details["arm"]["arm_id"] == "autoeval_released_replication"
    assert released.capability().details["arm"]["low_level_agent"] == "gc_bc"
    assert released.capability().details["arm"]["is_released_replication"] is True

    corrected = susie_adapter(tmp_path, profile=susie_profile(tmp_path, arm=SUSIE_UPSTREAM_CORRECTED_ARM))
    assert corrected.capability().details["arm"]["low_level_agent"] == "gc_ddpm_bc"
    assert corrected.capability().details["arm"]["is_released_replication"] is False

    mislabelled = SuSIEArmConfiguration(arm_id="autoeval_released_replication", low_level_agent="gc_ddpm_bc")
    assert "misattribute a corrected configuration" in (mislabelled.review_error() or "")
    assert "must be autoeval_released_replication" in (
        SuSIEArmConfiguration(arm_id="made_up", low_level_agent="gc_bc").review_error() or ""
    ).replace("arm_id must be 'autoeval_released_replication'", "must be autoeval_released_replication")

    minter = susie_adapter(tmp_path, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(15)
    certification = mint(minter, susie_observation(b"g0"), tmp_path, prefix=1, ensembling=False)
    adapter = susie_adapter(tmp_path, certification=certification)
    adapter.reset(15)
    adapter.plan_control(susie_observation(b"f0"), 80)
    snapshot = adapter.snapshot_state()
    other_arm = susie_adapter(tmp_path, certification=certification, profile=susie_profile(tmp_path, arm=SUSIE_UPSTREAM_CORRECTED_ARM))
    with pytest.raises(PolicyContractError, match="arms are separate policy identities"):
        other_arm.restore_state(snapshot)


# ------------------------------------------------------- cross-adapter invariants


def uncertified_adapters(tmp_path):
    """One uncertified instance of every native adapter in the matrix."""

    return (
        octo_adapter(),
        octo_adapter(
            cls=OctoBaseV1PolicyAdapter,
            profile=octo_profile(model_id=OCTO_BASE_MODEL_ID, checkpoint_step=OCTO_BASE_CHECKPOINT_STEP),
        ),
        minivla_adapter(tmp_path / "minivla"),
        openpizero_adapter(tmp_path / "openpizero"),
        susie_adapter(tmp_path / "susie"),
        susie_adapter(tmp_path / "susie-ll", cls=SuSIELowLevelPolicyAdapter),
    )


def test_every_declared_native_contract_starts_uncertified_and_blocked():
    for contract in uncertified_contracts():
        assert contract.certified_execute_prefix is None, contract.name
        assert contract.implementation_status is CapabilityStatus.BLOCKED, contract.name
        assert contract.limitation


def test_every_native_adapter_is_blocked_and_emits_no_action_without_a_certification(tmp_path):
    for adapter in uncertified_adapters(tmp_path):
        name = adapter.base_contract.name
        result = adapter.capability()
        assert result.status is CapabilityStatus.BLOCKED, name
        assert adapter.contract.certified_execute_prefix is None, name
        assert adapter.contract.implementation_status is CapabilityStatus.BLOCKED, name
        assert "no_certification_artifact" in adapter.certification_errors(), name
        adapter.reset(1)
        observation = PolicyObservation(
            image_history=frames(*([b"f0"] * adapter.base_contract.required_observation_history)),
            prompt="Open the drawer",
            proprio=PROPRIO if adapter.base_contract.requires_proprio else None,
            goal_image=FakeFrame(b"goal"),
            timestamp=0.0,
        )
        with pytest.raises(PolicyCertificationError, match="golden action fixture"):
            adapter.plan_control(observation, 70)
        with pytest.raises(PolicyCertificationError, match="refuses to produce actions in certified mode"):
            adapter.propose(observation)
        with pytest.raises(PolicyCertificationError, match="no certification fixture"):
            adapter.certify(observation)


def test_fixture_certification_mode_never_hands_a_rollout_a_verified_prefix(tmp_path):
    adapter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    adapter.reset(1)
    observation = PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0)

    assert adapter.propose(observation).actions
    with pytest.raises(PolicyCertificationError, match="will not hand a rollout a verified prefix"):
        adapter.plan_control(observation, 70)


def certified_octo(root):
    root.mkdir(parents=True, exist_ok=True)
    minter = octo_adapter(mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    certification = mint(
        minter,
        PolicyObservation(image_history=frames(b"g0", b"g0"), prompt="Open the drawer", timestamp=0.0),
        root,
        prefix=1,
        ensembling=True,
    )
    adapter = octo_adapter(certification=certification)
    adapter.reset(1)

    def observation(index, tag_index=None):
        tag = b"f%d" % (index if tag_index is None else tag_index)
        return PolicyObservation(
            image_history=(FakeFrame(b"prev"), FakeFrame(tag)), prompt="Open the drawer", timestamp=index * 0.2
        )

    return adapter, observation


def certified_minivla(root):
    root.mkdir(parents=True, exist_ok=True)
    minter = minivla_adapter(root, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    certification = mint(
        minter,
        PolicyObservation(image_history=frames(b"g0"), prompt="Close the drawer", timestamp=0.0),
        root,
        prefix=7,
        ensembling=False,
    )
    adapter = minivla_adapter(root, certification=certification)
    adapter.reset(1)

    def observation(index, tag_index=None):
        tag = b"f%d" % (index if tag_index is None else tag_index)
        return PolicyObservation(image_history=(FakeFrame(tag),), prompt="Close the drawer", timestamp=index * 0.2)

    return adapter, observation


def certified_openpizero(root):
    root.mkdir(parents=True, exist_ok=True)
    minter = openpizero_adapter(root, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    certification = mint(minter, openpizero_observation(b"g0"), root, prefix=4, ensembling=False)
    adapter = openpizero_adapter(root, certification=certification)
    adapter.reset(1)

    def observation(index, tag_index=None):
        tag = b"f%d" % (index if tag_index is None else tag_index)
        return openpizero_observation(tag, timestamp=index * 0.2)

    return adapter, observation


def certified_susie(root):
    root.mkdir(parents=True, exist_ok=True)
    minter = susie_adapter(root, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    certification = mint(minter, susie_observation(b"g0"), root, prefix=1, ensembling=False)
    adapter = susie_adapter(root, certification=certification)
    adapter.reset(1)

    def observation(index, tag_index=None):
        tag = b"f%d" % (index if tag_index is None else tag_index)
        return susie_observation(tag, timestamp=index * 0.2)

    return adapter, observation


def certified_susie_ll(root):
    root.mkdir(parents=True, exist_ok=True)
    minter = susie_adapter(root, cls=SuSIELowLevelPolicyAdapter, mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    minter.reset(1)
    certification = mint(minter, susie_observation(b"g0"), root, prefix=1, ensembling=False)
    adapter = susie_adapter(root, cls=SuSIELowLevelPolicyAdapter, certification=certification)
    adapter.reset(1)

    def observation(index, tag_index=None):
        tag = b"f%d" % (index if tag_index is None else tag_index)
        return susie_observation(tag, timestamp=index * 0.2)

    return adapter, observation


CERTIFIED_BUILDERS = (certified_octo, certified_minivla, certified_openpizero, certified_susie, certified_susie_ll)


@pytest.mark.parametrize("builder", CERTIFIED_BUILDERS, ids=lambda builder: builder.__name__)
def test_every_certified_adapter_is_ready_unqualified_and_never_qualified(tmp_path, builder):
    adapter, observation = builder(tmp_path / builder.__name__)
    result = adapter.capability()

    assert result.status is CapabilityStatus.READY_UNQUALIFIED
    assert result.allows_qualified_use is False
    assert "unqualified until Gate B" in result.reason
    assert adapter.contract.implementation_status is CapabilityStatus.READY_UNQUALIFIED
    assert adapter.contract.certified_execute_prefix == adapter.certification.certified_execute_prefix
    assert "Gate B" in (adapter.contract.limitation or "")
    plan = adapter.plan_control(observation(0), 70)
    assert plan.metadata["qualified"] is False
    assert plan.metadata["certification_fixture_sha256"] == adapter.certification.fixture_sha256
    assert len(plan.actions[0]) == 7


@pytest.mark.parametrize("builder", CERTIFIED_BUILDERS, ids=lambda builder: builder.__name__)
def test_every_certified_adapter_refuses_a_requeried_stale_frame(tmp_path, builder):
    adapter, observation = builder(tmp_path / builder.__name__)
    adapter.plan_control(observation(0), 70)

    with pytest.raises(PolicyContractError, match="requerying a policy against a stale image"):
        adapter.plan_control(observation(1, tag_index=0), 69)


@pytest.mark.parametrize("builder", CERTIFIED_BUILDERS, ids=lambda builder: builder.__name__)
def test_every_certified_adapter_round_trips_durable_state(tmp_path, builder):
    adapter, observation = builder(tmp_path / builder.__name__)
    adapter.plan_control(observation(0), 70)
    adapter.plan_control(observation(1), 69)
    snapshot = json.loads(json.dumps(adapter.snapshot_state()))

    resumed, _ = builder(tmp_path / (builder.__name__ + "-resume"))
    resumed.restore_state(snapshot)
    restored = json.loads(json.dumps(resumed.snapshot_state()))

    assert restored["step"] == snapshot["step"] == 2
    assert restored["seed"] == snapshot["seed"]
    assert restored["normalizer_counters"] == snapshot["normalizer_counters"]
    assert restored["last_frame_identity"] == snapshot["last_frame_identity"]
    assert restored["last_timestamp"] == snapshot["last_timestamp"]
    # Restoring a restored snapshot is a fixed point, so any documented
    # degradation (an unrestorable diffusion subgoal) happens exactly once.
    resumed.restore_state(restored)
    assert json.loads(json.dumps(resumed.snapshot_state())) == restored


ADAPTER_SOURCES = (
    "plumb/policies/octo.py",
    "plumb/policies/minivla.py",
    "plumb/policies/openpizero.py",
    "plumb/policies/susie.py",
)
HEAVY_TOP_LEVEL_MODULES = frozenset(
    {
        "torch",
        "torchvision",
        "jax",
        "jaxlib",
        "flax",
        "transformers",
        "tokenizers",
        "timm",
        "prismatic",
        "octo",
        "diffusers",
        "PIL",
        "numpy",
        "tensorflow",
        "huggingface_hub",
    }
)
NATIVE_CALL_NAMES = frozenset(
    {"sample_actions", "predict_action", "infer_action", "sample_subgoal", "create_tasks", "load_vla"}
)


def _repository_root():
    return Path(__file__).resolve().parents[1]


class _NativeCallAudit(ast.NodeVisitor):
    def __init__(self):
        self.loop_depth = 0
        self.calls = []
        self.calls_in_loops = []

    def _loop(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_For = _loop
    visit_AsyncFor = _loop
    visit_While = _loop
    visit_ListComp = _loop
    visit_SetComp = _loop
    visit_DictComp = _loop
    visit_GeneratorExp = _loop

    def visit_Call(self, node):
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
        if name in NATIVE_CALL_NAMES:
            self.calls.append(name)
            if self.loop_depth:
                self.calls_in_loops.append(name)
        self.generic_visit(node)


@pytest.mark.parametrize("relative", ADAPTER_SOURCES)
def test_static_guard_forbids_requerying_a_policy_to_fabricate_a_longer_proposal(relative):
    path = _repository_root() / relative
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    audit = _NativeCallAudit()
    audit.visit(tree)

    # A native wrapper is called at most once per name and never from inside a
    # loop or comprehension, so a 16-row proposal cannot be manufactured by
    # repeatedly querying a single-step policy against the same image.
    assert audit.calls_in_loops == [], "%s calls a native wrapper inside a loop" % relative
    for name in set(audit.calls):
        assert audit.calls.count(name) == 1, "%s calls %s more than once" % (relative, name)
    assert "* 16" not in source and "range(16)" not in source
    assert "* 15" not in source
    # Only a golden certification may set an execute prefix.
    assert "verified_prefix" not in source


@pytest.mark.parametrize("relative", ADAPTER_SOURCES)
def test_static_guard_keeps_every_heavy_dependency_inside_the_load_path(relative):
    path = _repository_root() / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in HEAVY_TOP_LEVEL_MODULES, (relative, alias.name)
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in HEAVY_TOP_LEVEL_MODULES, (relative, node.module)
            assert node.module != "plumb.rollout", "%s must import PolicyControlPlan lazily" % relative


def test_importing_plumb_policies_loads_no_model_framework():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import plumb.policies, sys;"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch','jax','transformers','flax','diffusers','numpy','PIL','prismatic','octo')))",
        ],
        capture_output=True,
        text=True,
        cwd=str(_repository_root()),
        check=True,
    )
    assert completed.stdout.strip() == "[]", completed.stdout


# -------------------------------------------------------- real controller integration


@dataclass(frozen=True)
class FakeCompiled:
    backend_actions: tuple
    forecast_states: tuple
    feedback_mode: FeedbackMode = FeedbackMode.FORECAST_STATE


class FakeCompiler:
    def compile(self, state, actions):
        states = [state]
        current = state
        for action in actions:
            values = list(current.values8())
            values[0] += action[0]
            current = BridgeState.from_values(values, source="forecast")
            states.append(current)
        return FakeCompiled(tuple(tuple(float(value) for value in action) for action in actions), tuple(states))


class FakeWorld:
    def __init__(self, lengths):
        from plumb.rollout import WorldActionProfile

        self.rollout_profile = WorldActionProfile("fake-world", "bridge_orig_lerobot", tuple(lengths))
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        frame_count = len(request.compiled_actions)
        generated = tuple(
            "generated:%02d:%02d" % (len(self.requests), index) for index in range(frame_count)
        )
        return WorldResult(
            backend="fake",
            profile_id="fake-world",
            frames=(request.conditioning_image,) + generated,
            nominal_frame_timestamps=tuple(index / 5.0 for index in range(frame_count + 1)),
            conditioning_frame_included=True,
            timing=ServerTiming(backend_calls=1, wall_seconds=0.001, cold_start=False),
        )


def test_certified_minivla_drives_the_real_rollout_controller_at_its_native_prefix(tmp_path):
    from plumb.rollout import FileArtifactStore, RolloutController, Scenario

    adapter, _ = certified_minivla(tmp_path / "minivla")
    world = FakeWorld((7,))
    scenario = Scenario(
        initial_rgb="initial:close_drawer",
        state=BridgeState.from_values((0.31, -0.02, 0.07, 0.01, -0.02, 0.03, 0.0, 0.2), source="measured"),
        task="close_drawer",
        horizon_actions=70,
        hashes={"image": "sha256:initial", "state": "sha256:state"},
        seed=1,
    )

    report = RolloutController().execute(
        scenario, adapter, world, FakeCompiler(), None, FileArtifactStore(tmp_path / "artifacts")
    )

    assert report.status == "completed"
    assert report.executed_actions == 70
    assert report.qualified is False
    assert [segment.action_count for segment in report.segments] == [7] * 10
    assert len(world.requests) == 10
    assert all(len(request.compiled_actions) == 7 for request in world.requests)
    # Every segment after the first conditions on a freshly generated frame.
    assert world.requests[0].conditioning_image == "initial:close_drawer"
    assert world.requests[1].conditioning_image == "generated:01:06"
    assert len(report.native_actions) == 70
    assert all(len(row) == 7 for row in report.native_actions)
    assert report.segments[0].world_timing["resumable"] is True


def test_certified_octo_one_step_prefix_is_refused_by_a_sixteen_only_backend(tmp_path):
    from plumb.rollout import FileArtifactStore, RolloutController, Scenario, UnsupportedActionLengthError

    adapter, _ = certified_octo(tmp_path / "octo")
    world = FakeWorld((16, 6))
    scenario = Scenario(
        initial_rgb="initial:open_drawer",
        state=BridgeState.from_values((0.31, -0.02, 0.07, 0.01, -0.02, 0.03, 0.0, 0.2), source="measured"),
        task="open_drawer",
        horizon_actions=70,
        hashes={"image": "sha256:initial", "state": "sha256:state"},
        seed=1,
    )

    # Octo's temporal ensembling executes one action per native call, so a
    # backend that only accepts 16-action requests cannot host it. The refusal
    # happens before any world call rather than padding the proposal to 16.
    with pytest.raises(UnsupportedActionLengthError, match="not be repeated or padded to 16"):
        RolloutController().execute(
            scenario, adapter, world, FakeCompiler(), None, FileArtifactStore(tmp_path / "artifacts")
        )
    assert world.requests == []


# ------------------------------------------------------------------------ asset plan


SPEC_REQUIRED_REPOS = (
    "nvidia/Cosmos3-Nano",
    "nvidia/Cosmos3-Edge",
    "nvidia/Cosmos-1.0-Guardrail",
    "openvla/openvla-7b",
    "rail-berkeley/octo-small",
    "rail-berkeley/octo-base",
    "Stanford-ILIAD/minivla-vq-bridge-prismatic",
    "Stanford-ILIAD/pretrain_vq",
    "allenzren/open-pi-zero",
    "google/paligemma-3b-pt-224",
    "leo009/paligemma-3b-pt-224",
    "kvablack/susie",
    "patreya/gcbc-bridge",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "facebook/vjepa2-vitl-fpc64-256",
    "IPEC-COMMUNITY/bridge_orig_lerobot",
    "zhouzypaul/auto_eval",
)


def test_asset_plan_is_internally_consistent_and_covers_every_spec_asset():
    from cluster.asset_plan import ASSET_PLAN, asset_plan_entries, plan_names, validate_plan

    assert validate_plan() == ()
    repos = {entry.repo_id for entry in ASSET_PLAN.values()}
    for repo in SPEC_REQUIRED_REPOS:
        assert repo in repos, repo
    assert len(plan_names()) == len(ASSET_PLAN)
    entries = asset_plan_entries()
    assert len(entries) == len(ASSET_PLAN)
    matrix = {entry["repo_id"] for entry in entries if entry["required_for_primary_matrix"]}
    for repo in (
        "openvla/openvla-7b",
        "rail-berkeley/octo-small",
        "Stanford-ILIAD/minivla-vq-bridge-prismatic",
        "Stanford-ILIAD/pretrain_vq",
        "allenzren/open-pi-zero",
        "kvablack/susie",
        "patreya/gcbc-bridge",
    ):
        assert repo in matrix, repo
    # Octo-Base is a diagnostic, not a seventh benchmark policy.
    assert "rail-berkeley/octo-base" not in matrix


def test_every_asset_plan_entry_has_a_resolved_license_or_an_explicit_null_prohibition():
    for entry in _asset_entries():
        name = entry["name"]
        assert entry["license_status"], name
        if entry["license"] is None:
            assert entry["redistribution"] == "prohibited_pending_resolution", name
            assert entry["license_status"] in (
                "absent_cardData_null",
                "gated_terms_required",
                "unresolved",
            ), name
        else:
            assert entry["license_status"] != "absent_cardData_null", name
            assert entry["redistribution"] != "unresolved", name
        if entry["license_status"] == "unresolved":
            assert "unresolved_license_or_access_terms" in entry["verification_blockers"], name


def _asset_entries():
    from cluster.asset_plan import asset_plan_entries

    return asset_plan_entries()


def _entry(name):
    return next(entry for entry in _asset_entries() if entry["name"] == name)


def test_asset_plan_encodes_the_octo_minivla_paligemma_and_auto_eval_traps():
    from cluster.asset_plan import get_plan

    octo = _entry("octo-small")
    assert any("270000/default/checkpoint" in trap for trap in octo["traps"])
    assert any("resolve/main/checkpoint returns 404" in trap for trap in octo["traps"])
    assert any("octo-small-1.5" in trap for trap in octo["traps"])
    assert octo["files"][0]["path"] == "270000/default/checkpoint"
    assert "270000/default/checkpoint/*" in octo["allow_patterns"]
    assert _entry("octo-base")["files"][0]["path"] == "300000/default/checkpoint"

    minivla = get_plan("minivla-vq-bridge")
    planned = minivla.files[0]
    assert planned.path.endswith("loss=0.2259.pt")
    assert planned.url_path.endswith("loss%3D0.2259.pt")
    assert "%3D" in minivla.direct_download_url(planned)
    # Hub API clients take the literal path, never the percent-encoded form.
    assert minivla.allow_patterns() == (planned.path,)
    assert "*.jsonl" in minivla.ignore_patterns
    assert any(".jsonl" in trap and "training log" in trap for trap in minivla.traps)

    vq = _entry("minivla-pretrain-vq")
    assert vq["license"] is None
    assert vq["license_status"] == "absent_cardData_null"
    assert vq["redistribution"] == "prohibited_pending_resolution"
    assert any("cardData: null" in trap for trap in vq["traps"])
    assert any("mirroring" in note.lower() for note in vq["notes"])

    official = _entry("paligemma-official")
    assert official["access"] == "gated_accept_terms"
    assert sorted(item["path"] for item in official["files"]) == sorted(PALIGEMMA_REQUIRED_FILES)
    mirror = _entry("paligemma-mirror-candidate")
    assert mirror["fetch"] is False
    assert any("byte identity" in trap for trap in mirror["traps"])

    auto_eval = _entry("auto-eval")
    assert any("robot_eval_logger" in trap and "wandb" in trap for trap in auto_eval["traps"])
    assert any("stub unpickler is NOT a security boundary" in trap for trap in auto_eval["traps"])
    assert _entry("bridge-orig-lerobot")["expected_file_count"] == 99673

    guardrail = _entry("cosmos-guardrail")
    assert guardrail["fetch"] is False
    assert any("enable_safety_checker=False" in note for note in guardrail["notes"])


def test_asset_plan_lock_entries_are_never_verified_and_keep_unknowns_null():
    for entry in _asset_entries():
        name = entry["name"]
        assert entry["verified"] is False, name
        assert entry["status"] == "planned", name
        assert entry["verification_blockers"], name
        assert entry["retrieved_at"] is None and entry["downloaded_bytes"] is None, name
        assert entry["container_digest"] is None, name
        for planned in entry["files"]:
            assert planned["sha256"] is None, name
            assert planned["verified"] is False, name
        if entry["revision"] is None:
            assert "unresolved_immutable_revision" in entry["verification_blockers"], name
            assert entry["revision_immutable"] is False, name
        else:
            assert entry["revision_immutable"] is True, name
    # Only exactly known byte counts are recorded; estimates stay separate.
    vjepa = _entry("vjepa2-vitl-fpc64-256")
    sizes = {item["path"]: item["expected_bytes"] for item in vjepa["files"]}
    assert sizes == {"config.json": 785, "model.safetensors": 1303947864, "video_preprocessor_config.json": 1298}
    assert "original/*" in vjepa["ignore_patterns"]
    assert _entry("susie-low-level")["files"][0]["expected_bytes"] == 258718956
    assert _entry("openvla-7b")["files"] == []
    assert _entry("openvla-7b")["historical_total_estimate_bytes"] == 15085000000


def test_asset_plan_rejects_an_internally_inconsistent_entry():
    from cluster.asset_plan import Access, AssetPlanEntry, LicenseStatus, Redistribution, RepoType

    fabricated = AssetPlanEntry(
        name="fabricated",
        repo_id="someone/thing",
        repo_type=RepoType.MODEL,
        purpose="test",
        source_url="https://example.invalid",
        license=None,
        license_status=LicenseStatus.DECLARED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.PUBLIC,
        revision=None,
    )
    problems = fabricated.errors()
    assert any("license is null but license_status claims a declared license" in problem for problem in problems)
    assert any("redistribution must be prohibited_pending_resolution" in problem for problem in problems)
    assert any("unresolved_immutable_revision" in problem for problem in problems)


def test_download_assets_refuses_unresolved_gated_non_hub_and_unfetched_plans(capsys):
    import cluster.download_assets as downloads

    def refusal(argv):
        parser = downloads.build_parser()
        with pytest.raises(SystemExit):
            downloads._resolve_plan(parser, parser.parse_args(argv))
        return capsys.readouterr().err

    assert "has no immutable revision yet" in refusal(["--from-plan", "octo-small"])
    assert "gated" in refusal(["--from-plan", "paligemma-official", "--revision", "a" * 40])
    assert "not a Hub snapshot" in refusal(["--from-plan", "code-octo"])
    assert "fetch=False" in refusal(["--from-plan", "cosmos-guardrail", "--revision", "a" * 40])
    assert "fetch=False" in refusal(["--from-plan", "paligemma-mirror-candidate", "--revision", "a" * 40])
    assert "Unknown asset plan" in refusal(["--from-plan", "not-a-plan"])
    assert "--revision is required" in refusal(["openvla/openvla-7b"])
    assert "disagrees with plan" in refusal(["someone/else", "--from-plan", "openvla-7b"])

    parser = downloads.build_parser()
    entry = downloads._resolve_plan(parser, parser.parse_args(["--from-plan", "openvla-7b"]))
    assert entry is not None and entry.revision is not None
    assert downloads._resolve_plan(parser, parser.parse_args(["openvla/openvla-7b", "--revision", "a" * 40])) is None


def test_download_assets_only_enforces_exactly_known_byte_lengths():
    import cluster.download_assets as downloads
    from cluster.asset_plan import get_plan

    vjepa = get_plan("vjepa2-vitl-fpc64-256")
    assert downloads._planned_expected_bytes(vjepa, "config.json") == 785
    assert downloads._planned_expected_bytes(vjepa, "model.safetensors") == 1303947864
    # A historical estimate is never enforced as a byte length.
    assert downloads._planned_expected_bytes(get_plan("open-pi-zero"), "bridge_beta_step19296_2024-12-26_22-30_42.pt") is None
    assert downloads._planned_expected_bytes(None, "anything") is None
    assert downloads._selected("checkpoints/a.pt", ["checkpoints/*"], ["*.jsonl"]) is True
    assert downloads._selected("logs/train.jsonl", ["*"], ["*.jsonl"]) is False
    assert downloads._selected("other.bin", ["checkpoints/*"], []) is False
