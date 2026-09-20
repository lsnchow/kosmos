"""Pinned local-only Octo v1 policy adapters.

This module follows the *pinned Octo v0.1 source API* for the released v1.0
checkpoints: it maintains the source ``stack_and_pad`` history, samples a
four-action normalized proposal with ``PRNGKey(0)``, applies the released
normal unnormalization using ``bridge_dataset`` statistics, and emits a
temporally-ensembled physical 7-D action. It deliberately never calls
``hf://`` or allows a model download at prediction time.

The pinned AutoEval Octo server calls a newer ``sample_actions`` signature
with ``unnormalization_statistics=``. That call is incompatible with Octo
v0.1 and is explicitly an unverified, separately named replication profile;
this adapter never silently substitutes it.

The small model is the primary matrix policy.  The base model is available only
as a separately named diagnostic; it is not a seventh matrix policy.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from ..contracts import PolicyContractError, PolicyLoadError


# The v0.1.0 release tag resolves to this commit.  The AutoEval wrapper below
# is pinned separately because it supplies the benchmark-specific history and
# temporal-ensemble behavior.
OCTO_V1_SOURCE_COMMIT = "37951e4e6d708fd76374f6e09e716763fe2673b1"
OCTO_AUTOEVAL_SOURCE_COMMIT = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
OCTO_JAX_VERSION = "0.4.20"
OCTO_SMALL_MODEL_ID = "rail-berkeley/octo-small"
OCTO_SMALL_MODEL_REVISION = "03d88976c54a58e10480d2043a8c762b35bc2611"
OCTO_SMALL_CHECKPOINT_STEP = 270000
OCTO_SMALL_CHECKPOINT_BYTES = 546696551
OCTO_SMALL_CHECKPOINT_SHA256 = "590df097f8a37bbc1c3aac2a488c0fb08e72bae8abaedfb89f0685677d848962"
OCTO_SMALL_EXAMPLE_BATCH_BYTES = 738368
OCTO_SMALL_EXAMPLE_BATCH_SHA256 = "0ce74dd8e433ce4a8a1534c4ab9687d9fc3e444b3eece750047dcb22125d73ff"
OCTO_BASE_MODEL_ID = "rail-berkeley/octo-base"
OCTO_BASE_MODEL_REVISION = "39d6c88fdbbcf6f841481a7d732f68c612d04609"
OCTO_BASE_CHECKPOINT_STEP = 300000
OCTO_BASE_CHECKPOINT_BYTES = 809848679
OCTO_BASE_CHECKPOINT_SHA256 = "a16e66b6aac743afca4d9a830e32291362237212c50843d1ac0895270ea441fe"
OCTO_BASE_EXAMPLE_BATCH_BYTES = 738368
OCTO_BASE_EXAMPLE_BATCH_SHA256 = OCTO_SMALL_EXAMPLE_BATCH_SHA256
OCTO_T5_MODEL_ID = "t5-base"
OCTO_T5_MODEL_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
OCTO_T5_REQUIRED_FILES = (
    "config.json",
    "spiece.model",
    "tokenizer.json",
)
OCTO_DATASET_STATISTICS_KEY = "bridge_dataset"
OCTO_OBSERVATION_HORIZON = 2
OCTO_ACTION_HORIZON = 4
OCTO_NATIVE_V0_1_PROFILE_ID = "octo-v0.1-native-normal-unnormalization"
OCTO_AUTOEVAL_NEWER_API_PROFILE_ID = "autoeval-octo-server-newer-api-unverified"

OCTO_MODEL_SOURCE = (
    "https://github.com/octo-models/octo/blob/"
    + OCTO_V1_SOURCE_COMMIT
    + "/octo/model/octo_model.py"
)
OCTO_GYM_WRAPPERS_SOURCE = (
    "https://github.com/octo-models/octo/blob/"
    + OCTO_V1_SOURCE_COMMIT
    + "/octo/utils/gym_wrappers.py#L12-L24"
)
OCTO_AUTOEVAL_SERVER_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + OCTO_AUTOEVAL_SOURCE_COMMIT
    + "/auto_eval/policy_server/octo_server.py"
)
OCTO_AUTOEVAL_CONTROL_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + OCTO_AUTOEVAL_SOURCE_COMMIT
    + "/run_eval.py#L354-L364"
)
OCTO_AUTOEVAL_CLIENT_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + OCTO_AUTOEVAL_SOURCE_COMMIT
    + "/auto_eval/robot/policy_clients.py#L24-L66"
)
OCTO_SMALL_MODEL_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    OCTO_SMALL_MODEL_ID,
    OCTO_SMALL_MODEL_REVISION,
)
OCTO_BASE_MODEL_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    OCTO_BASE_MODEL_ID,
    OCTO_BASE_MODEL_REVISION,
)


class OctoUnavailableError(PolicyLoadError):
    """The pinned Octo v1 local runtime or its immutable assets are absent."""


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


@dataclass(frozen=True)
class OctoV1PolicyProfile:
    """A reviewed local snapshot for one released Octo v1 policy.

    ``local_model_path`` is the root of the exact Hub snapshot, rather than
    the nested Orbax checkpoint file.  Octo's own ``load_pretrained`` reads
    its config/example/statistics at that root and restores the requested
    checkpoint step beneath it.
    """

    profile_id: str
    local_model_path: str
    checkpoint_revision: str
    code_revision: str = OCTO_V1_SOURCE_COMMIT
    model_id: str = OCTO_SMALL_MODEL_ID
    checkpoint_step: int = OCTO_SMALL_CHECKPOINT_STEP
    jax_version: str = OCTO_JAX_VERSION
    native_api_profile_id: str = OCTO_NATIVE_V0_1_PROFILE_ID
    compatibility_profile_id: Optional[str] = None
    dataset_statistics_key: str = OCTO_DATASET_STATISTICS_KEY
    observation_horizon: int = OCTO_OBSERVATION_HORIZON
    action_horizon: int = OCTO_ACTION_HORIZON
    temporal_ensembling: bool = True
    action_exp_weight: float = 0.0
    rng_seed: int = 0
    local_files_only: bool = True
    source_checkout_path: Optional[str] = None
    asset_manifest_path: Optional[str] = None
    t5_tokenizer_path: Optional[str] = None
    hf_home: Optional[str] = None
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self, *, expected_model_id: str, expected_revision: str, expected_step: int) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_model_path:
            return "local_model_path is required; Hub fetching is disabled"
        if self.model_id != expected_model_id:
            return "model_id must be %r" % expected_model_id
        if self.checkpoint_revision != expected_revision:
            return "checkpoint_revision must be the pinned %s revision" % expected_model_id
        if not _immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable 40-character hexadecimal revision"
        if self.code_revision != OCTO_V1_SOURCE_COMMIT:
            return "code_revision must be pinned to the released Octo v1 source commit"
        if self.native_api_profile_id != OCTO_NATIVE_V0_1_PROFILE_ID:
            return (
                "native_api_profile_id must be %r; the AutoEval newer sample_actions API is not interchangeable"
                % OCTO_NATIVE_V0_1_PROFILE_ID
            )
        if not self.jax_version or not all(part.isdigit() for part in self.jax_version.split(".")):
            return "jax_version must declare an exact numeric JAX base version"
        if self.jax_version != OCTO_JAX_VERSION:
            if not self.compatibility_profile_id:
                return (
                    "a JAX version other than released %s requires compatibility_profile_id" % OCTO_JAX_VERSION
                )
            if not self.runtime_lock_id or not _sha256_hex(self.runtime_lock_sha256):
                return "a JAX compatibility profile requires runtime_lock_id and a 64-character runtime_lock_sha256"
        if self.dataset_statistics_key != OCTO_DATASET_STATISTICS_KEY:
            return "dataset_statistics_key must be 'bridge_dataset'"
        if self.observation_horizon != OCTO_OBSERVATION_HORIZON:
            return "observation_horizon must be exactly 2"
        if self.action_horizon != OCTO_ACTION_HORIZON:
            return "action_horizon must be exactly 4"
        if self.checkpoint_step != expected_step:
            return "checkpoint_step must be %d for this immutable model" % expected_step
        if not self.temporal_ensembling:
            return "native-v0.1 diagnostic profile requires temporal_ensembling=True"
        if not math.isfinite(self.action_exp_weight):
            return "action_exp_weight must be finite"
        if self.rng_seed != 0:
            return "native-v0.1 diagnostic profile requires rng_seed=0"
        if not self.local_files_only:
            return "Octo loader is local-only; network retrieval is not permitted"
        return None


@dataclass(frozen=True)
class OctoActionReport:
    """One actual native Octo proposal and the source-wrapper selected action."""

    action: Tuple[float, float, float, float, float, float, float]
    proposal: Tuple[Tuple[float, float, float, float, float, float, float], ...]
    source_image_timestamp: Optional[float]
    backend_calls: int
    wall_seconds: Optional[float]
    observation_count: int
    rng_seed: int
    temporal_ensembling: bool
    action_exp_weight: float
    gripper_transformation: str
    normalization: str


@dataclass(frozen=True)
class _OctoRuntime:
    """Lazy imports kept injectable so source behavior can be fixture tested."""

    jax: Any
    numpy: Any
    model_cls: Any
    stack_and_pad: Callable[[Sequence[Mapping[str, Any]], int], Mapping[str, Any]]
    jax_version: str


OCTO_SMALL_V1_CONTRACT = PolicyContract(
    name="Octo-Small v1.0",
    required_observation_history=OCTO_OBSERVATION_HORIZON,
    requires_proprio=False,
    native_proposal_horizon=OCTO_ACTION_HORIZON,
    certified_execute_prefix=1,
    temporal_ensembling=True,
    preprocessing="Pinned Octo v0.1 stack_and_pad source history; image resizing remains caller-owned.",
    normalization=(
        "Pinned Octo v0.1 sample_actions returns normalized [4,7] actions; adapter applies all-seven-dimension "
        "normal unnormalization: normalized * bridge_dataset.action.std + bridge_dataset.action.mean."
    ),
    reset_rule="Clear observation/action histories and bound the cached language task to a new trajectory.",
    rng_rule="Pinned Octo v0.1 native call uses jax.random.PRNGKey(0) for every sample_actions proposal.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation=(
        "This exact native-v0.1 profile is distinct from the pinned AutoEval server's newer unnormalization_statistics "
        "API, which remains unverified/incompatible. Fixture and Gate-A/B evidence are required before named-policy qualification."
    ),
)

OCTO_BASE_V1_CONTRACT = PolicyContract(
    name="Octo-Base v1.0 diagnostic",
    required_observation_history=OCTO_OBSERVATION_HORIZON,
    requires_proprio=False,
    native_proposal_horizon=OCTO_ACTION_HORIZON,
    certified_execute_prefix=1,
    temporal_ensembling=True,
    preprocessing="Same pinned Octo v0.1 stack_and_pad history as Small; diagnostic only, not a primary-matrix policy.",
    normalization="Pinned Octo v0.1 normalized action proposal unnormalizes all seven dimensions with bridge_dataset mean/std.",
    reset_rule="Clear history/action proposal cache and language task per diagnostic trajectory.",
    rng_rule="Pinned Octo v0.1 native source uses jax.random.PRNGKey(0) for each proposal.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation="Separate diagnostic native-v0.1 profile only; AutoEval newer-API replication is separate and unverified.",
)


class _OctoV1PolicyAdapter:
    """Exact local Octo v0.1 sampling path shared by Small and Base.

    This adapter takes one current image on every call and retains its own
    source-compatible history. Supplying a pre-padded history would create a
    second history layer, so it is rejected explicitly.
    """

    contract: PolicyContract

    def __init__(
        self,
        profile: OctoV1PolicyProfile,
        *,
        expected_model_id: str,
        expected_revision: str,
        expected_step: int,
        source_uri: str,
        runtime_factory: Optional[Callable[[], _OctoRuntime]] = None,
        model_factory: Optional[Callable[[OctoV1PolicyProfile, _OctoRuntime], Any]] = None,
    ) -> None:
        self.profile = profile
        self._expected_model_id = expected_model_id
        self._expected_revision = expected_revision
        self._expected_step = expected_step
        self._source_uri = source_uri
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._runtime: Optional[_OctoRuntime] = None
        self._model: Any = None
        self._task: Any = None
        self._task_instruction: Optional[str] = None
        self._observation_history: Deque[Mapping[str, Any]] = deque(maxlen=OCTO_OBSERVATION_HORIZON)
        self._action_history: Deque[Any] = deque(maxlen=OCTO_ACTION_HORIZON)
        self._observation_count = 0
        self._backend_calls = 0
        self.last_report: Optional[OctoActionReport] = None

    def _profile_error(self) -> Optional[str]:
        return self.profile.review_error(
            expected_model_id=self._expected_model_id,
            expected_revision=self._expected_revision,
            expected_step=self._expected_step,
        )
    def _required_snapshot_paths(self) -> Tuple[Path, ...]:
        root = Path(self.profile.local_model_path)
        step = str(self._expected_step)
        return (
            root / "config.json",
            root / "dataset_statistics.json",
            root / "example_batch.msgpack",
            root / step / "default" / "checkpoint",
            root / step / "default" / "commit_success.txt",
            root / step / "commit_success.txt",
        )

    def _expected_large_file_records(self) -> Mapping[str, Tuple[int, str]]:
        if self._expected_model_id == OCTO_SMALL_MODEL_ID:
            return {
                "270000/default/checkpoint": (OCTO_SMALL_CHECKPOINT_BYTES, OCTO_SMALL_CHECKPOINT_SHA256),
                "example_batch.msgpack": (OCTO_SMALL_EXAMPLE_BATCH_BYTES, OCTO_SMALL_EXAMPLE_BATCH_SHA256),
            }
        return {
            "300000/default/checkpoint": (OCTO_BASE_CHECKPOINT_BYTES, OCTO_BASE_CHECKPOINT_SHA256),
            "example_batch.msgpack": (OCTO_BASE_EXAMPLE_BATCH_BYTES, OCTO_BASE_EXAMPLE_BATCH_SHA256),
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _real_binding_error(self) -> Optional[str]:
        """Check configuration-only prerequisites before importing a real runtime."""

        if self._model_factory is not None:
            return None
        if not self.profile.asset_manifest_path:
            return "asset_manifest_path is required for a real Octo load"
        if not Path(self.profile.asset_manifest_path).is_file():
            return "asset_manifest_path does not name a readable immutable asset manifest"
        if not self.profile.source_checkout_path:
            return "source_checkout_path is required for a real Octo load"
        source = Path(self.profile.source_checkout_path)
        if not (source / ".git").is_dir():
            return "source_checkout_path must be a clean checked-out Octo git repository"
        if not self.profile.t5_tokenizer_path:
            return "t5_tokenizer_path is required because the checkpoint config names t5-base"
        tokenizer = Path(self.profile.t5_tokenizer_path)
        missing_t5 = tuple(str(tokenizer / name) for name in OCTO_T5_REQUIRED_FILES if not (tokenizer / name).is_file())
        if missing_t5:
            return "offline t5-base tokenizer snapshot is incomplete: %s" % ", ".join(missing_t5)
        if not self.profile.hf_home:
            return "hf_home is required to bind the source's t5-base identifier to an offline cache"
        if os.environ.get("HF_HOME") != self.profile.hf_home:
            return "HF_HOME must exactly match the profile-pinned offline cache"
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            return "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required for the source's t5-base lookup"
        return None

    def _verify_asset_manifest(self) -> None:
        """Bind staged bytes to the immutable Hub revision before source loading."""

        assert self.profile.asset_manifest_path is not None
        manifest_path = Path(self.profile.asset_manifest_path)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise OctoUnavailableError("Octo asset manifest is not valid JSON: %s" % manifest_path) from error
        if not isinstance(raw, Mapping):
            raise OctoUnavailableError("Octo asset manifest must be a JSON object.")
        if raw.get("schema") != "plumb-octo-assets-v1":
            raise OctoUnavailableError("Octo asset manifest schema must be plumb-octo-assets-v1.")
        if raw.get("model_id") != self._expected_model_id or raw.get("checkpoint_revision") != self._expected_revision:
            raise OctoUnavailableError("Octo asset manifest does not bind the requested immutable model revision.")
        if raw.get("code_revision") != OCTO_V1_SOURCE_COMMIT:
            raise OctoUnavailableError("Octo asset manifest does not bind the released Octo v1 code revision.")
        t5 = raw.get("t5")
        if not isinstance(t5, Mapping) or t5.get("model_id") != OCTO_T5_MODEL_ID or t5.get("revision") != OCTO_T5_MODEL_REVISION:
            raise OctoUnavailableError("Octo asset manifest does not bind the required t5-base tokenizer revision.")
        if t5.get("full_t5_model_weights_downloaded") is not False:
            raise OctoUnavailableError("Octo asset manifest must record that unnecessary T5 model weights were not staged.")
        t5_records = t5.get("files")
        if not isinstance(t5_records, Sequence) or isinstance(t5_records, (str, bytes)):
            raise OctoUnavailableError("Octo asset manifest must record the offline t5-base tokenizer files.")
        t5_by_relative_path = {
            item.get("relative_path"): item
            for item in t5_records
            if isinstance(item, Mapping) and isinstance(item.get("relative_path"), str)
        }
        assert self.profile.t5_tokenizer_path is not None
        tokenizer_root = Path(self.profile.t5_tokenizer_path)
        for relative in OCTO_T5_REQUIRED_FILES:
            path = tokenizer_root / relative
            record = t5_by_relative_path.get(relative)
            if not isinstance(record, Mapping) or not path.is_file():
                raise OctoUnavailableError("Octo asset manifest lacks the offline t5-base tokenizer file %s." % relative)
            if record.get("bytes") != path.stat().st_size or record.get("sha256") != self._sha256(path):
                raise OctoUnavailableError("Octo asset manifest hash/size mismatch for offline t5-base %s." % relative)
        records = raw.get("files")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise OctoUnavailableError("Octo asset manifest files must be a list of file records.")
        by_relative_path = {
            item.get("relative_path"): item
            for item in records
            if isinstance(item, Mapping) and isinstance(item.get("relative_path"), str)
        }
        root = Path(self.profile.local_model_path)
        for path in self._required_snapshot_paths():
            try:
                relative = str(path.relative_to(root))
            except ValueError as error:
                raise OctoUnavailableError("Octo snapshot path escaped its configured root.") from error
            record = by_relative_path.get(relative)
            if not isinstance(record, Mapping):
                raise OctoUnavailableError("Octo asset manifest lacks a record for %s." % relative)
            if not path.is_file():
                raise OctoUnavailableError("Octo immutable snapshot file is missing: %s" % path)
            actual_bytes = path.stat().st_size
            actual_sha256 = self._sha256(path)
            if record.get("bytes") != actual_bytes or record.get("sha256") != actual_sha256:
                raise OctoUnavailableError("Octo asset manifest hash/size mismatch for %s." % relative)
        for relative, (expected_bytes, expected_sha256) in self._expected_large_file_records().items():
            record = by_relative_path.get(relative)
            if not isinstance(record, Mapping):
                raise OctoUnavailableError("Octo asset manifest lacks the immutable LFS record for %s." % relative)
            if record.get("bytes") != expected_bytes or record.get("sha256") != expected_sha256:
                raise OctoUnavailableError("Octo asset manifest does not match the published immutable LFS hash for %s." % relative)

    def _verify_source_checkout(self, runtime: _OctoRuntime) -> None:
        """Reject a mutable/editable package that is not the reviewed source tree."""

        if self._model_factory is not None:
            return
        assert self.profile.source_checkout_path is not None
        source = Path(self.profile.source_checkout_path).resolve()
        try:
            head = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise OctoUnavailableError("Could not verify the checked-out Octo source revision.") from error
        if head != OCTO_V1_SOURCE_COMMIT:
            raise OctoUnavailableError("Octo checked-out source HEAD %r does not match the reviewed v1 commit." % head)
        if dirty:
            raise OctoUnavailableError("Octo checked-out source has tracked modifications; refusing a mutable runtime.")
        try:
            imported_model_path = Path(inspect.getfile(runtime.model_cls)).resolve()
            imported_model_path.relative_to(source)
        except (OSError, TypeError, ValueError) as error:
            raise OctoUnavailableError("Imported OctoModel is not loaded from the profile-pinned checked-out source.") from error

    def capability(self) -> CapabilityResult:
        error = self._profile_error()
        sources = (
            OCTO_MODEL_SOURCE,
            OCTO_GYM_WRAPPERS_SOURCE,
            OCTO_AUTOEVAL_SERVER_SOURCE,
            OCTO_AUTOEVAL_CONTROL_SOURCE,
            OCTO_AUTOEVAL_CLIENT_SOURCE,
            self._source_uri,
        )
        if error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="Octo loading is blocked: %s." % error,
                source_verified=True,
                evidence_uris=sources,
                details={"profile_id": self.profile.profile_id, "local_files_only": self.profile.local_files_only},
            )
        binding_error = self._real_binding_error()
        if binding_error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="Octo real loading is blocked: %s." % binding_error,
                source_verified=True,
                evidence_uris=sources,
                details={"profile_id": self.profile.profile_id, "local_files_only": True},
            )
        missing = tuple(str(path) for path in self._required_snapshot_paths() if not path.is_file())
        if missing and self._model_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Octo local immutable snapshot is incomplete; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=sources,
                details={"local_model_path": self.profile.local_model_path, "missing_files": missing, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local %s uses the Octo v0.1 normalized-action API with explicit bridge_dataset normal "
                "unnormalization and source-compatible temporal-ensemble math. It remains unqualified until local "
                "fixtures and Gate-B evidence pass."
            ) % self.contract.name,
            source_verified=True,
            evidence_uris=sources,
            details={
                "model_id": self.profile.model_id,
                "checkpoint_revision": self.profile.checkpoint_revision,
                "code_revision": self.profile.code_revision,
                "jax_version": self.profile.jax_version,
                "jax_version_full": getattr(self._runtime, "jax_version", None),
                "native_api_profile_id": OCTO_NATIVE_V0_1_PROFILE_ID,
                "autoeval_newer_api_profile_id": OCTO_AUTOEVAL_NEWER_API_PROFILE_ID,
                "autoeval_newer_api_compatible": False,
                "compatibility_profile_id": self.profile.compatibility_profile_id,
                "requires_fixture_requalification": self.profile.jax_version != OCTO_JAX_VERSION,
                "checkpoint_step": self.profile.checkpoint_step,
                "native_proposal_horizon": self.contract.native_proposal_horizon,
                "certified_execute_prefix": self.contract.certified_execute_prefix,
                "temporal_ensembling": True,
                "rng_seed": self.profile.rng_seed,
                "local_files_only": True,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": self.profile.asset_manifest_sha256,
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": self.profile.runtime_lock_sha256,
            },
        )

    def _check_profile(self) -> None:
        error = self._profile_error()
        if error is not None:
            raise OctoUnavailableError("Refusing to load Octo: %s." % error)

    def _load_runtime(self) -> _OctoRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
        else:
            try:
                import jax  # type: ignore
                import numpy as np  # type: ignore
                from octo.model.octo_model import OctoModel  # type: ignore
                from octo.utils.gym_wrappers import stack_and_pad  # type: ignore
            except ImportError as error:
                raise OctoUnavailableError(
                    "Octo dependencies are unavailable. Install the pinned Octo v1 JAX/Flax/Orbax/T5 runtime "
                    "in an isolated policy environment; imports are lazy."
                ) from error
            runtime = _OctoRuntime(
                jax=jax,
                numpy=np,
                model_cls=OctoModel,
                stack_and_pad=stack_and_pad,
                jax_version=str(getattr(jax, "__version__", "")),
            )
        runtime_jax_base = runtime.jax_version.split("+", 1)[0]
        if runtime_jax_base != self.profile.jax_version:
            raise OctoUnavailableError(
                "Octo runtime JAX %r is incompatible; profile requires base version %s." % (runtime.jax_version, self.profile.jax_version)
            )
        if self._model_factory is None and not callable(getattr(runtime.model_cls, "load_pretrained", None)):
            raise OctoUnavailableError("Pinned Octo runtime did not expose OctoModel.load_pretrained().")
        if not callable(runtime.stack_and_pad):
            raise OctoUnavailableError("Pinned Octo runtime did not expose stack_and_pad().")
        if not callable(getattr(runtime.jax, "tree_map", None)):
            raise OctoUnavailableError("Pinned Octo runtime did not expose jax.tree_map().")
        if not callable(getattr(getattr(runtime.jax, "random", None), "PRNGKey", None)):
            raise OctoUnavailableError("Pinned Octo runtime did not expose jax.random.PRNGKey().")
        self._runtime = runtime
        return runtime

    def _ensure_model(self, runtime: _OctoRuntime) -> Any:
        self._check_profile()
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            model = self._model_factory(self.profile, runtime)
        else:
            binding_error = self._real_binding_error()
            if binding_error is not None:
                raise OctoUnavailableError("Refusing to load Octo: %s." % binding_error)
            missing = tuple(str(path) for path in self._required_snapshot_paths() if not path.is_file())
            if missing:
                raise OctoUnavailableError(
                    "Octo local immutable snapshot is incomplete; refusing a remote retrieval. Missing: %s" % ", ".join(missing)
                )
            self._verify_asset_manifest()
            self._verify_source_checkout(runtime)
            model = runtime.model_cls.load_pretrained(str(Path(self.profile.local_model_path)), step=self.profile.checkpoint_step)
        if not callable(getattr(model, "create_tasks", None)) or not callable(getattr(model, "sample_actions", None)):
            raise OctoUnavailableError("Octo local loader did not return the released model interface.")
        self._model = model
        return model

    @staticmethod
    def _check_observation(observation: PolicyObservation) -> Any:
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("Octo requires a nonempty exact task instruction.")
        if len(observation.image_history) != 1 or observation.image_history[0] is None:
            raise PolicyContractError(
                "Octo native-v0.1 profile takes exactly one fresh current image per tick; it maintains its own two-image history."
            )
        return observation.image_history[0]

    def _update_observation_history(self, image: Any, proprio: Optional[Tuple[float, ...]], runtime: _OctoRuntime) -> Mapping[str, Any]:
        current: dict[str, Any] = {"image_primary": image}
        # The released server accepts proprio only when supplied, even though
        # the Octo contract does not require it. Keep that source behavior.
        if proprio is not None:
            current["proprio"] = proprio
        if not self._observation_history:
            self._observation_history.extend([current] * OCTO_OBSERVATION_HORIZON)
            self._observation_count = 1
        else:
            self._observation_history.append(current)
            self._observation_count += 1
        return runtime.stack_and_pad(list(self._observation_history), self._observation_count)

    def _task_for_instruction(self, model: Any, instruction: str) -> Any:
        if self._task is None:
            self._task = model.create_tasks(texts=[instruction])
            self._task_instruction = instruction
        elif instruction != self._task_instruction:
            raise PolicyContractError(
                "Octo native-v0.1 profile binds the language task on its first tick; call reset() before a new instruction."
            )
        return self._task

    @staticmethod
    def _proposal_rows(proposal: Any, numpy_module: Any) -> Tuple[Tuple[float, float, float, float, float, float, float], ...]:
        array = numpy_module.asarray(proposal)
        if tuple(getattr(array, "shape", ())) != (OCTO_ACTION_HORIZON, 7):
            raise PolicyContractError(
                "Octo sample_actions() returned shape %r; expected exactly (4, 7)." % (getattr(array, "shape", None),)
            )
        values = array.tolist()
        try:
            rows = tuple(tuple(float(component) for component in row) for row in values)
        except (TypeError, ValueError) as error:
            raise PolicyContractError("Octo action proposal contains a nonnumeric component.") from error
        if any(len(row) != 7 or not all(math.isfinite(component) for component in row) for row in rows):
            raise PolicyContractError("Octo action proposal must contain four finite 7-D actions.")
        return rows  # type: ignore[return-value]

    @staticmethod
    def _unnormalize_native_v0_1(normalized_proposal: Any, action_statistics: Any, numpy_module: Any) -> Any:
        """Apply the exact normal formula used by the v0.1 inference notebook.

        ``OctoModel.sample_actions`` at :data:`OCTO_V1_SOURCE_COMMIT` has no
        ``unnormalization_statistics`` parameter and returns normalized values.
        The notebook and ``UnnormalizeActionProprio`` both apply this formula
        to every action dimension, including the gripper.
        """

        if tuple(getattr(normalized_proposal, "shape", ())) != (OCTO_ACTION_HORIZON, 7):
            raise PolicyContractError(
                "Octo native-v0.1 normalized proposal has shape %r; expected exactly (4, 7)."
                % (getattr(normalized_proposal, "shape", None),)
            )
        try:
            mean = numpy_module.asarray(action_statistics["mean"])
            std = numpy_module.asarray(action_statistics["std"])
        except (KeyError, TypeError) as error:
            raise OctoUnavailableError(
                "Octo checkpoint action statistics must expose bridge_dataset.action.mean and .std for v0.1 normal unnormalization."
            ) from error
        if tuple(getattr(mean, "shape", ())) != (7,) or tuple(getattr(std, "shape", ())) != (7,):
            raise OctoUnavailableError("Octo bridge_dataset action mean/std must each be exactly 7-D.")
        values = tuple(float(value) for value in mean.tolist()) + tuple(float(value) for value in std.tolist())
        if not all(math.isfinite(value) for value in values):
            raise OctoUnavailableError("Octo bridge_dataset action mean/std must be finite.")
        # Keep this computation on the native JAX array. The v0.1 notebook
        # performs the multiply/add before its later host-side collection;
        # moving it to NumPy first can change the dtype/rounding boundary.
        return normalized_proposal * action_statistics["std"] + action_statistics["mean"]

    def _apply_temporal_ensembling(self, proposal: Any, runtime: _OctoRuntime) -> Tuple[float, float, float, float, float, float, float]:
        """Use the released temporal-ensemble selection formula over physical actions."""

        self._action_history.append(proposal)
        count = len(self._action_history)
        current_predictions = runtime.numpy.stack(
            [
                predicted_actions[index]
                for index, predicted_actions in zip(range(count - 1, -1, -1), self._action_history)
            ]
        )
        weights = runtime.numpy.exp(-self.profile.action_exp_weight * runtime.numpy.arange(count))
        weights = weights / weights.sum()
        action = runtime.numpy.sum(weights[:, None] * current_predictions, axis=0)
        values = runtime.numpy.asarray(action).tolist()
        try:
            selected = tuple(float(component) for component in values)
        except (TypeError, ValueError) as error:
            raise PolicyContractError("Octo temporally-ensembled action contains a nonnumeric component.") from error
        if len(selected) != 7 or not all(math.isfinite(component) for component in selected):
            raise PolicyContractError("Octo temporally-ensembled action must be finite and 7-D.")
        return selected  # type: ignore[return-value]

    def reset(self) -> None:
        """Mirror the released server reset at a trajectory boundary."""

        self._observation_history.clear()
        self._action_history.clear()
        self._observation_count = 0
        self._task = None
        self._task_instruction = None
        self.last_report = None

    def restore_source_state(self, payload: Mapping[str, Any]) -> None:
        """Restore only source task/history state into a fresh wrapper.

        The experimental wall creates a new wrapper at every feedback
        boundary.  Model weights may be owned by a deployment cache, but the
        language task, two-observation history, and temporal proposal history
        must come from the cell's prior persisted state—not from another
        replica request.  This method deliberately accepts no raw reset seed,
        guessed proprioception, or pre-computed executed action.

        ``payload`` is produced by :mod:`plumb.policies.experimental`.  Its
        RGB frames have already been decoded by that layer; preserving them as
        opaque values here keeps this source adapter free of storage/encoding
        policy.
        """

        if self._observation_history or self._action_history or self._task is not None or self._task_instruction is not None:
            raise PolicyContractError("Octo source state may only be restored into a fresh wrapper.")
        if not isinstance(payload, Mapping) or payload.get("schema") != "plumb-octo-v0.1-source-state-v1":
            raise PolicyContractError("Octo source state has an unrecognised schema.")
        instruction = payload.get("task_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PolicyContractError("Octo source state must retain its exact nonempty task instruction.")
        observation_count = payload.get("observation_count")
        if not isinstance(observation_count, int) or observation_count < 1:
            raise PolicyContractError("Octo source state observation_count must be a positive integer.")
        observations = payload.get("observation_history")
        proposals = payload.get("proposal_history")
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise PolicyContractError("Octo source state must carry its real observation history.")
        if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
            raise PolicyContractError("Octo source state must carry its raw proposal history.")
        if not 1 <= len(observations) <= OCTO_OBSERVATION_HORIZON:
            raise PolicyContractError("Octo source state must retain one or two real observations.")
        if len(proposals) > OCTO_ACTION_HORIZON:
            raise PolicyContractError("Octo source state retains more proposal chunks than source temporal ensembling allows.")
        if observation_count < len(observations) or len(proposals) > observation_count:
            raise PolicyContractError("Octo source state observation/proposal counts are inconsistent.")

        restored_observations = []
        for entry in observations:
            if not isinstance(entry, Mapping):
                raise PolicyContractError("Octo source observation history contains a malformed entry.")
            image = entry.get("decoded_rgb")
            proprio = entry.get("proprio")
            if image is None:
                raise PolicyContractError("Octo source history cannot restore an absent RGB observation.")
            if isinstance(proprio, (str, bytes)) or not isinstance(proprio, Sequence):
                raise PolicyContractError("Octo source history cannot restore absent or malformed proprioception.")
            try:
                source_proprio = tuple(float(component) for component in proprio)
            except (TypeError, ValueError) as error:
                raise PolicyContractError("Octo source history proprioception must be numeric.") from error
            if len(source_proprio) not in (7, 8) or not all(math.isfinite(component) for component in source_proprio):
                raise PolicyContractError("Octo source history proprioception must be finite 7-D or 8-D state.")
            if not any(component != 0.0 for component in source_proprio):
                raise PolicyContractError("Octo source history refuses an all-zero invented proprioception.")
            restored_observations.append({"image_primary": image, "proprio": source_proprio})

        runtime = self._load_runtime()
        restored_proposals = []
        for index, proposal in enumerate(proposals):
            rows = self._proposal_rows(runtime.numpy.asarray(proposal), runtime.numpy)
            restored_proposals.append(runtime.numpy.asarray(rows))
        self._observation_history.extend(restored_observations)
        self._action_history.extend(restored_proposals)
        self._observation_count = observation_count
        # ``create_tasks`` remains lazy. Its runtime-specific pytree is never
        # serialised; the exact instruction is recreated before the next
        # sample call by _task_for_instruction().
        self._task = None
        self._task_instruction = instruction
        self.last_report = None

    def predict_with_report(self, observation: PolicyObservation) -> OctoActionReport:
        """Make one local native Octo proposal and return its selected 7-D action."""

        image = self._check_observation(observation)
        runtime = self._load_runtime()
        model = self._ensure_model(runtime)
        observations = self._update_observation_history(image, observation.proprio, runtime)
        task = self._task_for_instruction(model, observation.prompt)
        batched = runtime.jax.tree_map(lambda value: value[None], observations)
        statistics = getattr(model, "dataset_statistics", None)
        try:
            action_statistics = statistics[self.profile.dataset_statistics_key]["action"]
        except (KeyError, TypeError) as error:
            raise OctoUnavailableError(
                "Octo checkpoint did not expose dataset_statistics['bridge_dataset']['action']."
            ) from error
        started = time.perf_counter()
        # Exact Octo v0.1 signature: no ``unnormalization_statistics`` keyword.
        # It reads observations["pad_mask"] when pad_mask is omitted and returns
        # normalized actions, which we unnormalize below using its released
        # notebook/wrapper normal formula.
        normalized_batched = model.sample_actions(batched, task, rng=runtime.jax.random.PRNGKey(self.profile.rng_seed))
        if tuple(getattr(normalized_batched, "shape", ())) != (1, OCTO_ACTION_HORIZON, 7):
            raise PolicyContractError(
                "Octo native-v0.1 sample_actions() returned shape %r; expected exactly (1, 4, 7)."
                % (getattr(normalized_batched, "shape", None),)
            )
        native_unnormalized = self._unnormalize_native_v0_1(normalized_batched[0], action_statistics, runtime.numpy)
        # JAX dispatch is asynchronous. The source formula above remains on
        # the device; this host conversion synchronizes the completed physical
        # action proposal, so it is the latency boundary.
        proposal = runtime.numpy.asarray(native_unnormalized)
        wall_seconds = time.perf_counter() - started
        proposal_rows = self._proposal_rows(proposal, runtime.numpy)
        action = self._apply_temporal_ensembling(proposal, runtime)
        self._backend_calls += 1
        report = OctoActionReport(
            action=action,
            proposal=proposal_rows,
            source_image_timestamp=observation.timestamp,
            backend_calls=1,
            wall_seconds=wall_seconds,
            observation_count=self._observation_count,
            rng_seed=self.profile.rng_seed,
            temporal_ensembling=True,
            action_exp_weight=self.profile.action_exp_weight,
            gripper_transformation=(
                "native-v0.1 normal unnormalization applies mean/std to all seven dimensions, including gripper; "
                "no threshold or binary conversion is applied before the outer AutoEval safety clip to [-2, 2]."
            ),
            normalization=(
                "Octo v0.1 source-native normal unnormalization: normalized_actions * "
                "bridge_dataset.action.std + bridge_dataset.action.mean, all 7 dimensions."
            ),
        )
        self.last_report = report
        return report

    def predict_action(self, observation: PolicyObservation) -> Tuple[float, float, float, float, float, float, float]:
        """Return the released wrapper's selected action, never a padded proposal."""

        return self.predict_with_report(observation).action

    predict = predict_action


class OctoSmallV1Policy(_OctoV1PolicyAdapter):
    """The required Octo-Small v1.0 primary-matrix policy adapter."""

    contract = OCTO_SMALL_V1_CONTRACT

    def __init__(
        self,
        profile: OctoV1PolicyProfile,
        *,
        runtime_factory: Optional[Callable[[], _OctoRuntime]] = None,
        model_factory: Optional[Callable[[OctoV1PolicyProfile, _OctoRuntime], Any]] = None,
    ) -> None:
        super().__init__(
            profile,
            expected_model_id=OCTO_SMALL_MODEL_ID,
            expected_revision=OCTO_SMALL_MODEL_REVISION,
            expected_step=OCTO_SMALL_CHECKPOINT_STEP,
            source_uri=OCTO_SMALL_MODEL_SOURCE,
            runtime_factory=runtime_factory,
            model_factory=model_factory,
        )


class OctoBaseV1Policy(_OctoV1PolicyAdapter):
    """Separate Octo-Base v1.0 diagnostic adapter; not a matrix policy."""

    contract = OCTO_BASE_V1_CONTRACT

    def __init__(
        self,
        profile: OctoV1PolicyProfile,
        *,
        runtime_factory: Optional[Callable[[], _OctoRuntime]] = None,
        model_factory: Optional[Callable[[OctoV1PolicyProfile, _OctoRuntime], Any]] = None,
    ) -> None:
        super().__init__(
            profile,
            expected_model_id=OCTO_BASE_MODEL_ID,
            expected_revision=OCTO_BASE_MODEL_REVISION,
            expected_step=OCTO_BASE_CHECKPOINT_STEP,
            source_uri=OCTO_BASE_MODEL_SOURCE,
            runtime_factory=runtime_factory,
            model_factory=model_factory,
        )
