"""Pinned, local-only MiniVLA Bridge loading.

This is deliberately a narrow adapter for Stanford ILIAD's released
``minivla-vq-bridge-prismatic`` checkpoint.  It follows the release's
Prismatic inference API, but *does not* call its convenience loader: that
loader calls ``torch.load`` without ``weights_only=True``.  The small amount
of construction below follows the same source path while accepting only a
checked, tensor-only state tree.

The required VQ artifact currently has no declared license on its immutable
Hugging Face snapshot.  Consequently the default profile is blocked before it
opens any model or VQ weights.  Setting an arbitrary local path is not an
override: a profile needs recorded license evidence, immutable revisions, and
the known artifact digests before this adapter can even be attempted.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from ..contracts import PolicyContractError, PolicyLoadError


# Immutable sources inspected for this adapter.  The source package is a fork
# with MiniVLA additions; the VQ project is separately pinned because
# ``bridge_vq_extra_action_tokenizer`` imports it at runtime.
MINIVLA_SOURCE_COMMIT = "0822b36227b5a771be4eb2680e34c559734c8fdc"
MINIVLA_VQBET_SOURCE_COMMIT = "09d4851288ca5deaaa1ab367a208e520f8ee9a84"
MINIVLA_MODEL_ID = "Stanford-ILIAD/minivla-vq-bridge-prismatic"
MINIVLA_MODEL_REVISION = "931a637fbfc783220df9c47eb613bf6c9c6e1c4a"
MINIVLA_VQ_MODEL_ID = "Stanford-ILIAD/pretrain_vq"
MINIVLA_VQ_MODEL_REVISION = "30ef2227f97dde1abb6d522ea80af85384235008"
MINIVLA_CHECKPOINT_RELATIVE_PATH = "checkpoints/step-362500-epoch-21-loss=0.2259.pt"
MINIVLA_DATASET_STATISTICS_RELATIVE_PATH = "dataset_statistics.json"
MINIVLA_CONFIG_RELATIVE_PATH = "config.json"
MINIVLA_VQ_RELATIVE_DIRECTORY = "pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512"
MINIVLA_VQ_CHECKPOINT_RELATIVE_PATH = MINIVLA_VQ_RELATIVE_DIRECTORY + "/checkpoints/model.pt"
MINIVLA_VQ_CONFIG_RELATIVE_PATH = MINIVLA_VQ_RELATIVE_DIRECTORY + "/config.json"
MINIVLA_CHECKPOINT_SHA256 = "2b1828f4fb96b0b7a4f3d191fde4ee96938b293c70f8616fb44dd85f5c85cadc"
MINIVLA_DATASET_STATISTICS_SHA256 = "49742ae0009501e1ab283641ab1794a218cee41401cc0c978ed32cc5551adb4f"
MINIVLA_CONFIG_SHA256 = "a241c94667d023877ee11f872bac65c3107edc2b0509fc88a4fa0aa7f3bcebce"
MINIVLA_VQ_CHECKPOINT_SHA256 = "4c3ac8c89f9f092b8b2055b20ed6113a45b7359c5bcd0244f4e725ab5b2a7ce2"
MINIVLA_VQ_CONFIG_SHA256 = "2abad97e073f6a5ec4c0c4a0a28936d36a8d675f87d3ea690233c144be89b2eb"
MINIVLA_TRANSFORMERS_VERSION = "4.40.1"
MINIVLA_TORCH_VERSION = "2.2.0"
MINIVLA_TORCHVISION_VERSION = "0.17.0"
MINIVLA_TOKENIZERS_VERSION = "0.19.1"
MINIVLA_TIMM_VERSION = "0.9.10"
MINIVLA_BRIDGE_UNNORM_KEY = "bridge_dataset"
MINIVLA_VQ_INPUT_HORIZON = 8
MINIVLA_VQ_FUTURE_ACTION_HORIZON = 7

MINIVLA_MODEL_SOURCE = (
    "https://github.com/Stanford-ILIAD/openvla-mini/blob/"
    + MINIVLA_SOURCE_COMMIT
    + "/prismatic/models/load.py#L130-L257"
)
MINIVLA_ACTION_SOURCE = (
    "https://github.com/Stanford-ILIAD/openvla-mini/blob/"
    + MINIVLA_SOURCE_COMMIT
    + "/prismatic/models/vlas/openvla.py#L37-L121"
)
MINIVLA_VQ_TOKENIZER_SOURCE = (
    "https://github.com/Stanford-ILIAD/openvla-mini/blob/"
    + MINIVLA_SOURCE_COMMIT
    + "/prismatic/vla/action_tokenizer.py#L110-L180"
)
MINIVLA_VQBET_SOURCE = (
    "https://github.com/jayLEE0301/vq_bet_official/blob/"
    + MINIVLA_VQBET_SOURCE_COMMIT
    + "/vqvae/vqvae.py#L44-L136"
)
MINIVLA_MODEL_ARTIFACT_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    MINIVLA_MODEL_ID,
    MINIVLA_MODEL_REVISION,
)
MINIVLA_VQ_ARTIFACT_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    MINIVLA_VQ_MODEL_ID,
    MINIVLA_VQ_MODEL_REVISION,
)
MINIVLA_VQ_LICENSE_BLOCKER = (
    "Stanford-ILIAD/pretrain_vq at revision "
    "30ef2227f97dde1abb6d522ea80af85384235008 has cardData: null and declares no license. "
    "Do not acquire, copy, convert, redistribute, or use its VQ weights until the publisher supplies "
    "terms and those terms are recorded in the asset manifest."
)
MINIVLA_AUXILIARY_ASSET_BLOCKER = (
    "The reviewed MiniVLA source constructs DINO and SigLIP with timm.create_model(..., pretrained=True) and "
    "constructs its Qwen2.5 backbone through a source helper. HF_HUB_OFFLINE does not prove that timm/torch hub "
    "cannot fetch those auxiliary weights. No pinned local DINO, SigLIP, and Qwen cache bindings with checksums "
    "were supplied, so real MiniVLA construction is blocked rather than attempting those helpers."
)


class MiniVLAUnavailableError(PolicyLoadError):
    """The reviewed local MiniVLA runtime or required evidence is unavailable."""


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_without_build(value: object) -> str:
    return str(value).split("+", 1)[0]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _checked_git_checkout(root_value: str, revision: str, anchors: Tuple[str, ...], *, label: str) -> Path:
    """Verify a local, clean checkout before trusting a declared code revision."""

    root = Path(root_value)
    if not root.is_dir():
        raise MiniVLAUnavailableError("Reviewed %s source checkout is absent at %s." % (label, root))
    try:
        head = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MiniVLAUnavailableError("Could not inspect reviewed %s source checkout." % label) from error
    if head.returncode != 0 or head.stdout.strip() != revision:
        raise MiniVLAUnavailableError("%s source checkout HEAD does not match reviewed commit %s." % (label, revision))
    if status.returncode != 0 or status.stdout.strip():
        raise MiniVLAUnavailableError("%s source checkout is dirty; refusing mutable code." % label)
    missing = tuple(anchor for anchor in anchors if not (root / anchor).is_file())
    if missing:
        raise MiniVLAUnavailableError("%s source checkout lacks reviewed files: %s." % (label, ", ".join(missing)))
    return root.resolve()


def _is_tensor_safe(value: Any, torch: Any) -> bool:
    """Accept only a data-only PyTorch checkpoint tree.

    ``weights_only=True`` is the primary deserialization protection.  This
    second check prevents silently forwarding arbitrary Python objects into
    model or optimizer ``load_state_dict`` implementations.
    """

    if bool(getattr(torch, "is_tensor", lambda _: False)(value)):
        return True
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, (str, int)) and _is_tensor_safe(item, torch) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_is_tensor_safe(item, torch) for item in value)
    return False


@contextlib.contextmanager
def _offline_hub_environment() -> Iterator[None]:
    """Keep source loaders offline even where their older APIs lack a flag."""

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class MiniVLAPolicyProfile:
    """Exact local assets required by the released MiniVLA source path."""

    profile_id: str
    local_checkpoint_path: str
    local_config_path: str
    local_dataset_statistics_path: str
    local_vq_checkpoint_path: str
    local_vq_config_path: str
    local_source_path: Optional[str] = None
    local_vq_source_path: Optional[str] = None
    model_revision: str = MINIVLA_MODEL_REVISION
    vq_revision: str = MINIVLA_VQ_MODEL_REVISION
    code_revision: str = MINIVLA_SOURCE_COMMIT
    vq_code_revision: str = MINIVLA_VQBET_SOURCE_COMMIT
    checkpoint_sha256: str = MINIVLA_CHECKPOINT_SHA256
    config_sha256: str = MINIVLA_CONFIG_SHA256
    dataset_statistics_sha256: str = MINIVLA_DATASET_STATISTICS_SHA256
    vq_checkpoint_sha256: str = MINIVLA_VQ_CHECKPOINT_SHA256
    vq_config_sha256: str = MINIVLA_VQ_CONFIG_SHA256
    vq_license_status: str = "unresolved"
    vq_license_evidence_uri: Optional[str] = None
    transformers_version: str = MINIVLA_TRANSFORMERS_VERSION
    torch_version: str = MINIVLA_TORCH_VERSION
    torchvision_version: str = MINIVLA_TORCHVISION_VERSION
    tokenizers_version: str = MINIVLA_TOKENIZERS_VERSION
    timm_version: str = MINIVLA_TIMM_VERSION
    unnorm_key: str = MINIVLA_BRIDGE_UNNORM_KEY
    device: str = "cuda:0"
    local_files_only: bool = True
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        required_paths = (
            self.local_checkpoint_path,
            self.local_config_path,
            self.local_dataset_statistics_path,
            self.local_vq_checkpoint_path,
            self.local_vq_config_path,
        )
        if not all(isinstance(path, str) and path for path in required_paths):
            return "all local MiniVLA/VQ artifact paths are required; Hub fetching is disabled"
        if self.model_revision != MINIVLA_MODEL_REVISION or not _immutable_revision(self.model_revision):
            return "model_revision must be the pinned MiniVLA release revision"
        if self.vq_revision != MINIVLA_VQ_MODEL_REVISION or not _immutable_revision(self.vq_revision):
            return "vq_revision must be the pinned MiniVLA VQ release revision"
        if self.code_revision != MINIVLA_SOURCE_COMMIT:
            return "code_revision must be the reviewed MiniVLA source commit"
        if self.vq_code_revision != MINIVLA_VQBET_SOURCE_COMMIT:
            return "vq_code_revision must be the reviewed VQ-Bet source commit"
        expected_hashes = (
            (self.checkpoint_sha256, MINIVLA_CHECKPOINT_SHA256),
            (self.config_sha256, MINIVLA_CONFIG_SHA256),
            (self.dataset_statistics_sha256, MINIVLA_DATASET_STATISTICS_SHA256),
            (self.vq_checkpoint_sha256, MINIVLA_VQ_CHECKPOINT_SHA256),
            (self.vq_config_sha256, MINIVLA_VQ_CONFIG_SHA256),
        )
        if any(actual != expected for actual, expected in expected_hashes):
            return "MiniVLA profile artifact SHA-256 values must match the pinned release files"
        if self.vq_license_status != "verified" or not self.vq_license_evidence_uri:
            return MINIVLA_VQ_LICENSE_BLOCKER
        if self.transformers_version != MINIVLA_TRANSFORMERS_VERSION:
            return "transformers_version must be exactly %s" % MINIVLA_TRANSFORMERS_VERSION
        if self.torch_version != MINIVLA_TORCH_VERSION:
            return "torch_version must be exactly %s" % MINIVLA_TORCH_VERSION
        if self.torchvision_version != MINIVLA_TORCHVISION_VERSION:
            return "torchvision_version must be exactly %s" % MINIVLA_TORCHVISION_VERSION
        if self.tokenizers_version != MINIVLA_TOKENIZERS_VERSION:
            return "tokenizers_version must be exactly %s" % MINIVLA_TOKENIZERS_VERSION
        if self.timm_version != MINIVLA_TIMM_VERSION:
            return "timm_version must be exactly %s" % MINIVLA_TIMM_VERSION
        if self.unnorm_key != MINIVLA_BRIDGE_UNNORM_KEY:
            return "MiniVLA Bridge checkpoint must use unnorm_key='bridge_dataset'"
        if not self.local_files_only:
            return "MiniVLA loader is local-only; network retrieval is not permitted"
        if not self.device:
            return "device is required"
        return None

    def auxiliary_assets_error(self) -> str:
        """Why a real source construction remains deliberately unavailable.

        This is separate from the VQ license check so a later license fix
        cannot accidentally turn on potentially networked timm/HF helper
        calls. A subsequent implementation must replace this blocker with
        source-supported, local-path cache bindings and hashes for all three
        auxiliary components.
        """

        return MINIVLA_AUXILIARY_ASSET_BLOCKER


@dataclass(frozen=True)
class MiniVLAActionReport:
    """One source-native MiniVLA prediction, with the VQ limitation explicit."""

    action: Tuple[float, float, float, float, float, float, float]
    unnorm_key: str
    source_image_timestamp: Optional[float]
    backend_calls: int
    wall_seconds: Optional[float]
    vq_input_horizon: int
    vq_future_action_horizon: int
    returned_action_count: int


@dataclass(frozen=True)
class _MiniVLARuntime:
    """Lazy source imports. Factories exist only for isolated fixture tests."""

    torch: Any
    numpy: Any
    image_fromarray: Callable[[Any], Any]
    model_config_cls: Any
    vision_factory: Callable[..., Any]
    llm_factory: Callable[..., Any]
    vla_cls: Any
    vqvae_cls: Any
    transformers_version: str
    torch_version: str
    torchvision_version: str
    tokenizers_version: str
    timm_version: str


class _TensorSafeVQActionDecoder:
    """Source-equivalent VQ decoding without VQ-Bet's unsafe ``torch.load`` path."""

    def __init__(self, tokenizer: Any, vqvae: Any, torch: Any, numpy_module: Any, device: str) -> None:
        self.vq_vae = vqvae
        self._torch = torch
        self._numpy = numpy_module
        self._device = device
        self.n_bins = int(vqvae.vqvae_n_embed)
        # The checked MiniVLA config selects ``use_extra=True`` for a Qwen2.5
        # tokenizer.  This is exactly the branch used by VQActionTokenizer.
        self.tokenizer_len = int(len(tokenizer))

    def decode_token_ids_to_actions(self, action_token_ids: Any) -> Any:
        code_ids = self.tokenizer_len - 1 - self._numpy.asarray(action_token_ids)
        initial_shape = code_ids.shape
        code_ids = self._numpy.clip(code_ids, 0, self.n_bins - 1)
        code_ids = self._torch.from_numpy(code_ids).to(self._device).reshape(-1, self.vq_vae.vqvae_groups)
        latent = self.vq_vae.draw_code_forward(code_ids)
        actions = self.vq_vae.get_action_from_latent(latent)
        # This is deliberately source-faithful.  The upstream tokenizer decodes
        # an H=8 latent (seven future elements) then returns only [:, 0].
        if code_ids.shape[0] == 1 and len(initial_shape) == 1:
            return actions[0, 0]
        return actions[:, 0]


MINIVLA_SOURCE_CONTRACT = PolicyContract(
    name="MiniVLA source-release single-action boundary",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=1,
    temporal_ensembling=False,
    preprocessing="Released Prismatic MiniVLA predict_action(PIL RGB image, exact instruction, unnorm_key='bridge_dataset').",
    normalization="The released model's VQ decoder and bridge_dataset q01/q99 normalization execute inside source predict_action.",
    reset_rule="No source action cache; a fresh image is required for every call.",
    rng_rule="Source generation defaults are preserved; fixture/Gate-B evidence is still required.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation=(
        "The released VQActionTokenizer decodes an H=8 latent but returns only the first 7-D action. "
        "It does not establish the spec's asserted seven-action exposed proposal or its execute prefix."
    ),
)


class MiniVLAPolicyAdapter:
    """Exact, local-only MiniVLA call path with strict tensor-safe state loading."""

    contract = MINIVLA_SOURCE_CONTRACT

    def __init__(
        self,
        profile: MiniVLAPolicyProfile,
        *,
        runtime_factory: Optional[Callable[[], _MiniVLARuntime]] = None,
        model_factory: Optional[Callable[[MiniVLAPolicyProfile, _MiniVLARuntime], Any]] = None,
    ) -> None:
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._runtime: Optional[_MiniVLARuntime] = None
        self._model: Any = None
        self._backend_calls = 0
        self.last_report: Optional[MiniVLAActionReport] = None

    def _artifact_paths(self) -> Tuple[Tuple[Path, str], ...]:
        return (
            (Path(self.profile.local_checkpoint_path), self.profile.checkpoint_sha256),
            (Path(self.profile.local_config_path), self.profile.config_sha256),
            (Path(self.profile.local_dataset_statistics_path), self.profile.dataset_statistics_sha256),
            (Path(self.profile.local_vq_checkpoint_path), self.profile.vq_checkpoint_sha256),
            (Path(self.profile.local_vq_config_path), self.profile.vq_config_sha256),
        )

    def _source_paths_missing(self) -> Tuple[str, ...]:
        missing = []
        if not self.profile.local_source_path:
            missing.append("local_source_path (clean Stanford-ILIAD/openvla-mini checkout)")
        if not self.profile.local_vq_source_path:
            missing.append("local_vq_source_path (clean jayLEE0301/vq_bet_official checkout)")
        return tuple(missing)

    def capability(self) -> CapabilityResult:
        error = self.profile.review_error()
        sources = (
            MINIVLA_MODEL_SOURCE,
            MINIVLA_ACTION_SOURCE,
            MINIVLA_VQ_TOKENIZER_SOURCE,
            MINIVLA_VQBET_SOURCE,
            MINIVLA_MODEL_ARTIFACT_SOURCE,
            MINIVLA_VQ_ARTIFACT_SOURCE,
        )
        if error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="MiniVLA loading is blocked: %s." % error,
                source_verified=True,
                evidence_uris=sources,
                details={
                    "profile_id": self.profile.profile_id,
                    "vq_license_status": self.profile.vq_license_status,
                    "vq_license_blocker": MINIVLA_VQ_LICENSE_BLOCKER,
                    "local_files_only": self.profile.local_files_only,
                },
            )
        source_paths_missing = self._source_paths_missing()
        if source_paths_missing and self._runtime_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="MiniVLA reviewed local source checkouts are required before import; no source was fetched.",
                source_verified=True,
                evidence_uris=sources,
                details={"missing_source_paths": source_paths_missing, "local_files_only": True},
            )
        if self._runtime_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="MiniVLA loading is blocked: %s" % self.profile.auxiliary_assets_error(),
                source_verified=True,
                evidence_uris=sources,
                details={"auxiliary_asset_blocker": MINIVLA_AUXILIARY_ASSET_BLOCKER, "local_files_only": True},
            )
        missing = tuple(str(path) for path, _ in self._artifact_paths() if not path.is_file())
        if missing and self._model_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="MiniVLA local immutable artifacts are incomplete; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=sources,
                details={"missing_files": missing, "local_files_only": True},
            )
        if self._model_factory is not None:
            return CapabilityResult(
                status=CapabilityStatus.READY_UNQUALIFIED,
                reason=(
                    "A MiniVLA model factory is injected for an isolated fixture; it is not a source-loaded or "
                    "validated MiniVLA deployment and cannot clear any policy gate."
                ),
                source_verified=True,
                evidence_uris=sources,
                details={"fixture_model_factory": True, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local MiniVLA source loading is configured with tensor-only checkpoint validation. "
                "It remains unqualified: the source VQ decoder returns its first action only, so the seven-action "
                "proposal/executed-prefix contract still needs source reconciliation and fixtures."
            ),
            source_verified=True,
            evidence_uris=sources,
            details={
                "model_id": MINIVLA_MODEL_ID,
                "model_revision": self.profile.model_revision,
                "vq_model_id": MINIVLA_VQ_MODEL_ID,
                "vq_revision": self.profile.vq_revision,
                "code_revision": self.profile.code_revision,
                "vq_code_revision": self.profile.vq_code_revision,
                "unnorm_key": self.profile.unnorm_key,
                "vq_input_horizon": MINIVLA_VQ_INPUT_HORIZON,
                "vq_future_action_horizon": MINIVLA_VQ_FUTURE_ACTION_HORIZON,
                "returned_action_count": 1,
                "local_files_only": True,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": self.profile.asset_manifest_sha256,
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": self.profile.runtime_lock_sha256,
            },
        )

    def _check_profile(self) -> None:
        error = self.profile.review_error()
        if error is not None:
            raise MiniVLAUnavailableError("Refusing to load MiniVLA: %s." % error)

    def _load_runtime(self) -> _MiniVLARuntime:
        if self._runtime is not None:
            return self._runtime
        source_root: Optional[Path] = None
        vq_source_root: Optional[Path] = None
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
        else:
            source_paths_missing = self._source_paths_missing()
            if source_paths_missing:
                raise MiniVLAUnavailableError(
                    "MiniVLA requires reviewed local source checkouts before import: %s."
                    % ", ".join(source_paths_missing)
                )
            source_root = _checked_git_checkout(
                str(self.profile.local_source_path),
                MINIVLA_SOURCE_COMMIT,
                ("prismatic/models/load.py", "prismatic/models/vlas/openvla.py", "prismatic/vla/action_tokenizer.py"),
                label="MiniVLA",
            )
            vq_source_root = _checked_git_checkout(
                str(self.profile.local_vq_source_path),
                MINIVLA_VQBET_SOURCE_COMMIT,
                ("vqvae/vqvae.py",),
                label="VQ-Bet",
            )
            try:
                import numpy as np  # type: ignore
                import timm  # type: ignore
                import tokenizers  # type: ignore
                import torch  # type: ignore
                import torchvision  # type: ignore
                import transformers  # type: ignore
                from PIL import Image  # type: ignore
                from prismatic.conf import ModelConfig  # type: ignore
                from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform  # type: ignore
                from prismatic.models.vlas import OpenVLA  # type: ignore
                from vqvae.vqvae import VqVae  # type: ignore
            except ImportError as error:
                raise MiniVLAUnavailableError(
                    "MiniVLA dependencies are unavailable. Install the pinned MiniVLA/Prismatic/VQ-Bet runtime "
                    "in an isolated policy image; imports are lazy."
                ) from error
            runtime = _MiniVLARuntime(
                torch=torch,
                numpy=np,
                image_fromarray=Image.fromarray,
                model_config_cls=ModelConfig,
                vision_factory=get_vision_backbone_and_transform,
                llm_factory=get_llm_backbone_and_tokenizer,
                vla_cls=OpenVLA,
                vqvae_cls=VqVae,
                transformers_version=str(getattr(transformers, "__version__", "")),
                torch_version=str(getattr(torch, "__version__", "")),
                torchvision_version=str(getattr(torchvision, "__version__", "")),
                tokenizers_version=str(getattr(tokenizers, "__version__", "")),
                timm_version=str(getattr(timm, "__version__", "")),
            )
            # Do not accept a site-package or a different checkout merely
            # because a profile repeats the desired commit string.
            if not _is_within(Path(inspect.getfile(ModelConfig)), source_root):
                raise MiniVLAUnavailableError("Imported prismatic ModelConfig does not resolve inside local_source_path.")
            if not _is_within(Path(inspect.getfile(OpenVLA)), source_root):
                raise MiniVLAUnavailableError("Imported MiniVLA OpenVLA class does not resolve inside local_source_path.")
            if not _is_within(Path(inspect.getfile(VqVae)), vq_source_root):
                raise MiniVLAUnavailableError("Imported VqVae does not resolve inside local_vq_source_path.")
        versions = (
            ("Transformers", runtime.transformers_version, MINIVLA_TRANSFORMERS_VERSION),
            ("Torch", _version_without_build(runtime.torch_version), MINIVLA_TORCH_VERSION),
            ("Torchvision", _version_without_build(runtime.torchvision_version), MINIVLA_TORCHVISION_VERSION),
            ("Tokenizers", runtime.tokenizers_version, MINIVLA_TOKENIZERS_VERSION),
            ("timm", runtime.timm_version, MINIVLA_TIMM_VERSION),
        )
        for name, actual, expected in versions:
            if actual != expected:
                raise MiniVLAUnavailableError("MiniVLA runtime %s %r is incompatible; exactly %s is required." % (name, actual, expected))
        self._runtime = runtime
        return runtime

    def _checked_tensor_load(self, path: Path, expected_sha256: str, runtime: _MiniVLARuntime) -> Mapping[str, Any]:
        if not path.is_file():
            raise MiniVLAUnavailableError("Expected local immutable MiniVLA artifact at %s." % path)
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise MiniVLAUnavailableError("SHA-256 mismatch for %s; refusing to deserialize it." % path)
        try:
            loaded = runtime.torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError as error:
            raise MiniVLAUnavailableError(
                "This Torch runtime does not support torch.load(..., weights_only=True); refusing unsafe MiniVLA deserialization."
            ) from error
        except Exception as error:
            raise MiniVLAUnavailableError("Tensor-safe loading failed for %s." % path) from error
        if not isinstance(loaded, Mapping) or not _is_tensor_safe(loaded, runtime.torch):
            raise MiniVLAUnavailableError("MiniVLA artifact %s is not a tensor-only state tree." % path)
        return loaded

    def _validate_small_artifact(self, path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise MiniVLAUnavailableError("Expected local immutable MiniVLA artifact at %s." % path)
        if _sha256_file(path) != expected_sha256:
            raise MiniVLAUnavailableError("SHA-256 mismatch for %s." % path)

    def _build_source_model(self, runtime: _MiniVLARuntime) -> Any:
        self._check_profile()
        auxiliary_error = self.profile.auxiliary_assets_error()
        if auxiliary_error:
            raise MiniVLAUnavailableError("Refusing to construct MiniVLA: %s" % auxiliary_error)
        # The source-matched, tensor-safe construction below is intentionally
        # retained for the future reviewed support-asset implementation. It is
        # unreachable until that implementation removes the explicit blocker
        # above and binds local DINO/SigLIP/Qwen artifacts without downloads.
        checkpoint_path = Path(self.profile.local_checkpoint_path)
        config_path = Path(self.profile.local_config_path)
        statistics_path = Path(self.profile.local_dataset_statistics_path)
        vq_checkpoint_path = Path(self.profile.local_vq_checkpoint_path)
        vq_config_path = Path(self.profile.local_vq_config_path)
        self._validate_small_artifact(config_path, self.profile.config_sha256)
        self._validate_small_artifact(statistics_path, self.profile.dataset_statistics_sha256)
        self._validate_small_artifact(vq_config_path, self.profile.vq_config_sha256)
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                vla_config = json.load(stream)["vla"]
            with statistics_path.open("r", encoding="utf-8") as stream:
                statistics = json.load(stream)
            with vq_config_path.open("r", encoding="utf-8") as stream:
                vq_config = json.load(stream)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise MiniVLAUnavailableError("MiniVLA JSON configuration/statistics are malformed.") from error
        if vla_config.get("action_tokenizer") != "bridge_vq_extra_action_tokenizer":
            raise MiniVLAUnavailableError("Pinned MiniVLA config does not select bridge_vq_extra_action_tokenizer.")
        if vla_config.get("base_vlm") != "prism-qwen25-extra-dinosiglip-224px+0_5b":
            raise MiniVLAUnavailableError("Pinned MiniVLA config does not identify its reviewed Qwen2.5 base VLM.")
        if vq_config.get("input_dim_w") != 7 or vq_config.get("input_dim_h") != MINIVLA_VQ_INPUT_HORIZON:
            raise MiniVLAUnavailableError("Pinned MiniVLA VQ config must have action dimension 7 and input horizon 8.")
        if vq_config.get("vqvae_n_embed") != 256 or vq_config.get("vqvae_groups") != 7:
            raise MiniVLAUnavailableError("Pinned MiniVLA VQ codebook/groups do not match the released configuration.")
        if self.profile.unnorm_key not in statistics:
            raise MiniVLAUnavailableError("MiniVLA dataset statistics do not contain bridge_dataset normalization.")

        # Mirror the reviewed source construction, except that both .pt files
        # are opened via the explicit tensor-only loader above.
        with _offline_hub_environment():
            model_config = runtime.model_config_cls.get_choice_class(vla_config["base_vlm"])()
            vision_backbone, _ = runtime.vision_factory(
                model_config.vision_backbone_id,
                model_config.image_resize_strategy,
                1,
            )
            llm_backbone, _ = runtime.llm_factory(
                model_config.llm_backbone_id,
                llm_max_length=model_config.llm_max_length,
                hf_token=None,
                inference_mode=True,
            )
            vq_options = dict(vq_config)
            vq_options["load_dir"] = None
            vq_options["eval"] = True
            vq_options["device"] = self.profile.device
            vqvae = runtime.vqvae_cls(**vq_options)
            vq_state = self._checked_tensor_load(vq_checkpoint_path, self.profile.vq_checkpoint_sha256, runtime)
            vqvae.load_state_dict(vq_state)
            decoder = _TensorSafeVQActionDecoder(llm_backbone.get_tokenizer(), vqvae, runtime.torch, runtime.numpy, self.profile.device)
            model = runtime.vla_cls(
                model_config.model_id,
                vision_backbone,
                llm_backbone,
                arch_specifier=model_config.arch_specifier,
                freeze_weights=True,
                norm_stats=statistics,
                action_tokenizer=decoder,
            )
            checkpoint = self._checked_tensor_load(checkpoint_path, self.profile.checkpoint_sha256, runtime)
        model_state = checkpoint.get("model")
        if not isinstance(model_state, Mapping):
            raise MiniVLAUnavailableError("MiniVLA checkpoint has no tensor-only 'model' state mapping.")
        expected_components = {"projector", "llm_backbone", "vision_backbone"}
        if not {"projector", "llm_backbone"}.issubset(model_state) or not set(model_state).issubset(expected_components):
            raise MiniVLAUnavailableError("MiniVLA checkpoint component coverage is not the reviewed source schema.")
        for component in ("projector", "llm_backbone"):
            value = model_state[component]
            if not isinstance(value, Mapping):
                raise MiniVLAUnavailableError("MiniVLA checkpoint component %s is not a state mapping." % component)
            getattr(model, component).load_state_dict(value, strict=True)
        if "vision_backbone" in model_state:
            value = model_state["vision_backbone"]
            if not isinstance(value, Mapping):
                raise MiniVLAUnavailableError("MiniVLA vision state is not a state mapping.")
            model.vision_backbone.load_state_dict(value, strict=True)
        model.requires_grad_(False)
        model.eval()
        model.vision_backbone.to(dtype=model.vision_backbone.half_precision_dtype)
        model.llm_backbone.to(dtype=model.llm_backbone.half_precision_dtype)
        model.to(dtype=model.llm_backbone.half_precision_dtype)
        model.to(self.profile.device)
        if not callable(getattr(model, "predict_action", None)):
            raise MiniVLAUnavailableError("Reviewed MiniVLA source construction did not expose predict_action().")
        return model

    def _ensure_model(self, runtime: _MiniVLARuntime) -> Any:
        self._check_profile()
        if self._model is None:
            self._model = self._model_factory(self.profile, runtime) if self._model_factory is not None else self._build_source_model(runtime)
            if not callable(getattr(self._model, "predict_action", None)):
                raise MiniVLAUnavailableError("MiniVLA loader did not return the source predict_action interface.")
        return self._model

    @staticmethod
    def _check_observation(observation: PolicyObservation) -> Any:
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("MiniVLA requires a nonempty exact task instruction.")
        if len(observation.image_history) != 1 or observation.image_history[0] is None:
            raise PolicyContractError("MiniVLA requires exactly one fresh current image; do not supply stale or padded history.")
        if observation.proprio is not None:
            raise PolicyContractError("Released MiniVLA Bridge predict_action does not take proprioception.")
        return observation.image_history[0]

    @staticmethod
    def _single_action(raw_action: Any) -> Tuple[float, float, float, float, float, float, float]:
        if isinstance(raw_action, (str, bytes)):
            raise PolicyContractError("MiniVLA predict_action() returned text instead of one numeric 7-D action.")
        try:
            values = tuple(float(value) for value in raw_action)
        except (TypeError, ValueError) as error:
            raise PolicyContractError("MiniVLA predict_action() returned a nonnumeric action.") from error
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise PolicyContractError("MiniVLA source boundary must return exactly one finite 7-D action.")
        return values  # type: ignore[return-value]

    def reset(self) -> None:
        """Discard report state; the released source exposes no action-proposal cache."""

        self.last_report = None

    def predict_with_report(self, observation: PolicyObservation) -> MiniVLAActionReport:
        image = self._check_observation(observation)
        # Keep license/source verification ahead of all ML imports and before
        # any potential checkpoint access.
        self._check_profile()
        runtime = self._load_runtime()
        model = self._ensure_model(runtime)
        try:
            source_image = runtime.image_fromarray(image).convert("RGB")
        except Exception as error:
            raise PolicyContractError("MiniVLA requires an array image accepted by PIL.Image.fromarray.") from error
        started = time.perf_counter()
        raw_action = model.predict_action(source_image, observation.prompt, unnorm_key=self.profile.unnorm_key)
        wall_seconds = time.perf_counter() - started
        action = self._single_action(raw_action)
        self._backend_calls += 1
        report = MiniVLAActionReport(
            action=action,
            unnorm_key=self.profile.unnorm_key,
            source_image_timestamp=observation.timestamp,
            backend_calls=1,
            wall_seconds=wall_seconds,
            vq_input_horizon=MINIVLA_VQ_INPUT_HORIZON,
            vq_future_action_horizon=MINIVLA_VQ_FUTURE_ACTION_HORIZON,
            returned_action_count=1,
        )
        self.last_report = report
        return report

    def predict_action(self, observation: PolicyObservation) -> Tuple[float, float, float, float, float, float, float]:
        return self.predict_with_report(observation).action

    predict = predict_action
