"""Pinned, local-only OpenPiZero Bridge proposal adapter.

The adapter follows the exact AutoEval ``OpenPiZero`` wrapper at
``3ea3ff44…``: one current RGB image and an eight-value Bridge EEF pose are
converted to a fresh seven-value proprio input, the PiZero model proposes four
actions, and the wrapper's ``bounds`` action normalization is applied to every
proposal row.  The released AutoEval control path wraps those chunks in an
external Octo temporal-ensemble wrapper whose executed prefix is not pinned in
this repository.  Consequently this module exposes the native four-action
proposal but intentionally does not certify or select an executed prefix.

No ML framework is imported on module import.  A real load requires a clean
source checkout, an immutable staged asset manifest, local PaliGemma support
files with accepted terms, and ``torch.load(weights_only=True)`` strict state
dict coverage.  It never calls a Hub, uses an unsafe pickle path, or invents an
action when those requirements are absent.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from ..contracts import AUTOEVAL_POLICY_SOURCE_COMMIT, OPEN_PI_ZERO_SOURCE_COMMIT, PolicyContractError, PolicyLoadError


OPENPI_MODEL_ID = "allenzren/open-pi-zero"
OPENPI_SOURCE_COMMIT = OPEN_PI_ZERO_SOURCE_COMMIT
OPENPI_BRIDGE_BETA_CHECKPOINT = "bridge_beta_step19296_2024-12-26_22-30_42.pt"
OPENPI_PALIGEMMA_MODEL_ID = "google/paligemma-3b-pt-224"
OPENPI_TORCH_VERSION = "2.5.0"
OPENPI_TRANSFORMERS_VERSION = "4.47.1"
OPENPI_PYTHON_MINOR = (3, 10)
OPENPI_ACTION_HORIZON = 4
OPENPI_ACTION_DIM = 7
OPENPI_SOURCE_PROPRIO_DIM = 8
OPENPI_MODEL_PROPRIO_DIM = 7
OPENPI_BRIDGE_EEF_QUATERNION_WXYZ = "bridge_eef_quaternion_wxyz"
OPENPI_QUATERNION_NORM_TOLERANCE = 1e-3
OPENPI_EVAL_CONFIG = "config/eval/bridge.yaml"
OPENPI_REQUIRED_PALIGEMMA_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "config.json",
)

OPENPI_SOURCE = "https://github.com/allenzren/open-pi-zero/tree/" + OPEN_PI_ZERO_SOURCE_COMMIT
OPENPI_PI_ZERO_SOURCE = (
    "https://github.com/allenzren/open-pi-zero/blob/"
    + OPEN_PI_ZERO_SOURCE_COMMIT
    + "/src/model/vla/pizero.py#L416-L490"
)
OPENPI_PROCESSING_SOURCE = (
    "https://github.com/allenzren/open-pi-zero/blob/"
    + OPEN_PI_ZERO_SOURCE_COMMIT
    + "/src/model/vla/processing.py"
)
OPENPI_AUTOEVAL_POLICY_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/auto_eval/robot/policy.py#L264-L451"
)
OPENPI_AUTOEVAL_CONTROL_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/run_eval.py#L354-L365"
)
OPENPI_PALIGEMMA_SOURCE = "https://huggingface.co/google/paligemma-3b-pt-224"


class OpenPiUnavailableError(PolicyLoadError):
    """The immutable OpenPiZero local runtime/assets are unavailable or unsafe."""


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class OpenPiZeroPolicyProfile:
    """Reviewed local inputs for one source-bound OpenPiZero Bridge proposal.

    The upstream checkpoint is a legacy ``.pt`` container.  It may only be
    read with PyTorch's weights-only loader and must be bound to a manifest
    checksum.  A conversion to a different format is not inferred or accepted
    here because it would be a distinct artifact/provenance claim.
    """

    profile_id: str
    local_checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_revision: str
    source_checkout_path: str
    source_revision: str = OPEN_PI_ZERO_SOURCE_COMMIT
    autoeval_checkout_path: str = ""
    autoeval_revision: str = AUTOEVAL_POLICY_SOURCE_COMMIT
    eval_config_path: str = ""
    autoeval_statistics_path: str = ""
    paligemma_path: str = ""
    paligemma_model_id: str = OPENPI_PALIGEMMA_MODEL_ID
    paligemma_revision: str = ""
    paligemma_terms_evidence_uri: Optional[str] = None
    asset_manifest_path: str = ""
    asset_manifest_sha256: str = ""
    torch_version: str = OPENPI_TORCH_VERSION
    transformers_version: str = OPENPI_TRANSFORMERS_VERSION
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    action_normalization_type: str = "bounds"
    local_files_only: bool = True
    container_digest: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_checkpoint_path:
            return "local_checkpoint_path is required; remote checkpoint retrieval is disabled"
        if not _immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable checkpoint-host revision"
        if self.source_revision != OPEN_PI_ZERO_SOURCE_COMMIT:
            return "source_revision must equal the pinned allenzren/open-pi-zero commit"
        if self.autoeval_revision != AUTOEVAL_POLICY_SOURCE_COMMIT:
            return "autoeval_revision must equal the pinned AutoEval wrapper commit"
        if not _sha256_hex(self.checkpoint_sha256):
            return "checkpoint_sha256 must be a 64-character SHA-256 hex digest"
        if self.paligemma_model_id != OPENPI_PALIGEMMA_MODEL_ID:
            return "paligemma_model_id must be the official google/paligemma-3b-pt-224 path"
        if not _immutable_revision(self.paligemma_revision):
            return "paligemma_revision must be an immutable 40-character revision"
        if not self.paligemma_terms_evidence_uri:
            return "paligemma_terms_evidence_uri is required after accepting the official access terms"
        if not _sha256_hex(self.asset_manifest_sha256):
            return "asset_manifest_sha256 must be a 64-character SHA-256 hex digest"
        if self.torch_version != OPENPI_TORCH_VERSION:
            return "OpenPiZero requires source-pinned torch %s" % OPENPI_TORCH_VERSION
        if self.transformers_version != OPENPI_TRANSFORMERS_VERSION:
            return "OpenPiZero requires source-pinned transformers %s" % OPENPI_TRANSFORMERS_VERSION
        if self.torch_dtype not in ("bfloat16", "float32"):
            return "torch_dtype must be bfloat16 or float32"
        if self.action_normalization_type != "bounds":
            return "AutoEval OpenPiZero replication requires action_normalization_type='bounds'"
        if not self.local_files_only:
            return "OpenPiZero loader is local-only; network retrieval is not permitted"
        if not self.device:
            return "device is required"
        return None


@dataclass(frozen=True)
class OpenPiActionReport:
    """A source-native four-row proposal with no certified execution selection."""

    proposal: Tuple[Tuple[float, float, float, float, float, float, float], ...]
    source_image_timestamp: Optional[float]
    source_proprio: Tuple[float, ...]
    source_proprio_convention: str
    processed_proprio: Tuple[float, ...]
    backend_calls: int
    wall_seconds: Optional[float]
    native_proposal_horizon: int
    certified_execute_prefix: Optional[int]
    action_normalization_type: str
    source_provenance: Mapping[str, str]


@dataclass(frozen=True)
class _OpenPiRuntime:
    """Lazy source dependencies; factories exist only for offline behavior tests."""

    torch: Any
    numpy: Any
    cv2: Any
    omega_conf: Any
    model_cls: Any
    processor_cls: Any
    tokenizer_cls: Any
    quat2mat: Callable[[Any], Any]
    mat2euler: Callable[[Any], Any]
    torch_version: str
    transformers_version: str
    module_paths: Mapping[str, str]


OPENPI_CONTRACT = PolicyContract(
    name="OpenPiZero",
    required_observation_history=1,
    requires_proprio=True,
    native_proposal_horizon=OPENPI_ACTION_HORIZON,
    certified_execute_prefix=None,
    temporal_ensembling=True,
    preprocessing=(
        "Pinned AutoEval OpenPiZero wrapper: current 256x256 RGB image is Lanczos-resized to 224 and an explicitly "
        "tagged fresh Bridge EEF WXYZ quaternion pose is converted to normalized seven-value proprio."
    ),
    normalization=(
        "Pinned AutoEval policy.py bounds: q01/q99 denormalize action[:6]; gripper is binary "
        "(normalized > 0 becomes open=1, otherwise closed=0)."
    ),
    reset_rule="No observation/proprio cache in this adapter; each proposal receives current image and proprio.",
    rng_rule=(
        "PiZero flow inference samples action noise internally; source exposes no per-call generator argument. "
        "Record runtime RNG state/seed externally before qualification."
    ),
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation=(
        "The AutoEval source invokes an external TemporalEnsembleWrapper(env, 4); its executed prefix is not "
        "source-bound here. Native proposals are four actions, but no prefix is certified until an exact wrapper "
        "and fixture establish it."
    ),
)


class OpenPiZeroPolicyAdapter:
    """Fail-closed source-bound local OpenPiZero proposal loader."""

    contract = OPENPI_CONTRACT

    def __init__(
        self,
        profile: OpenPiZeroPolicyProfile,
        *,
        runtime_factory: Optional[Callable[[], _OpenPiRuntime]] = None,
        model_factory: Optional[Callable[[OpenPiZeroPolicyProfile, _OpenPiRuntime], Any]] = None,
        processor_factory: Optional[Callable[[OpenPiZeroPolicyProfile, _OpenPiRuntime], Any]] = None,
    ) -> None:
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._runtime: Optional[_OpenPiRuntime] = None
        self._model: Any = None
        self._processor: Any = None
        self._statistics: Optional[Mapping[str, Any]] = None
        self._backend_calls = 0
        self.last_report: Optional[OpenPiActionReport] = None

    def capability(self) -> CapabilityResult:
        error = self.profile.review_error()
        if error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="OpenPiZero loading is blocked: " + error + ".",
                source_verified=True,
                evidence_uris=self._sources(),
                details={"profile_id": self.profile.profile_id, "native_proposal_horizon": OPENPI_ACTION_HORIZON},
            )
        if self._model_factory is None and not Path(self.profile.local_checkpoint_path).is_file():
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="OpenPiZero checkpoint is absent locally; no download was attempted.",
                source_verified=True,
                evidence_uris=self._sources(),
                details={"local_checkpoint_path": self.profile.local_checkpoint_path, "local_files_only": True},
            )
        if self._model_factory is None:
            binding_error = self._real_binding_error()
            if binding_error is not None:
                return CapabilityResult(
                    status=CapabilityStatus.UNAVAILABLE,
                    reason="OpenPiZero local load is unavailable: " + binding_error + ".",
                    source_verified=True,
                    evidence_uris=self._sources(),
                    details={"local_files_only": True, "checkpoint_format": "torch weights_only state dict"},
                )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local OpenPiZero four-action proposals and fresh Bridge proprio are configured. "
                "The execution prefix, source-state conversion fixture, and Gate B remain unqualified."
            ),
            source_verified=True,
            evidence_uris=self._sources(),
            details={
                "checkpoint_revision": self.profile.checkpoint_revision,
                "source_revision": self.profile.source_revision,
                "autoeval_revision": self.profile.autoeval_revision,
                "paligemma_revision": self.profile.paligemma_revision,
                "native_proposal_horizon": OPENPI_ACTION_HORIZON,
                "certified_execute_prefix": None,
                "action_normalization_type": "bounds",
                "local_files_only": True,
                "unsafe_pickle_disabled": True,
            },
        )

    def _sources(self) -> Tuple[str, ...]:
        return (
            OPENPI_SOURCE,
            OPENPI_PI_ZERO_SOURCE,
            OPENPI_PROCESSING_SOURCE,
            OPENPI_AUTOEVAL_POLICY_SOURCE,
            OPENPI_AUTOEVAL_CONTROL_SOURCE,
            OPENPI_PALIGEMMA_SOURCE,
        )

    def _check_profile(self) -> None:
        error = self.profile.review_error()
        if error is not None:
            raise OpenPiUnavailableError("Refusing to load OpenPiZero: " + error + ".")

    def _real_binding_error(self) -> Optional[str]:
        required = {
            "source_checkout_path": self.profile.source_checkout_path,
            "autoeval_checkout_path": self.profile.autoeval_checkout_path,
            "eval_config_path": self.profile.eval_config_path,
            "autoeval_statistics_path": self.profile.autoeval_statistics_path,
            "paligemma_path": self.profile.paligemma_path,
            "asset_manifest_path": self.profile.asset_manifest_path,
        }
        absent = [name for name, value in required.items() if not value]
        if absent:
            return "missing required local binding(s): " + ", ".join(absent)
        if sys.version_info[:2] != OPENPI_PYTHON_MINOR:
            return "source runtime requires Python %d.%d" % OPENPI_PYTHON_MINOR
        return None

    @staticmethod
    def _verify_clean_checkout(path: Path, expected_commit: str, label: str) -> None:
        if not (path / ".git").exists():
            raise OpenPiUnavailableError("%s must be a git checkout" % label)
        try:
            head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise OpenPiUnavailableError("unable to inspect %s checkout" % label) from error
        if head != expected_commit:
            raise OpenPiUnavailableError("%s HEAD must equal pinned commit %s" % (label, expected_commit))
        if dirty:
            raise OpenPiUnavailableError("%s checkout must be clean before import" % label)

    def _verify_manifest(self) -> Mapping[str, Any]:
        manifest_path = Path(self.profile.asset_manifest_path)
        if not manifest_path.is_file():
            raise OpenPiUnavailableError("OpenPiZero asset manifest is absent")
        actual = _sha256_file(manifest_path)
        if actual != self.profile.asset_manifest_sha256:
            raise OpenPiUnavailableError("OpenPiZero asset manifest SHA-256 does not match the reviewed profile")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise OpenPiUnavailableError("OpenPiZero asset manifest is not valid JSON") from error
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "plumb-openpi-assets-v1":
            raise OpenPiUnavailableError("OpenPiZero asset manifest must use schema plumb-openpi-assets-v1")
        source = manifest.get("source")
        autoeval = manifest.get("autoeval")
        checkpoint = manifest.get("checkpoint")
        paligemma = manifest.get("paligemma")
        if (
            not isinstance(source, Mapping)
            or source.get("repository") != OPENPI_MODEL_ID
            or source.get("commit") != self.profile.source_revision
        ):
            raise OpenPiUnavailableError("asset manifest source commit does not bind the checked OpenPiZero code")
        if (
            not isinstance(autoeval, Mapping)
            or autoeval.get("repository") != "zhouzypaul/auto_eval"
            or autoeval.get("commit") != self.profile.autoeval_revision
        ):
            raise OpenPiUnavailableError("asset manifest AutoEval commit does not bind the normalization wrapper")
        if not isinstance(checkpoint, Mapping):
            raise OpenPiUnavailableError("asset manifest checkpoint record is required")
        if checkpoint.get("filename") != OPENPI_BRIDGE_BETA_CHECKPOINT:
            raise OpenPiUnavailableError("asset manifest must identify the reviewed Bridge beta checkpoint filename")
        if checkpoint.get("model_id") != OPENPI_MODEL_ID:
            raise OpenPiUnavailableError("asset manifest checkpoint model ID does not match OpenPiZero")
        if checkpoint.get("revision") != self.profile.checkpoint_revision:
            raise OpenPiUnavailableError("asset manifest checkpoint revision does not match profile")
        if checkpoint.get("sha256") != self.profile.checkpoint_sha256:
            raise OpenPiUnavailableError("asset manifest checkpoint SHA-256 does not match profile")
        if checkpoint.get("safe_loader") != "torch.load(weights_only=True)":
            raise OpenPiUnavailableError("asset manifest must require torch.load(weights_only=True)")
        if not isinstance(paligemma, Mapping):
            raise OpenPiUnavailableError("asset manifest PaliGemma record is required")
        if paligemma.get("model_id") != self.profile.paligemma_model_id or paligemma.get("revision") != self.profile.paligemma_revision:
            raise OpenPiUnavailableError("asset manifest PaliGemma identity does not match profile")
        if paligemma.get("terms_evidence_uri") != self.profile.paligemma_terms_evidence_uri:
            raise OpenPiUnavailableError("asset manifest PaliGemma terms evidence must match profile")
        return manifest

    @staticmethod
    def _require_manifest_file(
        root: Path, filename: str, record: Mapping[str, Any], label: str
    ) -> None:
        digest = record.get("sha256")
        if not _sha256_hex(digest):
            raise OpenPiUnavailableError("%s manifest record for %s lacks SHA-256" % (label, filename))
        candidate = root / filename
        if not candidate.is_file():
            raise OpenPiUnavailableError("required %s file is absent: %s" % (label, filename))
        if _sha256_file(candidate) != digest:
            raise OpenPiUnavailableError("%s SHA-256 mismatch: %s" % (label, filename))

    def _verify_real_bindings(self) -> Mapping[str, str]:
        binding_error = self._real_binding_error()
        if binding_error is not None:
            raise OpenPiUnavailableError(binding_error)
        source_root = Path(self.profile.source_checkout_path)
        autoeval_root = Path(self.profile.autoeval_checkout_path)
        self._verify_clean_checkout(source_root, self.profile.source_revision, "OpenPiZero source")
        self._verify_clean_checkout(autoeval_root, self.profile.autoeval_revision, "AutoEval source")
        manifest = self._verify_manifest()
        checkpoint = Path(self.profile.local_checkpoint_path)
        if not checkpoint.is_file() or checkpoint.name != OPENPI_BRIDGE_BETA_CHECKPOINT:
            raise OpenPiUnavailableError("reviewed Bridge beta checkpoint is absent or has the wrong filename")
        if _sha256_file(checkpoint) != self.profile.checkpoint_sha256:
            raise OpenPiUnavailableError("Bridge beta checkpoint SHA-256 mismatch")
        config_path = Path(self.profile.eval_config_path)
        stats_path = Path(self.profile.autoeval_statistics_path)
        source_config = source_root / OPENPI_EVAL_CONFIG
        if config_path.resolve() != source_config.resolve():
            raise OpenPiUnavailableError("profile must use the exact pinned source Bridge eval config")
        autoeval_policy = autoeval_root / "auto_eval" / "robot" / "policy.py"
        if not autoeval_policy.is_file():
            raise OpenPiUnavailableError("pinned AutoEval checkout lacks auto_eval/robot/policy.py")
        config_record = manifest.get("eval_config")
        stats_record = manifest.get("autoeval_statistics")
        if not isinstance(config_record, Mapping) or not isinstance(stats_record, Mapping):
            raise OpenPiUnavailableError("asset manifest must record the eval config and AutoEval bounds statistics")
        self._require_manifest_file(source_root, OPENPI_EVAL_CONFIG, config_record, "eval config")
        self._require_manifest_file(stats_path.parent, stats_path.name, stats_record, "AutoEval bounds statistics")
        paligemma = manifest.get("paligemma")
        assert isinstance(paligemma, Mapping)
        files = paligemma.get("files")
        if not isinstance(files, Mapping):
            raise OpenPiUnavailableError("asset manifest PaliGemma file records are required")
        pali_root = Path(self.profile.paligemma_path)
        for filename in OPENPI_REQUIRED_PALIGEMMA_FILES:
            record = files.get(filename)
            if not isinstance(record, Mapping):
                raise OpenPiUnavailableError("asset manifest lacks PaliGemma support file: " + filename)
            self._require_manifest_file(pali_root, filename, record, "PaliGemma")
        return {
            "source_revision": self.profile.source_revision,
            "autoeval_revision": self.profile.autoeval_revision,
            "checkpoint_revision": self.profile.checkpoint_revision,
            "checkpoint_sha256": "sha256:" + self.profile.checkpoint_sha256,
            "asset_manifest_sha256": "sha256:" + self.profile.asset_manifest_sha256,
            "paligemma_revision": self.profile.paligemma_revision,
            "source_eval_config": str(source_config),
            "autoeval_policy": str(autoeval_policy),
        }

    @contextlib.contextmanager
    def _source_import_path(self, source_root: Path) -> Iterator[None]:
        root = str(source_root.resolve())
        prior = list(sys.path)
        sys.path.insert(0, root)
        try:
            existing = sys.modules.get("src")
            if existing is not None:
                origin = getattr(existing, "__file__", None)
                if not origin or not _path_is_within(Path(str(origin)), source_root):
                    raise OpenPiUnavailableError("another checkout already owns the Python module name 'src'")
            yield
        finally:
            sys.path[:] = prior

    def _load_runtime(self) -> _OpenPiRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.torch_version != OPENPI_TORCH_VERSION or runtime.transformers_version != OPENPI_TRANSFORMERS_VERSION:
                raise OpenPiUnavailableError("injected OpenPiZero runtime does not match pinned Torch/Transformers versions")
            self._runtime = runtime
            return runtime
        provenance = self._verify_real_bindings()
        source_root = Path(self.profile.source_checkout_path)
        try:
            with self._source_import_path(source_root):
                import cv2  # type: ignore
                import numpy  # type: ignore
                import omegaconf  # type: ignore
                import torch  # type: ignore
                import transformers  # type: ignore
                from transformers import AutoTokenizer  # type: ignore

                model_module = importlib.import_module("src.model.vla.pizero")
                processing_module = importlib.import_module("src.model.vla.processing")
                geometry_module = importlib.import_module("src.utils.geometry")
        except ImportError as error:
            raise OpenPiUnavailableError(
                "OpenPiZero dependencies are unavailable. Install the pinned Python 3.10/Torch 2.5.0/Transformers "
                "4.47.1 policy environment; imports are lazy."
            ) from error
        if str(getattr(torch, "__version__", "")).split("+")[0] != OPENPI_TORCH_VERSION:
            raise OpenPiUnavailableError("installed Torch must be exactly %s" % OPENPI_TORCH_VERSION)
        if str(getattr(transformers, "__version__", "")) != OPENPI_TRANSFORMERS_VERSION:
            raise OpenPiUnavailableError("installed Transformers must be exactly %s" % OPENPI_TRANSFORMERS_VERSION)
        modules = {
            "pizero": str(getattr(model_module, "__file__", "")),
            "processing": str(getattr(processing_module, "__file__", "")),
            "geometry": str(getattr(geometry_module, "__file__", "")),
        }
        if any(not value or not _path_is_within(Path(value), source_root) for value in modules.values()):
            raise OpenPiUnavailableError("imported OpenPiZero module did not resolve inside the verified checkout")
        provenance.update({"import_" + key: value for key, value in modules.items()})
        runtime = _OpenPiRuntime(
            torch=torch,
            numpy=numpy,
            cv2=cv2,
            omega_conf=omegaconf.OmegaConf,
            model_cls=getattr(model_module, "PiZeroInference"),
            processor_cls=getattr(processing_module, "VLAProcessor"),
            tokenizer_cls=AutoTokenizer,
            quat2mat=getattr(geometry_module, "quat2mat"),
            mat2euler=getattr(geometry_module, "mat2euler"),
            torch_version=OPENPI_TORCH_VERSION,
            transformers_version=OPENPI_TRANSFORMERS_VERSION,
            module_paths=provenance,
        )
        self._runtime = runtime
        return runtime

    @staticmethod
    def _safe_state_dict(torch: Any, checkpoint_path: Path) -> Mapping[str, Any]:
        """Load the legacy checkpoint with PyTorch's restricted weights-only path."""

        try:
            payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except TypeError as error:
            raise OpenPiUnavailableError("installed Torch lacks required weights_only checkpoint loading") from error
        except Exception as error:
            raise OpenPiUnavailableError("weights-only OpenPiZero checkpoint loading failed") from error
        if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
            raise OpenPiUnavailableError("weights-only checkpoint must contain a mapping under 'model'")
        state: Dict[str, Any] = {}
        is_tensor = getattr(torch, "is_tensor", None)
        if not callable(is_tensor):
            raise OpenPiUnavailableError("Torch runtime must expose is_tensor for safe state-dict validation")
        for key, value in payload["model"].items():
            if not isinstance(key, str):
                raise OpenPiUnavailableError("weights-only state dict keys must be strings")
            if not is_tensor(value):
                raise OpenPiUnavailableError("weights-only state dict must contain tensors only")
            canonical = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
            if canonical in state:
                raise OpenPiUnavailableError("compiled checkpoint key normalization produced a collision")
            state[canonical] = value
        if not state:
            raise OpenPiUnavailableError("weights-only checkpoint state dict is empty")
        return state

    def _load_model(self, runtime: _OpenPiRuntime) -> Any:
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            model = self._model_factory(self.profile, runtime)
        else:
            self._verify_real_bindings()
            config = runtime.omega_conf.load(self.profile.eval_config_path)
            if int(config.horizon_steps) != OPENPI_ACTION_HORIZON or int(config.action_dim) != OPENPI_ACTION_DIM:
                raise OpenPiUnavailableError("pinned Bridge eval config must declare a 4x7 action proposal")
            if int(config.proprio_dim) != OPENPI_MODEL_PROPRIO_DIM:
                raise OpenPiUnavailableError("pinned Bridge eval config must declare seven model proprio values")
            if int(config.cond_steps) != 1 or str(config.flow_sampling) != "beta":
                raise OpenPiUnavailableError(
                    "Bridge beta checkpoint requires source eval config cond_steps=1 and flow_sampling='beta'"
                )
            model = runtime.model_cls(config, use_ddp=False)
            state = self._safe_state_dict(runtime.torch, Path(self.profile.local_checkpoint_path))
            loaded = model.load_state_dict(state, strict=True)
            missing = getattr(loaded, "missing_keys", ())
            unexpected = getattr(loaded, "unexpected_keys", ())
            if missing or unexpected:
                raise OpenPiUnavailableError("strict checkpoint load reported missing or unexpected state keys")
            freeze = getattr(model, "freeze_all_weights", None)
            if not callable(freeze):
                raise OpenPiUnavailableError("pinned PiZero model lacks freeze_all_weights")
            freeze()
            dtype = getattr(runtime.torch, self.profile.torch_dtype)
            model.to(dtype)
            model.to(self.profile.device)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        self._model = model
        return model

    def _load_processor(self, runtime: _OpenPiRuntime) -> Any:
        if self._processor is not None:
            return self._processor
        if self._processor_factory is not None:
            processor = self._processor_factory(self.profile, runtime)
        else:
            self._verify_real_bindings()
            try:
                tokenizer = runtime.tokenizer_cls.from_pretrained(
                    self.profile.paligemma_path,
                    padding_side="right",
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as error:
                raise OpenPiUnavailableError("local PaliGemma tokenizer support files could not be loaded") from error
            if getattr(tokenizer, "padding_side", None) != "right":
                raise OpenPiUnavailableError("PaliGemma tokenizer must preserve source padding_side='right'")
            processor = runtime.processor_cls(tokenizer, num_image_tokens=256, max_seq_len=276, tokenizer_padding="max_length")
        self._processor = processor
        return processor

    def _load_statistics(self, runtime: _OpenPiRuntime) -> Mapping[str, Any]:
        if self._statistics is not None:
            return self._statistics
        if self._model_factory is not None:
            # Fixture factories attach source-shaped statistics directly to the
            # profile metadata-free test path through this controlled default.
            raise OpenPiUnavailableError("fixture runtime must supply statistics through set_fixture_statistics")
        self._verify_real_bindings()
        try:
            parsed = json.loads(Path(self.profile.autoeval_statistics_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise OpenPiUnavailableError("Bridge action/proprio statistics are unreadable") from error
        if not isinstance(parsed, Mapping):
            raise OpenPiUnavailableError("Bridge statistics must be a JSON object")
        bridge_orig = parsed.get("bridge_orig") if isinstance(parsed, Mapping) else None
        self._statistics = self._coerce_statistics(bridge_orig if isinstance(bridge_orig, Mapping) else parsed, runtime.numpy)
        return self._statistics

    def set_fixture_statistics(self, statistics: Mapping[str, Any]) -> None:
        """Inject source-shaped statistics only for no-framework fixture tests."""

        if self._model_factory is None:
            raise OpenPiUnavailableError("fixture statistics cannot replace a real manifest-bound statistics file")
        runtime = self._load_runtime()
        self._statistics = self._coerce_statistics(statistics, runtime.numpy)

    @staticmethod
    def _coerce_statistics(statistics: Mapping[str, Any], numpy_module: Any) -> Mapping[str, Any]:
        action = statistics.get("action")
        proprio = statistics.get("proprio")
        if not isinstance(action, Mapping) or not isinstance(proprio, Mapping):
            raise OpenPiUnavailableError("Bridge statistics require action and proprio mappings")
        required = ("q01", "q99")
        if any(key not in action or key not in proprio for key in required):
            raise OpenPiUnavailableError("AutoEval bounds normalization requires q01 and q99 for actions and proprio")
        result = {
            "action": {key: numpy_module.asarray(value, dtype=float) for key, value in action.items()},
            "proprio": {key: numpy_module.asarray(value, dtype=float) for key, value in proprio.items()},
        }
        for name, record, expected in (("action", result["action"], OPENPI_ACTION_DIM), ("proprio", result["proprio"], OPENPI_MODEL_PROPRIO_DIM)):
            for key in required:
                value = record[key]
                if tuple(getattr(value, "shape", ())) != (expected,):
                    raise OpenPiUnavailableError("Bridge %s.%s must be a %d-value vector" % (name, key, expected))
                if not bool(numpy_module.isfinite(value).all()):
                    raise OpenPiUnavailableError("Bridge %s.%s must be finite" % (name, key))
            if bool((record["q99"] <= record["q01"]).any()):
                raise OpenPiUnavailableError("Bridge %s bounds must have q99 > q01" % name)
        return result

    @staticmethod
    def _finite_vector(values: Optional[Sequence[float]], expected: int, label: str) -> Tuple[float, ...]:
        if values is None or len(values) != expected:
            raise PolicyContractError("OpenPiZero %s must contain exactly %d values" % (label, expected))
        converted = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in converted):
            raise PolicyContractError("OpenPiZero %s values must be finite" % label)
        return converted

    def _prepare_proprio(self, observation: PolicyObservation, runtime: _OpenPiRuntime, statistics: Mapping[str, Any]) -> Any:
        if getattr(observation, "proprio_convention", None) != OPENPI_BRIDGE_EEF_QUATERNION_WXYZ:
            raise PolicyContractError(
                "OpenPiZero requires proprio_convention=%r; canonical Bridge Euler state must not be misread as a quaternion"
                % OPENPI_BRIDGE_EEF_QUATERNION_WXYZ
            )
        raw = self._finite_vector(observation.proprio, OPENPI_SOURCE_PROPRIO_DIM, "Bridge EEF quaternion proprio")
        np = runtime.numpy
        # This follows the AutoEval OpenPiZero wrapper exactly. It consumes a
        # WXYZ quaternion at positions 3..6, not PLUMB's separate Euler-state
        # convention; callers must supply a provenance-bound conversion.
        source = np.asarray(raw, dtype=float)
        quaternion_norm = float(np.linalg.norm(source[3:7]))
        if not math.isfinite(quaternion_norm) or abs(quaternion_norm - 1.0) > OPENPI_QUATERNION_NORM_TOLERANCE:
            raise PolicyContractError(
                "OpenPiZero Bridge EEF quaternion must have unit norm within %.0e" % OPENPI_QUATERNION_NORM_TOLERANCE
            )
        rotation = runtime.quat2mat(source[3:7])
        default_rotation = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
        rpy = runtime.mat2euler(rotation @ default_rotation.T)
        raw_model = np.concatenate([source[:3], np.asarray(rpy, dtype=float), [source[7]]])
        if tuple(getattr(raw_model, "shape", ())) != (OPENPI_MODEL_PROPRIO_DIM,) or not bool(np.isfinite(raw_model).all()):
            raise PolicyContractError("OpenPiZero converted Bridge proprio must be finite 7-D")
        lower = statistics["proprio"]["q01"]
        upper = statistics["proprio"]["q99"]
        normalized = 2.0 * (raw_model - lower) / (upper - lower + 1e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        if not bool(np.isfinite(normalized).all()):
            raise PolicyContractError("OpenPiZero normalized proprio must be finite")
        return raw, tuple(float(value) for value in normalized), normalized

    @staticmethod
    def _prepare_image(image: Any, runtime: _OpenPiRuntime) -> Any:
        np = runtime.numpy
        source = np.asarray(image)
        if tuple(getattr(source, "shape", ())) != (256, 256, 3) or str(getattr(source, "dtype", "")) != "uint8":
            raise PolicyContractError("OpenPiZero requires one current 256x256 RGB uint8 image")
        resized = runtime.cv2.resize(source, (224, 224), interpolation=runtime.cv2.INTER_LANCZOS4)
        if tuple(getattr(resized, "shape", ())) != (224, 224, 3):
            raise OpenPiUnavailableError("source image resize did not produce 224x224 RGB")
        return runtime.torch.as_tensor(resized, dtype=runtime.torch.uint8).permute(2, 0, 1)[None]

    @staticmethod
    def _to_device(value: Any, device: str, dtype: Optional[Any] = None) -> Any:
        method = getattr(value, "to", None)
        if not callable(method):
            return value
        if dtype is None:
            return method(device)
        return method(device, dtype=dtype)

    def _model_proposal(self, observation: PolicyObservation, runtime: _OpenPiRuntime) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        if len(observation.image_history) != 1:
            raise PolicyContractError("OpenPiZero requires exactly one fresh current image; it owns no image history")
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("OpenPiZero requires a non-empty task instruction")
        statistics = self._load_statistics(runtime)
        raw_proprio, processed_proprio, normalized_proprio = self._prepare_proprio(observation, runtime, statistics)
        image = self._prepare_image(observation.image_history[0], runtime)
        model = self._load_model(runtime)
        processor = self._load_processor(runtime)
        tokenized = processor(text=[observation.prompt], images=image)
        required_tokens = ("input_ids", "pixel_values", "attention_mask")
        if not isinstance(tokenized, Mapping) or any(key not in tokenized for key in required_tokens):
            raise OpenPiUnavailableError("source PaliGemma processor did not produce required model tokens")
        dtype = getattr(runtime.torch, self.profile.torch_dtype)
        causal_mask, vlm_positions, proprio_positions, action_positions = model.build_causal_mask_and_position_ids(
            tokenized["attention_mask"], dtype=dtype
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
        propri_tensor = runtime.torch.as_tensor(normalized_proprio, dtype=dtype)[None, None]
        model_inputs = {
            "input_ids": self._to_device(tokenized["input_ids"], self.profile.device),
            "pixel_values": self._to_device(tokenized["pixel_values"], self.profile.device, dtype),
            "image_text_proprio_mask": self._to_device(image_text_proprio_mask, self.profile.device),
            "action_mask": self._to_device(action_mask, self.profile.device),
            "vlm_position_ids": self._to_device(vlm_positions, self.profile.device),
            "proprio_position_ids": self._to_device(proprio_positions, self.profile.device),
            "action_position_ids": self._to_device(action_positions, self.profile.device),
            "proprios": self._to_device(propri_tensor, self.profile.device, dtype),
        }
        context = runtime.torch.inference_mode()
        with context:
            proposal = model(**model_inputs)
        row = proposal[0]
        float_value = getattr(row, "float", None)
        if callable(float_value):
            row = float_value()
        cpu = getattr(row, "cpu", None)
        if callable(cpu):
            row = cpu()
        numpy_value = getattr(row, "numpy", None)
        if callable(numpy_value):
            row = numpy_value()
        raw_actions = runtime.numpy.asarray(row, dtype=float)
        if tuple(getattr(raw_actions, "shape", ())) != (OPENPI_ACTION_HORIZON, OPENPI_ACTION_DIM):
            raise OpenPiUnavailableError("PiZero native model must return a four-by-seven action proposal")
        if not bool(runtime.numpy.isfinite(raw_actions).all()):
            raise OpenPiUnavailableError("PiZero native model returned non-finite action values")
        lower = statistics["action"]["q01"]
        upper = statistics["action"]["q99"]
        proposal_rows = []
        for action in raw_actions:
            first_six = (action[:6] + 1.0) / 2.0 * (upper[:6] - lower[:6]) + lower[:6]
            gripper = 1.0 if float(action[6]) > 0.0 else 0.0
            converted = tuple(float(value) for value in runtime.numpy.concatenate([first_six, [gripper]]))
            if not all(math.isfinite(value) for value in converted):
                raise OpenPiUnavailableError("AutoEval bounds denormalization returned non-finite action")
            proposal_rows.append(converted)
        self._backend_calls += 1
        return tuple(proposal_rows), raw_proprio, processed_proprio

    def predict_proposal_with_report(self, observation: PolicyObservation) -> OpenPiActionReport:
        self._check_profile()
        runtime = self._load_runtime()
        started = time.monotonic()
        proposal, raw_proprio, processed_proprio = self._model_proposal(observation, runtime)
        report = OpenPiActionReport(
            proposal=proposal,
            source_image_timestamp=observation.timestamp,
            source_proprio=raw_proprio,
            source_proprio_convention=OPENPI_BRIDGE_EEF_QUATERNION_WXYZ,
            processed_proprio=processed_proprio,
            backend_calls=self._backend_calls,
            wall_seconds=time.monotonic() - started,
            native_proposal_horizon=OPENPI_ACTION_HORIZON,
            certified_execute_prefix=None,
            action_normalization_type="bounds",
            source_provenance=dict(runtime.module_paths),
        )
        self.last_report = report
        return report

    def predict_proposal(self, observation: PolicyObservation) -> Tuple[Tuple[float, float, float, float, float, float, float], ...]:
        """Return the source-native four-row proposal without selecting a prefix."""

        return self.predict_proposal_with_report(observation).proposal

    def predict_action(self, observation: PolicyObservation) -> Tuple[float, float, float, float, float, float, float]:
        """Refuse a one-action facade until the external execution wrapper is pinned."""

        del observation
        raise OpenPiUnavailableError(
            "OpenPiZero exposes a four-action native proposal only; its AutoEval execution prefix is not certified"
        )

    def reset(self) -> None:
        """Clear only reporting counters; no stale image/proprio/action state is retained."""

        self._backend_calls = 0
        self.last_report = None


__all__ = [
    "OPENPI_ACTION_DIM",
    "OPENPI_ACTION_HORIZON",
    "OPENPI_AUTOEVAL_CONTROL_SOURCE",
    "OPENPI_AUTOEVAL_POLICY_SOURCE",
    "OPENPI_BRIDGE_EEF_QUATERNION_WXYZ",
    "OPENPI_QUATERNION_NORM_TOLERANCE",
    "OPENPI_BRIDGE_BETA_CHECKPOINT",
    "OPENPI_MODEL_ID",
    "OPENPI_PALIGEMMA_MODEL_ID",
    "OPENPI_REQUIRED_PALIGEMMA_FILES",
    "OPENPI_SOURCE_COMMIT",
    "OPENPI_TORCH_VERSION",
    "OPENPI_TRANSFORMERS_VERSION",
    "OPENPI_CONTRACT",
    "OpenPiActionReport",
    "OpenPiUnavailableError",
    "OpenPiZeroPolicyAdapter",
    "OpenPiZeroPolicyProfile",
]
