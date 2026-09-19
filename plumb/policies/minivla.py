"""Pinned, local-only MiniVLA Bridge inference with a seven-action chunk.

MiniVLA's supplied configuration proposes seven actions per fresh image.  Two
properties make this adapter unusually strict:

* both of its released artifacts ship as legacy ``.pt`` pickles, so inference
  consumes only reviewed *converted* artifacts produced by an isolated,
  credential-free converter (see ``cluster/convert_irasim_checkpoint.py`` for
  the pattern).  A pickle scan or ``strict=True`` alone does not establish
  safety, so the converted artifact's hash and its conversion report's hash are
  both required here;
* ``Stanford-ILIAD/pretrain_vq`` declares no license at all (``cardData:
  null``).  Loading therefore needs an explicit operator acknowledgement, and
  the capability record always carries ``license: null`` with redistribution
  marked prohibited pending resolution.  PLUMB never mirrors it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from plumb.adapters.contracts import PolicyContract, PolicyObservation

from .contracts import (
    ActionNormalizationType,
    FrameIdentity,
    GripperPolarityConvention,
    NativePolicyAdapter,
    NativeProposal,
    PolicyActionNormalizer,
    PolicyCertification,
    PolicyContractError,
    PolicyExecutionMode,
    PolicyLoadError,
    UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    immutable_revision,
    inference_context,
    peak_memory_bytes,
    rows_from_native,
)
from .native import AUTOEVAL_POLICY_SOURCE, MINIVLA_CONTRACT, MINIVLA_SOURCE


MINIVLA_MODEL_ID = "Stanford-ILIAD/minivla-vq-bridge-prismatic"
MINIVLA_VQ_MODEL_ID = "Stanford-ILIAD/pretrain_vq"
MINIVLA_CHECKPOINT_FILE = "checkpoints/step-362500-epoch-21-loss=0.2259.pt"
MINIVLA_VQ_CHECKPOINT_FILE = (
    "pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt"
)
MINIVLA_TRANSFORMERS_VERSION = "4.40.1"
MINIVLA_TOKENIZERS_VERSION = "0.19.1"
MINIVLA_ACTION_CHUNK = 7
MINIVLA_REFUSED_WEIGHT_SUFFIXES = (".pt", ".pth", ".pkl", ".bin", ".ckpt")
MINIVLA_VQ_LICENSE_STATUS = "absent_cardData_null"
MINIVLA_VQ_REDISTRIBUTION = "prohibited_pending_resolution"
KNOWN_LICENSE_STATUSES = ("declared", "advertised_unverified", "absent_cardData_null", "unresolved")


class MiniVLAUnavailableError(PolicyLoadError):
    """The approved local MiniVLA environment or converted artifact is absent."""


@dataclass(frozen=True)
class ConvertedWeightArtifact:
    """A reviewed, converted weights artifact produced outside the inference path.

    ``license`` is ``None`` when upstream declares none.  ``None`` means unknown
    or absent; it is never rewritten as a permissive default.
    """

    label: str
    path: str
    sha256: Optional[str] = None
    source_pickle_sha256: Optional[str] = None
    conversion_report_path: Optional[str] = None
    conversion_report_sha256: Optional[str] = None
    license: Optional[str] = None
    license_status: str = "unresolved"
    redistribution: str = "unresolved"

    def review_error(self) -> Optional[str]:
        if not self.path:
            return "%s converted artifact path is required" % self.label
        suffix = Path(self.path).suffix.lower()
        if suffix in MINIVLA_REFUSED_WEIGHT_SUFFIXES:
            return (
                "%s must be a reviewed converted artifact, not a legacy %s pickle; convert it in an isolated, "
                "credential-free, network-free environment first" % (self.label, suffix)
            )
        if suffix != ".safetensors":
            return "%s converted artifact must be a .safetensors file, got %r" % (self.label, suffix or "<none>")
        if not _is_sha256_hex(self.sha256):
            return "%s converted artifact needs its recorded 64-character SHA-256" % self.label
        if not _is_sha256_hex(self.source_pickle_sha256):
            return "%s needs the source pickle's SHA-256 so the conversion is traceable" % self.label
        if not _is_sha256_hex(self.conversion_report_sha256):
            return (
                "%s needs its isolated conversion report hash; a pickle scan or strict=True alone does not "
                "establish safety" % self.label
            )
        if self.license_status not in KNOWN_LICENSE_STATUSES:
            return "%s license_status %r is not one of %s" % (
                self.label,
                self.license_status,
                ", ".join(KNOWN_LICENSE_STATUSES),
            )
        if self.license is None and self.redistribution != MINIVLA_VQ_REDISTRIBUTION:
            return (
                "%s declares no license, so redistribution must be marked %r rather than left %r"
                % (self.label, MINIVLA_VQ_REDISTRIBUTION, self.redistribution)
            )
        if self.license_status == "unresolved" and self.redistribution != MINIVLA_VQ_REDISTRIBUTION:
            return (
                "%s has an unresolved license, so redistribution must be marked %r; local use may proceed but "
                "mirroring may not" % (self.label, MINIVLA_VQ_REDISTRIBUTION)
            )
        return None

    def payload(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "sha256": self.sha256,
            "source_pickle_sha256": self.source_pickle_sha256,
            "conversion_report_path": self.conversion_report_path,
            "conversion_report_sha256": self.conversion_report_sha256,
            "license": self.license,
            "license_status": self.license_status,
            "redistribution": self.redistribution,
        }


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value[len("sha256:"):] if value.startswith("sha256:") else value
    return len(candidate) == 64 and all(character in "0123456789abcdefABCDEF" for character in candidate)


@dataclass(frozen=True)
class MiniVLAPolicyProfile:
    """The approval boundary for one local MiniVLA snapshot."""

    profile_id: str
    converted_checkpoint: ConvertedWeightArtifact
    converted_vq: ConvertedWeightArtifact
    checkpoint_revision: str
    vq_revision: str
    loader_revision: Optional[str] = None
    model_id: str = MINIVLA_MODEL_ID
    vq_model_id: str = MINIVLA_VQ_MODEL_ID
    vq_license_acknowledged: bool = False
    use_extra: bool = True
    action_chunk: int = MINIVLA_ACTION_CHUNK
    transformers_version: str = MINIVLA_TRANSFORMERS_VERSION
    tokenizers_version: str = MINIVLA_TOKENIZERS_VERSION
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    local_files_only: bool = True
    native_unnormalizes: bool = False
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if self.model_id != MINIVLA_MODEL_ID:
            return "model_id must be %r" % MINIVLA_MODEL_ID
        if self.vq_model_id != MINIVLA_VQ_MODEL_ID:
            return "vq_model_id must be %r" % MINIVLA_VQ_MODEL_ID
        if not immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable 40-character hexadecimal revision"
        if not immutable_revision(self.vq_revision):
            return "vq_revision must be an immutable 40-character hexadecimal revision"
        if not immutable_revision(self.loader_revision):
            return (
                "loader_revision must pin the reviewed Stanford-ILIAD/openvla-mini commit as a 40-character "
                "hexadecimal revision"
            )
        checkpoint_error = self.converted_checkpoint.review_error()
        if checkpoint_error is not None:
            return checkpoint_error
        vq_error = self.converted_vq.review_error()
        if vq_error is not None:
            return vq_error
        if self.converted_vq.license is not None:
            return (
                "%s declares no license upstream (cardData: null); recording a license for it here would be a "
                "fabrication" % MINIVLA_VQ_MODEL_ID
            )
        if self.converted_vq.license_status != MINIVLA_VQ_LICENSE_STATUS:
            return "%s license status must be recorded as %r" % (MINIVLA_VQ_MODEL_ID, MINIVLA_VQ_LICENSE_STATUS)
        if not self.vq_license_acknowledged:
            return (
                "%s declares no license. Loading requires an explicit operator acknowledgement; PLUMB does not "
                "mirror or redistribute it, and private hosting is not a substitute for redistribution rights"
                % MINIVLA_VQ_MODEL_ID
            )
        if not self.use_extra:
            return "MiniVLA's supplied configuration is use_extra=True"
        if int(self.action_chunk) != MINIVLA_ACTION_CHUNK:
            return "MiniVLA's supplied configuration is a %d-action chunk, not %r" % (
                MINIVLA_ACTION_CHUNK,
                self.action_chunk,
            )
        if self.transformers_version != MINIVLA_TRANSFORMERS_VERSION:
            return "MiniVLA requires its pinned Transformers %s runtime, not %r" % (
                MINIVLA_TRANSFORMERS_VERSION,
                self.transformers_version,
            )
        if self.tokenizers_version != MINIVLA_TOKENIZERS_VERSION:
            return "MiniVLA requires its pinned tokenizers %s, not %r" % (
                MINIVLA_TOKENIZERS_VERSION,
                self.tokenizers_version,
            )
        if not self.local_files_only:
            return "MiniVLA loader is local-only; network retrieval is not permitted"
        if self.native_unnormalizes:
            return (
                "PLUMB applies its own revisioned action normalizer at exactly one documented boundary; letting "
                "the MiniVLA wrapper also unnormalize would apply the transform twice"
            )
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            return "torch_dtype must be bfloat16, float16, or float32"
        return None


@dataclass(frozen=True)
class _MiniVLARuntime:
    """Injection seam for the lazily imported Torch/prismatic stack."""

    torch: Any
    transformers_version: str
    image_fromarray: Callable[[Any], Any]
    model_loader: Callable[[MiniVLAPolicyProfile], Any]


class MiniVLAPolicyAdapter(NativePolicyAdapter):
    """Local-only, lazily imported MiniVLA seven-action chunk predictor."""

    base_contract: PolicyContract = MINIVLA_CONTRACT
    source_urls = (MINIVLA_SOURCE, AUTOEVAL_POLICY_SOURCE)

    def __init__(
        self,
        profile: MiniVLAPolicyProfile,
        *,
        normalizer: PolicyActionNormalizer,
        certification: Optional[PolicyCertification] = None,
        execution_mode: PolicyExecutionMode = PolicyExecutionMode.CERTIFIED,
        gripper_convention: GripperPolarityConvention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
        runtime_factory: Optional[Callable[[], _MiniVLARuntime]] = None,
        model_factory: Optional[Callable[[MiniVLAPolicyProfile, _MiniVLARuntime], Any]] = None,
    ) -> None:
        super().__init__(
            normalizer=normalizer,
            certification=certification,
            execution_mode=execution_mode,
            gripper_convention=gripper_convention,
        )
        if normalizer.statistics.normalization_type is not ActionNormalizationType.NORMAL:
            raise PolicyContractError(
                "MiniVLA's checkpoint statistics are the mean/standard-deviation family; the 'bounds' setting is "
                "OpenPiZero's, not a universal native-policy API."
            )
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._runtime: Optional[_MiniVLARuntime] = None
        self._model: Any = None
        self.last_gpu_peak_memory_bytes: Optional[int] = None

    # -- capability -------------------------------------------------------------

    def _profile_errors(self) -> Tuple[str, ...]:
        error = self.profile.review_error()
        return () if error is None else (error,)

    def _availability_error(self) -> Optional[str]:
        if self._model_factory is not None:
            return None
        missing = [
            artifact.path
            for artifact in (self.profile.converted_checkpoint, self.profile.converted_vq)
            if not Path(artifact.path).is_file()
        ]
        if missing:
            return "reviewed converted artifacts are absent: %s" % ", ".join(sorted(missing))
        return None

    def _capability_details(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "model_id": self.profile.model_id,
            "vq_model_id": self.profile.vq_model_id,
            "checkpoint_revision": self.profile.checkpoint_revision,
            "vq_revision": self.profile.vq_revision,
            "loader_revision": self.profile.loader_revision,
            "upstream_checkpoint_file": MINIVLA_CHECKPOINT_FILE,
            "upstream_vq_checkpoint_file": MINIVLA_VQ_CHECKPOINT_FILE,
            "converted_checkpoint": self.profile.converted_checkpoint.payload(),
            "converted_vq": self.profile.converted_vq.payload(),
            "vq_license": None,
            "vq_license_status": MINIVLA_VQ_LICENSE_STATUS,
            "vq_redistribution": MINIVLA_VQ_REDISTRIBUTION,
            "vq_license_acknowledged": bool(self.profile.vq_license_acknowledged),
            "use_extra": bool(self.profile.use_extra),
            "action_chunk": int(self.profile.action_chunk),
            "transformers_version": self.profile.transformers_version,
            "tokenizers_version": self.profile.tokenizers_version,
            "trust_remote_code": False,
            "local_files_only": True,
            "asset_manifest_id": self.profile.asset_manifest_id,
            "asset_manifest_sha256": self.profile.asset_manifest_sha256,
            "runtime_lock_id": self.profile.runtime_lock_id,
            "runtime_lock_sha256": self.profile.runtime_lock_sha256,
        }

    # -- durable state ----------------------------------------------------------

    def _snapshot_extra(self) -> Dict[str, Any]:
        return {
            "sampling": {"do_sample": False, "action_chunk": int(self.profile.action_chunk)},
            "use_extra": bool(self.profile.use_extra),
        }

    def _restore_extra(self, payload: Mapping[str, Any]) -> None:
        sampling = payload.get("sampling")
        if not isinstance(sampling, Mapping):
            raise PolicyContractError("MiniVLA snapshot must carry its sampling configuration.")
        if bool(sampling.get("do_sample", True)):
            raise PolicyContractError("MiniVLA snapshot records sampling; the certified path is deterministic.")
        if int(sampling.get("action_chunk", 0)) != int(self.profile.action_chunk):
            raise PolicyContractError("MiniVLA snapshot action chunk does not match the live profile.")
        if bool(payload.get("use_extra")) != bool(self.profile.use_extra):
            raise PolicyContractError("MiniVLA snapshot use_extra does not match the live profile.")

    # -- runtime ----------------------------------------------------------------

    def _check_profile(self) -> None:
        error = self.profile.review_error()
        if error is not None:
            raise MiniVLAUnavailableError("Refusing to load MiniVLA: " + error + ".")

    def _load_runtime(self) -> _MiniVLARuntime:
        if self._runtime is not None:
            return self._runtime
        self._check_profile()
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.transformers_version != self.profile.transformers_version:
                raise MiniVLAUnavailableError(
                    "Injected MiniVLA runtime is Transformers %r; exactly %s is required."
                    % (runtime.transformers_version, self.profile.transformers_version)
                )
            self._runtime = runtime
            return runtime
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from PIL import Image  # type: ignore
            from prismatic import load_vla  # type: ignore
        except ImportError as error:
            raise MiniVLAUnavailableError(
                "MiniVLA dependencies are unavailable. Install the pinned python 3.10 / Torch 2.2.0 / "
                "Transformers %s / timm 0.9.10 / flash-attn 2.5.5 image in an isolated policy container; imports "
                "here are lazy." % MINIVLA_TRANSFORMERS_VERSION
            ) from error
        installed = str(getattr(transformers, "__version__", ""))
        if installed != self.profile.transformers_version:
            raise MiniVLAUnavailableError(
                "Installed Transformers %r is incompatible; MiniVLA requires exactly %s."
                % (installed, self.profile.transformers_version)
            )

        def _load(profile: MiniVLAPolicyProfile) -> Any:
            return load_vla(
                profile.converted_checkpoint.path,
                load_for_training=False,
                use_extra=profile.use_extra,
                vq_checkpoint=profile.converted_vq.path,
            )

        self._runtime = _MiniVLARuntime(
            torch=torch,
            transformers_version=installed,
            image_fromarray=Image.fromarray,
            model_loader=_load,
        )
        return self._runtime

    def _ensure_model(self, runtime: _MiniVLARuntime) -> Any:
        self._check_profile()
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory(self.profile, runtime)
        else:
            for artifact in (self.profile.converted_checkpoint, self.profile.converted_vq):
                if not Path(artifact.path).is_file():
                    raise MiniVLAUnavailableError(
                        "Expected the reviewed converted %s at %s; refusing network retrieval and refusing to "
                        "load the upstream legacy pickle." % (artifact.label, artifact.path)
                    )
            self._model = runtime.model_loader(self.profile)
        if not callable(getattr(self._model, "predict_action", None)):
            raise MiniVLAUnavailableError("Loaded MiniVLA model does not expose native predict_action().")
        eval_method = getattr(self._model, "eval", None)
        if callable(eval_method):
            maybe_model = eval_method()
            if maybe_model is not None:
                self._model = maybe_model
        return self._model

    def _prepare_image(self, runtime: _MiniVLARuntime, image: Any) -> Any:
        convert = getattr(image, "convert", None)
        if callable(convert):
            return convert("RGB")
        try:
            converted = runtime.image_fromarray(image)
        except Exception as error:
            raise PolicyContractError(
                "MiniVLA requires a current RGB uint8 image or PIL-compatible array for native preprocessing."
            ) from error
        convert = getattr(converted, "convert", None)
        if not callable(convert):
            raise PolicyContractError("MiniVLA image factory did not return a PIL-compatible image.")
        return convert("RGB")

    # -- native call ------------------------------------------------------------

    def _native_proposal(self, observation: PolicyObservation, identity: FrameIdentity) -> NativeProposal:
        runtime = self._load_runtime()
        model = self._ensure_model(runtime)
        image = self._prepare_image(runtime, observation.image_history[0])
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass
        started = time.perf_counter()
        # ``unnorm_key=None`` asks the released wrapper for model-space actions so
        # PLUMB's revisioned normalizer is the single denormalization boundary.
        # Whether the pinned openvla-mini commit honours that argument is a
        # wrapper detail the golden action fixture must confirm.
        with inference_context(runtime.torch):
            raw = model.predict_action(
                image=image,
                instruction=observation.prompt,
                unnorm_key=None,
                do_sample=False,
            )
        wall_seconds = time.perf_counter() - started
        self.last_gpu_peak_memory_bytes = peak_memory_bytes(runtime.torch)
        normalized = rows_from_native(
            raw,
            label="MiniVLA predict_action",
            expected_rows=self.base_contract.native_proposal_horizon,
        )
        actions = tuple(self.normalizer.denormalize(row) for row in normalized)
        return NativeProposal(
            policy_name=self.base_contract.name,
            actions=actions,
            executed_actions=actions,
            history_length=self.base_contract.required_observation_history,
            normalizer_revision=self.normalizer.revision,
            source_revision=self.profile.checkpoint_revision,
            observation_sha256=identity.key,
            observation_identity_stable=identity.stable,
            backend_calls=1,
            wall_seconds=wall_seconds,
            normalizer_counters=self.normalizer.counters(),
            metadata={
                "model_id": self.profile.model_id,
                "vq_model_id": self.profile.vq_model_id,
                "use_extra": bool(self.profile.use_extra),
                "action_chunk": int(self.profile.action_chunk),
                "do_sample": False,
                "unnorm_key": None,
                "denormalization_boundary": "plumb_policy_action_normalizer",
                "gpu_peak_memory_bytes": self.last_gpu_peak_memory_bytes,
            },
        )
