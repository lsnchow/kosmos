"""Private Baseten runtime for the exact semantic epoch-02 Qwen judge adapter.

The public request surface accepts only the blinded assessment contract.  It
does not accept a policy name, action transcript, expected score, browser URL,
or an adapter selector.  A load failure is returned as a failure; it never
falls back to bare Qwen or the formatting-only adapter.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


MODEL_ROOT = Path("/app/model")
ADAPTER_DIR = MODEL_ROOT / "adapter"
MANIFEST_PATH = MODEL_ROOT / "adapter-manifest.json"
RUNTIME_ROOT = MODEL_ROOT
ADAPTER_ID = "semantic_pilot_epoch_02"
ADAPTER_TREE_SHA256 = "sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BASE_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 4_000_000
REQUEST_KIND = "plumb_demo_semantic_judge_v1"


class JudgeWorkerError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tree_hash(directory: Path) -> str:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise JudgeWorkerError("adapter tree contains a symlink")
        if path.is_file():
            records.append({"path": path.relative_to(directory).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size})
    if not records:
        raise JudgeWorkerError("adapter tree contains no regular files")
    return "sha256:" + hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str):
        raise JudgeWorkerError("missing SHA-256 value")
    value = value.lower().removeprefix("sha256:")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise JudgeWorkerError("invalid SHA-256 value")
    return "sha256:" + value


def _png(value: Mapping[str, Any], Image: Any) -> Any:
    encoded, expected = value.get("png_base64"), _digest(value.get("sha256"))
    if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 128:
        raise JudgeWorkerError("frame payload is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as error:
        raise JudgeWorkerError("frame payload is not base64") from error
    if len(raw) > MAX_IMAGE_BYTES or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise JudgeWorkerError("frame payload is not a bounded PNG")
    actual = "sha256:" + hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise JudgeWorkerError("frame transport bytes do not match their SHA-256")
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        if image.width * image.height > MAX_PIXELS:
            raise JudgeWorkerError("frame exceeds pixel safety limit")
        return image.convert("RGB")
    except JudgeWorkerError:
        raise
    except Exception as error:
        raise JudgeWorkerError("frame PNG cannot be decoded") from error


class Model:
    def __init__(self, **_: Any):
        self.judge = None
        self.classes: Dict[str, Any] = {}
        self.receipt: Dict[str, Any] = {}
        self.load_seconds: float | None = None

    def load(self) -> None:
        started = time.perf_counter()
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.is_symlink() or not ADAPTER_DIR.is_dir() or ADAPTER_DIR.is_symlink():
            raise JudgeWorkerError("semantic epoch-02 adapter staging is absent")
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JudgeWorkerError("semantic adapter manifest is unreadable") from error
        if not isinstance(manifest, Mapping) or manifest.get("adapter_id") != ADAPTER_ID or manifest.get("base_model_id") != BASE_MODEL_ID or manifest.get("base_model_revision") != BASE_MODEL_REVISION:
            raise JudgeWorkerError("semantic adapter manifest does not bind the expected base and adapter")
        if _tree_hash(ADAPTER_DIR) != ADAPTER_TREE_SHA256:
            raise JudgeWorkerError("semantic adapter tree hash does not match epoch-02")
        # The staged runtime is deliberately minimal: it is the existing
        # QwenRubricJudge plus only its pure-Python dependencies, copied by the
        # staging script and bound to the artifact receipt below.
        if str(RUNTIME_ROOT) not in sys.path:
            sys.path.insert(0, str(RUNTIME_ROOT))
        try:
            import torch
            import transformers
            from PIL import Image
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from plumb.policies.judge import (
                JudgeInputProvenance,
                JudgeRequest,
                JudgeSamplingConfig,
                QwenJudgeProfile,
                QwenRubricJudge,
                ReferenceImage,
            )
        except ImportError as error:
            raise JudgeWorkerError("judge runtime dependency is unavailable") from error
        if str(transformers.__version__) != "4.49.0":
            raise JudgeWorkerError("Transformers runtime does not match the pinned 4.49.0 judge profile")
        if not torch.cuda.is_available():
            raise JudgeWorkerError("semantic judge requires an H100 CUDA runtime")
        try:
            base_path = Path(snapshot_download(BASE_MODEL_ID, revision=BASE_MODEL_REVISION, local_files_only=False)).resolve(strict=True)
        except Exception as error:
            raise JudgeWorkerError("pinned Qwen base snapshot could not be resolved") from error
        # Hugging Face snapshots are content-addressed at the commit in their
        # resolved path. The check makes a branch/tag substitution impossible.
        if base_path.name != BASE_MODEL_REVISION:
            raise JudgeWorkerError("resolved Qwen snapshot does not expose the pinned commit directory")
        dtype = torch.bfloat16
        processor = transformers.AutoProcessor.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False)
        base = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=dtype
        )
        try:
            model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), adapter_name=ADAPTER_ID, is_trainable=False, local_files_only=True)
            model.set_adapter(ADAPTER_ID)
        except Exception as error:
            raise JudgeWorkerError("semantic epoch-02 adapter could not be loaded and enabled") from error
        model = model.to("cuda")
        model.eval()
        active = getattr(model, "active_adapter", None)
        active_value = active[0] if isinstance(active, (list, tuple)) and active else active
        if active_value != ADAPTER_ID:
            raise JudgeWorkerError("semantic epoch-02 adapter is not active after load")
        profile = QwenJudgeProfile(
            profile_id="plumb-qwen25vl-semantic-epoch-02",
            local_model_path=str(base_path),
            model_revision=BASE_MODEL_REVISION,
            processor_revision=BASE_MODEL_REVISION,
            transformers_version="4.49.0",
            asset_manifest_id=ADAPTER_ID,
            asset_manifest_sha256="sha256:" + hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            runtime_lock_id="requirements.txt",
            runtime_lock_sha256="sha256:" + hashlib.sha256((MODEL_ROOT.parent / "requirements.txt").read_bytes()).hexdigest(),
        )
        # Passing the PEFT model via QwenRubricJudge's supported factory keeps
        # its original prompt, video/reference protocol, seeded five samples,
        # schema retry boundary, and aggregation implementation intact.
        self.judge = QwenRubricJudge(
            profile,
            sampling=JudgeSamplingConfig(),
            model_factory=lambda _profile, _runtime: model,
            processor_factory=lambda _profile, _runtime: processor,
        )
        self.classes = {
            "Image": Image,
            "JudgeRequest": JudgeRequest,
            "ReferenceImage": ReferenceImage,
            "JudgeInputProvenance": JudgeInputProvenance,
        }
        self.load_seconds = time.perf_counter() - started
        self.receipt = {
            "adapter_id": ADAPTER_ID,
            "adapter_tree_sha256": ADAPTER_TREE_SHA256,
            "active_adapter": ADAPTER_ID,
            "adapter_enabled": True,
            "base_model_id": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "processor_revision": BASE_MODEL_REVISION,
            "transformers_version": str(transformers.__version__),
            "peft_model_class": type(model).__name__,
            "adapter_manifest_sha256": "sha256:" + hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        }

    def _validate_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        required = {"schema_version", "kind", "request_id", "frames", "frame_timestamps", "reference_images", "task_id", "seeds", "provenance"}
        if set(request) != required or request.get("schema_version") != 1 or request.get("kind") != REQUEST_KIND:
            raise JudgeWorkerError("request schema is invalid")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id.startswith("judge-") or len(request_id) > 128:
            raise JudgeWorkerError("request ID is invalid")
        if request.get("task_id") != "close_drawer":
            raise JudgeWorkerError("only the frozen Close the drawer task is supported")
        frames = request.get("frames")
        timestamps = request.get("frame_timestamps")
        references = request.get("reference_images")
        seeds = request.get("seeds")
        provenance = request.get("provenance")
        if not isinstance(frames, list) or len(frames) != 16 or not isinstance(timestamps, list) or len(timestamps) != 16:
            raise JudgeWorkerError("judge request needs exactly 16 frames and timestamps")
        if not isinstance(references, list) or len(references) != 1 or not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
            raise JudgeWorkerError("judge request has invalid reference or seed count")
        if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
            raise JudgeWorkerError("judge sampling seeds are invalid")
        try:
            converted_timestamps = [float(value) for value in timestamps]
        except (TypeError, ValueError) as error:
            raise JudgeWorkerError("judge timestamps are invalid") from error
        if any(not math.isfinite(value) for value in converted_timestamps) or any(a >= b for a, b in zip(converted_timestamps, converted_timestamps[1:])):
            raise JudgeWorkerError("judge timestamps must be finite and strictly increasing")
        if not isinstance(provenance, Mapping) or set(provenance) != {"clip_id", "video_sha256", "protocol_id", "calibration_manifest_hash"}:
            raise JudgeWorkerError("judge provenance schema is invalid")
        reference = references[0]
        if not isinstance(reference, Mapping) or set(reference) != {"png_base64", "sha256", "source_uri", "role"} or reference.get("role") != "scene_reference_not_goal":
            raise JudgeWorkerError("judge reference must be the recorded scene reference, not a relabelled goal")
        return {"request_id": request_id, "frames": frames, "timestamps": converted_timestamps, "reference": reference, "seeds": seeds, "provenance": provenance}

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = request.get("request_id") if isinstance(request, Mapping) and isinstance(request.get("request_id"), str) else None
        try:
            if self.judge is None:
                raise JudgeWorkerError("judge was not loaded")
            checked = self._validate_request(request)
            Image = self.classes["Image"]
            frames = tuple(_png(frame, Image) for frame in checked["frames"])
            reference_raw = checked["reference"]
            reference_image = _png(reference_raw, Image)
            reference = self.classes["ReferenceImage"](
                image=reference_image,
                source_uri=reference_raw["source_uri"],
                sha256=_digest(reference_raw["sha256"]).removeprefix("sha256:"),
            )
            provenance = self.classes["JudgeInputProvenance"](**dict(checked["provenance"]))
            judge_request = self.classes["JudgeRequest"](
                frames=frames,
                frame_timestamps=tuple(checked["timestamps"]),
                reference_images=(reference,),
                task_id="close_drawer",
                provenance=provenance,
            )
            judge_request.validate()
            started = time.perf_counter()
            report = self.judge.evaluate(judge_request, seeds=checked["seeds"])
            report.validate_aggregation()
            return {
                "schema_version": 1,
                "request_id": checked["request_id"],
                "status": "completed",
                "report": report.as_dict(),
                "adapter_receipt": self.receipt,
                "timing": {
                    "worker_inference_seconds": time.perf_counter() - started,
                    "worker_load_seconds": self.load_seconds,
                    "gpu_peak_memory_bytes": getattr(report, "gpu_peak_memory_bytes", None),
                },
                "experimental": True,
                "qualified": False,
            }
        except Exception as error:
            return {
                "schema_version": 1,
                "request_id": request_id,
                "status": "failed",
                "reason": "%s: %s" % (type(error).__name__, str(error)[:1000]),
                "experimental": True,
                "qualified": False,
            }
