"""Stateless one-action public SuSIE_LL gc_bc diagnostic for Truss.

It is intentionally not a PLUMB benchmark worker.  It accepts two bounded
PNG inputs, invokes exactly one released low-level action call, and returns an
unqualified engineering record.  It never downloads an input URL, accepts a
checkpoint/model override, imports SuSIE diffusion, or restores optimizer
state.
"""

import base64
import ctypes
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


SCHEMA_VERSION = 1
MODEL_REPO = "patreya/gcbc-bridge"
MODEL_REVISION = "1a4c15dd9ad780a257e9494f0fac79cbe8e64793"
CHECKPOINT_FILE = "checkpoint/checkpoint"
README_FILE = "README.md"
COMMIT_MARKER_FILE = "checkpoint/commit_success.txt"
CHECKPOINT_SHA256 = "80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33"
README_SHA256 = "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17"
SOAR_REPO = "https://github.com/rail-berkeley/soar.git"
SOAR_REVISION = "eabd5f16a856e484884a22e257a941bb358cea08"
SOAR_SUBDIRECTORY = "model_training"
MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_PROMPT_BYTES = 1024
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

ACT_MEAN = (1.9296819e-04, 1.3667766e-04, -1.4583133e-04, -1.8390431e-04, -3.0808983e-04, 2.7425270e-04)
ACT_STD = (0.00912848, 0.0127196, 0.01229497, 0.02606696, 0.02875283, 0.07807977)


class ContractError(ValueError):
    pass


class RuntimeLoadError(RuntimeError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_base(value):
    return str(value).split("+", 1)[0]


def _cuda_probe():
    """Compact, path-only venv CUDA inventory for fail-closed startup errors."""
    names = {"cublas": "cublas", "cuda_cupti": "cupti", "cuda_nvrtc": "nvrtc", "cuda_runtime": "cudart", "cudnn": "cudnn", "cufft": "cufft", "cusolver": "cusolver", "cusparse": "cusparse", "nccl": "nccl", "nvjitlink": "nvJitLink"}
    roots = [Path(item) for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item.startswith("/opt/plumb-ml-venv/")]
    records = []
    for name, library_basename in names.items():
        matches = []
        for root in roots:
            matches.extend(sorted(root.glob("lib%s.so*" % library_basename)))
        if not matches:
            records.append({"library": name, "path": None, "cdll": "missing"})
            continue
        path = matches[0]
        try:
            ctypes.CDLL(str(path))
            status = "ok"
        except OSError as error:
            status = "error:" + str(error)[:160]
        records.append({"library": name, "path": str(path), "cdll": status})
    return records


def _strict_mapping(expected, actual, path="params"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError("checkpoint %s must be a mapping" % path)
        expected_keys = set(str(key) for key in expected)
        actual_keys = set(str(key) for key in actual)
        if expected_keys != actual_keys:
            raise RuntimeError("checkpoint %s keys differ; missing=%s extra=%s" % (path, sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)))
        for key, value in expected.items():
            _strict_mapping(value, actual[str(key)], path + "." + str(key))


def _strict_shapes(jax, expected, restored):
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    restored_leaves, restored_tree = jax.tree_util.tree_flatten(restored)
    if expected_tree != restored_tree or len(expected_leaves) != len(restored_leaves):
        raise RuntimeError("checkpoint parameter pytree differs from initialized source model")
    for index, (left, right) in enumerate(zip(expected_leaves, restored_leaves)):
        left, right = jax.device_get(left), jax.device_get(right)
        if tuple(left.shape) != tuple(right.shape) or str(left.dtype) != str(right.dtype):
            raise RuntimeError("checkpoint parameter leaf %d shape/dtype differs from initialized source model" % index)


def _tree_digest(jax, tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jax.device_get(leaf)
        digest.update((repr(tuple(value.shape)) + "|" + str(value.dtype) + "|").encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _decode_png(value, label):
    if not isinstance(value, str) or not value:
        raise ContractError("%s must be a nonempty base64 PNG string" % label)
    if len(value) > ((MAX_PNG_BYTES + 2) // 3) * 4:
        raise ContractError("%s base64 length is outside the allowed bound" % label)
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as error:
        raise ContractError("%s is not valid base64" % label) from error
    if not raw or len(raw) > MAX_PNG_BYTES:
        raise ContractError("%s PNG byte length is outside the allowed bound" % label)
    try:
        from PIL import Image
        import numpy as np
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ContractError("%s must be PNG, not %s" % (label, image.format))
            if image.width != 256 or image.height != 256:
                raise ContractError("%s declared dimensions must be 256x256 before decode" % label)
            image.load()
            pixels = np.asarray(image.convert("RGB").copy())
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("%s is not a decodable PNG" % label) from error
    if tuple(pixels.shape) != (256, 256, 3) or str(pixels.dtype) != "uint8":
        raise ContractError("%s must decode to RGB uint8 (256, 256, 3)" % label)
    return raw, pixels


class PolicyRuntime:
    def __init__(self, **kwargs):
        self._root = Path(os.environ.get("PLUMB_MVP_MODEL_CACHE", "/tmp/plumb-susie-ll-gcbc"))
        self._agent = None
        self._runtime = None
        self._load_seconds = None
        self._restore = None
        self._lock = threading.Lock()

    def _verify_soar_source(self):
        source = Path(os.environ.get("PLUMB_SOAR_SOURCE_ROOT", "/app/source-soar-gcbc"))
        module_root = source / SOAR_SUBDIRECTORY
        if not source.is_dir() or not (module_root / "jaxrl_m" / "agents" / "continuous" / "gc_bc.py").is_file():
            raise RuntimeError("pinned SOAR source tree is absent")
        try:
            head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except Exception as error:
            raise RuntimeError("pinned SOAR source checkout cannot be inspected") from error
        if head != SOAR_REVISION or dirty:
            raise RuntimeError("SOAR source checkout is not the pinned clean revision")
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        return module_root

    def load(self):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        started = time.perf_counter()
        source_root = self._verify_soar_source()
        # Explicit package facts make cloud/cluster runtime drift observable.
        # Never log arbitrary environment variables or credential material.
        package_versions = {}
        for name in ("jax", "jaxlib", "flax", "numpy", "tensorflow", "protobuf",
                     "nvidia-cusolver-cu12", "nvidia-cublas-cu12", "nvidia-nvjitlink-cu12",
                     "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"):
            try:
                package_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                package_versions[name] = "missing"
        print("PLUMB_MVP_RUNTIME " + json.dumps(package_versions, sort_keys=True), flush=True)
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            import jax
            import flax
            import distrax
            import numpy as np
            from flax import serialization
            from flax.training import checkpoints
            from huggingface_hub import hf_hub_download
            from jaxrl_m.agents import agents
            from jaxrl_m.vision import encoders
        except Exception as error:
            raise RuntimeLoadError("MVP runtime dependencies are unavailable", {"packages": package_versions, "cuda_probe": _cuda_probe()}) from error
        if _version_base(jax.__version__) != "0.4.20" or _version_base(flax.__version__) != "0.7.5" or _version_base(distrax.__version__) != "0.1.5":
            raise RuntimeError("MVP runtime JAX/Flax/Distrax versions do not match pinned source requirements")
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeLoadError("MVP requires a JAX-visible GPU", {"packages": package_versions, "jax_devices": [str(device) for device in jax.devices()], "cuda_probe": _cuda_probe()})
        agent_source = agents["gc_bc"]
        encoder_source = encoders["resnetv1-34"]
        while hasattr(encoder_source, "func"):
            encoder_source = encoder_source.func
        if not str(Path(inspect.getfile(agent_source)).resolve()).startswith(str(source_root.resolve())) or not str(Path(inspect.getfile(encoder_source)).resolve()).startswith(str(source_root.resolve())):
            raise RuntimeError("installed jaxrl_m does not resolve inside pinned SOAR source tree")
        self._root.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(hf_hub_download(MODEL_REPO, CHECKPOINT_FILE, revision=MODEL_REVISION, local_dir=str(self._root), local_dir_use_symlinks=False))
        readme = Path(hf_hub_download(MODEL_REPO, README_FILE, revision=MODEL_REVISION, local_dir=str(self._root), local_dir_use_symlinks=False))
        marker = Path(hf_hub_download(MODEL_REPO, COMMIT_MARKER_FILE, revision=MODEL_REVISION, local_dir=str(self._root), local_dir_use_symlinks=False))
        if _sha256_file(checkpoint) != CHECKPOINT_SHA256 or _sha256_file(readme) != README_SHA256:
            raise RuntimeError("downloaded MVP artifact hash differs from immutable declaration")
        if "license: mit" not in readme.read_text(encoding="utf-8").lower() or not marker.is_file():
            raise RuntimeError("publisher license/provenance marker is absent from pinned artifact")
        encoder = encoders["resnetv1-34"](pooling_method="avg", add_spatial_coordinates=False, act="swish")
        batch = {"observations": {"proprio": np.zeros((1, 7)), "image": np.zeros((1, 256, 256, 3))}, "goals": {"image": np.zeros((1, 256, 256, 3))}, "actions": np.zeros((1, 7))}
        _, construct_rng = jax.random.split(jax.random.PRNGKey(42))
        agent = agents["gc_bc"].create(
            rng=construct_rng, observations=batch["observations"], goals=batch["goals"], actions=batch["actions"], encoder_def=encoder,
            early_goal_concat=True, shared_goal_encoder=True, use_proprio=False, learning_rate=3e-4, warmup_steps=2000, decay_steps=int(2e6),
            network_kwargs={"hidden_dims": (256, 256, 256), "dropout_rate": 0.1}, policy_kwargs={"tanh_squash_distribution": False, "std_parameterization": "fixed", "fixed_std": [1, 1, 1, 1, 1, 1, 0.1]},
        )
        raw = checkpoints.restore_checkpoint(str(checkpoint), target=None)
        if not isinstance(raw, dict) or set(raw) != {"state"} or not isinstance(raw["state"], dict):
            raise RuntimeError("checkpoint raw schema is not the reviewed inference-only state")
        state = raw["state"]
        if set(state) != {"opt_states", "params", "rng", "step", "target_params"} or state["target_params"] is not None or agent.state.target_params is not None:
            raise RuntimeError("checkpoint target/optimizer schema is not eligible for params-only inference")
        initial_params_digest = _tree_digest(jax, agent.state.params)
        expected = serialization.to_state_dict(agent.state.params)
        _strict_mapping(expected, state["params"])
        params = serialization.from_state_dict(agent.state.params, state["params"])
        _strict_shapes(jax, agent.state.params, params)
        agent = agent.replace(state=agent.state.replace(params=params, target_params=None))
        restored_params_digest = _tree_digest(jax, agent.state.params)
        if initial_params_digest == restored_params_digest:
            raise RuntimeError("params-only restore left initialized parameters byte-identical")
        self._agent = agent
        self._runtime = {"jax": jax, "numpy": np}
        self._runtime_payload = {"jax_version": jax.__version__, "flax_version": flax.__version__, "distrax_version": distrax.__version__, "tensorflow_version": tf.__version__, "tensorflow_gpu_visible": False}
        self._restore = {"method": "inference_params_only_source_checkpoint_no_optimizer_restore", "optimizer_state_excluded": True, "inference_only_not_resumable": True, "raw_checkpoint_keys": {"top_level": sorted(raw), "state": sorted(state)}, "initialized_params_sha256": initial_params_digest, "restored_params_sha256": restored_params_digest, "changed": True}
        self._load_seconds = time.perf_counter() - started

    def predict(self, request):
        if not isinstance(request, dict) or set(request) != {"schema_version", "request_id", "current_png_base64", "goal_png_base64", "prompt"}:
            raise ContractError("request must contain exactly schema_version, request_id, current_png_base64, goal_png_base64, prompt")
        if isinstance(request["schema_version"], bool) or request["schema_version"] != SCHEMA_VERSION or not isinstance(request["request_id"], str) or not REQUEST_ID_RE.fullmatch(request["request_id"]):
            raise ContractError("request schema_version or request_id is invalid")
        if not isinstance(request["prompt"], str) or not request["prompt"].strip() or len(request["prompt"].encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ContractError("prompt is missing or outside byte bounds")
        current_raw, current = _decode_png(request["current_png_base64"], "current_png_base64")
        goal_raw, goal = _decode_png(request["goal_png_base64"], "goal_png_base64")
        if current.tobytes() == goal.tobytes():
            raise ContractError("goal image must differ from current image")
        if self._agent is None or self._runtime is None:
            raise RuntimeError("model is not loaded")
        with self._lock:
            started = time.perf_counter()
            raw = self._agent.sample_actions({"image": current[None, ...]}, {"image": goal[None, ...]}, temperature=0.0, argmax=True, seed=None)
            if isinstance(raw, tuple):
                if len(raw) != 2:
                    raise RuntimeError("source gc_bc tuple output is malformed")
                raw, mode = raw
                mode = self._runtime["numpy"].asarray(self._runtime["jax"].device_get(mode))
                if mode.shape != (1, 7) or not self._runtime["numpy"].isfinite(mode).all():
                    raise RuntimeError("source gc_bc action_mode is malformed")
            action = self._runtime["numpy"].asarray(self._runtime["jax"].device_get(raw))
            if action.shape != (1, 7) or not self._runtime["numpy"].isfinite(action).all():
                raise RuntimeError("source gc_bc action is malformed")
            normalized = [float(item) for item in action[0]]
            physical = [normalized[index] * ACT_STD[index] + ACT_MEAN[index] for index in range(6)] + [1.0 if normalized[6] > 0.0 else 0.0]
            inference_seconds = time.perf_counter() - started
        return {"schema_version": SCHEMA_VERSION, "status": "completed_unqualified", "qualified": False, "reason": "One static-goal source-native gc_bc action only; not a rollout, task success, benchmark, Gate result, or throughput measurement.", "request_id": request["request_id"], "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "soar_revision": SOAR_REVISION, "soar_subdirectory": SOAR_SUBDIRECTORY, "restore": self._restore}, "inputs": {"current_png_sha256": _sha256(current_raw), "current_pixels_sha256": _sha256(current.tobytes()), "goal_png_sha256": _sha256(goal_raw), "goal_pixels_sha256": _sha256(goal.tobytes()), "prompt_sha256": _sha256(request["prompt"].encode("utf-8"))}, "actions": {"native_model_normalized": normalized, "transformed_physical": [physical], "shape": [1, 7], "finite": True}, "timing": {"load_seconds_once": self._load_seconds, "inference_seconds": inference_seconds}, "runtime": self._runtime_payload}
