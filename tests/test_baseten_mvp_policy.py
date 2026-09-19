"""CPU contract tests for the standalone Baseten SuSIE_LL MVP package."""

import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "deploy" / "baseten" / "mvp_policy" / "model" / "model.py"
SPEC = importlib.util.spec_from_file_location("plumb_mvp_policy_model", MODEL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _png(value):
    buffer = io.BytesIO()
    Image.fromarray(value, "RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeJax:
    @staticmethod
    def device_get(value):
        return value


class FakeAgent:
    def __init__(self, output, tuple_output=False):
        self.output = output
        self.tuple_output = tuple_output
        self.calls = []

    def sample_actions(self, observations, goals, **kwargs):
        self.calls.append((observations, goals, kwargs))
        return (self.output, self.output.copy()) if self.tuple_output else self.output


def _loaded_model(output=None, tuple_output=False):
    model = MODULE.Model()
    model._agent = FakeAgent(
        output if output is not None else np.asarray([[0.1, -0.2, 0.0, 0.5, -0.5, 0.25, 0.7]], dtype=np.float32),
        tuple_output=tuple_output,
    )
    model._runtime = {"jax": FakeJax(), "numpy": np}
    model._runtime_payload = {"jax_version": "0.4.20", "flax_version": "0.7.5", "distrax_version": "0.1.5", "tensorflow_version": "2.15.0", "tensorflow_gpu_visible": False}
    model._load_seconds = 1.25
    model._restore = {"method": "inference_params_only_source_checkpoint_no_optimizer_restore", "optimizer_state_excluded": True, "changed": True}
    return model


def _request(**changes):
    current = np.zeros((256, 256, 3), dtype=np.uint8)
    goal = np.full((256, 256, 3), 5, dtype=np.uint8)
    payload = {
        "schema_version": 1,
        "request_id": "mvp-test-1",
        "current_png_base64": _png(current),
        "goal_png_base64": _png(goal),
        "prompt": "static diagnostic",
    }
    payload.update(changes)
    return payload


def test_mvp_response_is_json_safe_unqualified_and_has_one_boundary_physical_action():
    model = _loaded_model(tuple_output=True)

    response = model.predict(_request())

    assert json.loads(json.dumps(response))["runtime"]["jax_version"] == "0.4.20"
    assert response["schema_version"] == 1
    assert response["status"] == "completed_unqualified" and response["qualified"] is False
    assert response["actions"]["shape"] == [1, 7]
    assert response["actions"]["native_model_normalized"][-1] == pytest.approx(0.7)
    assert response["actions"]["transformed_physical"] == [[pytest.approx(0.00110581619), pytest.approx(-0.00240724234), pytest.approx(-0.00014583133), pytest.approx(0.01284957569), pytest.approx(-0.01468450483), pytest.approx(0.0197941952), 1.0]]
    assert response["model"]["restore"]["optimizer_state_excluded"] is True
    assert model._agent.calls[0][2] == {"temperature": 0.0, "argmax": True, "seed": None}


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"schema_version": True}, "schema_version"),
        ({"request_id": "https://not-an-id"}, "request_id"),
        ({"goal_png_base64": _png(np.zeros((256, 256, 3), dtype=np.uint8))}, "differ"),
        ({"current_png_base64": "https://input.invalid/a.png"}, "base64"),
        ({"extra": "not-allowed"}, "exactly"),
    ],
)
def test_mvp_rejects_malformed_url_substituted_duplicate_or_extra_inputs(changes, message):
    with pytest.raises(MODULE.ContractError, match=message):
        _loaded_model().predict(_request(**changes))


def test_mvp_png_decoder_rejects_non_256_image_before_full_decode():
    small = _png(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(MODULE.ContractError, match="declared dimensions"):
        _loaded_model().predict(_request(current_png_base64=small))
