"""Offline contracts for the authenticated one-tick live-demo GPU worker."""
from __future__ import annotations

import base64
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from cluster.live_demo_server import (
    EXPERIMENTAL_WARNING,
    LiveDemoInputError,
    LiveDemoService,
    MANUAL_DIRECTIONS,
    STATE_MODE,
    _manual_action,
    _warm_real_adapters,
    create_http_server,
    parse_step_request,
)


class FakePolicy:
    def __init__(self):
        self.calls = []

    def action_for(self, image, prompt, timestamp):
        self.calls.append((image, prompt, timestamp))
        return (0.01, 0, 0, 0, 0, 0, 1), {"backend_calls": 1, "native": True}


class FakeWorld:
    def __init__(self):
        self.calls = []

    def generate_one_step(self, image, action, seed):
        self.calls.append((image, tuple(action), seed))
        return SimpleNamespace(
            frames=("echo:" + str(image), "generated:" + str(image)),
            timing={"backend_calls": 1, "wall_seconds": 0.012},
            native_action_scaled=tuple(action),
        )


def encode(frame):
    return "encoded:" + str(frame)


def decode(payload):
    return "decoded:" + payload


def service(policy=None, world=None):
    return LiveDemoService(
        policy=policy or FakePolicy(),
        world=world or FakeWorld(),
        vendor_frame="vendor-frame-0",
        model_info={
            "model": {"policy": "OpenVLA", "world": "IRASim"},
            "revision": {"openvla": "pinned", "irasim_source": "pinned"},
            "integrity": {"irasim_checkpoint": {"expected_sha256": "sha256:abc"}},
            "source_release": {"status": "verified"},
        },
        load_timing={"total_load_seconds": 1.0, "gpu_name": "H100"},
        png_decoder=decode,
        png_encoder=encode,
        frame_converter=lambda frame: "rgb:" + frame,
    )


def test_policy_step_calls_openvla_once_and_real_world_once_per_request_without_action_cache():
    policy = FakePolicy()
    world = FakeWorld()
    worker = service(policy, world)

    first = worker.step({"mode": "policy", "prompt": "Move the pot."})
    second = worker.step({"mode": "policy", "prompt": "Move the pot.", "png_base64": "prior-png", "seed": 9})

    assert [call[0] for call in policy.calls] == ["vendor-frame-0", "decoded:prior-png"]
    assert len(world.calls) == 2
    assert world.calls[0][0] == "vendor-frame-0"
    assert world.calls[1][0] == "decoded:prior-png"
    assert first["native_policy_calls"] == second["native_policy_calls"] == 1
    assert first["world_calls"] == second["world_calls"] == 1
    assert second["seed"] == 9
    assert first["predicted_png_base64"] == "encoded:rgb:generated:vendor-frame-0"
    assert second["source"] == "provided_png_base64"
    assert first["world_post_frame0_dropped"] is True
    assert first["state_mode"] == STATE_MODE
    assert "not native latent carry" in first["warning"]


def test_manual_direction_is_a_clamped_native_7d_action_and_never_wakes_policy():
    policy = FakePolicy()
    world = FakeWorld()
    worker = service(policy, world)

    response = worker.step({"mode": "manual", "direction": "left", "seed": 7})

    assert policy.calls == []
    assert response["native_policy_calls"] == 0
    assert response["action_origin"] == "manual_clamped_direction"
    assert response["action7"] == [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert world.calls == [("vendor-frame-0", tuple(response["action7"]), 7)]
    assert _manual_action("up") == (0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(LiveDemoInputError, match="manual direction"):
        _manual_action("stop")


def test_health_and_fixture_are_cached_read_only_and_never_call_models():
    policy = FakePolicy()
    world = FakeWorld()
    worker = service(policy, world)

    health = worker.health()
    fixture = worker.fixture()

    assert policy.calls == [] and world.calls == []
    assert health["status"] == "ready"
    assert health["manual_directions"] == list(MANUAL_DIRECTIONS)
    assert health["load_timing"]["gpu_name"] == "H100"
    assert fixture["fixture_png_base64"] == "encoded:vendor-frame-0"
    assert fixture["source"] == "vendor_fixture_frame_0_hash_bound"
    assert fixture["warning"] == EXPERIMENTAL_WARNING


def test_parser_accepts_client_png_aliases_but_refuses_ambiguous_images_and_arbitrary_fields():
    for key in ("png_base64", "current_png_base64", "fixture_png_base64", "image_base64"):
        parsed = parse_step_request({"mode": "policy", key: "png"})
        assert parsed.png_base64 == "png"
    with pytest.raises(LiveDemoInputError, match="at most one"):
        parse_step_request({"png_base64": "a", "image_base64": "b"})
    with pytest.raises(LiveDemoInputError, match="unsupported"):
        parse_step_request({"actions": [0] * 7})
    with pytest.raises(LiveDemoInputError, match="direction"):
        parse_step_request({"mode": "manual", "direction": "rotate_left"})
    with pytest.raises(LiveDemoInputError, match="signed 64-bit"):
        parse_step_request({"seed": True})


def test_png_codec_requires_bounded_single_png_rgb():
    np = pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    from cluster.live_demo_server import _decode_png_base64, _encode_png_base64

    frame = np.full((2, 3, 3), 123, dtype=np.uint8)
    payload = _encode_png_base64(frame)
    decoded = _decode_png_base64(payload)
    assert decoded.shape == (2, 3, 3)
    assert decoded.dtype == np.uint8
    assert int(decoded[0, 0, 0]) == 123
    jpeg_stream = __import__("io").BytesIO()
    PIL.fromarray(frame).save(jpeg_stream, format="JPEG")
    with pytest.raises(LiveDemoInputError, match="PNG"):
        _decode_png_base64(base64.b64encode(jpeg_stream.getvalue()).decode("ascii"))


def test_http_worker_requires_bearer_token_and_exposes_only_the_small_contract():
    worker = service()
    token = "t" * 24
    # Port zero lets the OS choose a local ephemeral test endpoint.
    config = SimpleNamespace(host="127.0.0.1", port=0, max_request_bytes=4096)
    server = create_http_server(config, worker, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        assert response.getheader("WWW-Authenticate") == "Bearer"

        connection.request("GET", "/health", headers={"Authorization": "Bearer " + token})
        response = connection.getresponse()
        health = json.loads(response.read())
        assert response.status == 200
        assert health["manual_directions"] == ["up", "down", "left", "right", "forward", "back"]

        body = json.dumps({"mode": "manual", "direction": "right", "seed": 2})
        connection.request(
            "POST",
            "/step",
            body=body,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert result["action7"] == [0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert result["native_policy_calls"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_warmup_loads_components_only_and_not_a_policy_prediction():
    class Adapter:
        def __init__(self):
            self.runtime_calls = 0
            self.component_calls = 0

        def _load_runtime(self):
            self.runtime_calls += 1
            return object()

        def _ensure_components(self, runtime):
            assert runtime is not None
            self.component_calls += 1

    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    class World:
        def __init__(self):
            self.load_calls = 0

        def _load(self):
            self.load_calls += 1
            return Torch(), object()

    adapter = Adapter()
    wrapper = SimpleNamespace(adapter=adapter)
    world = World()
    timing = _warm_real_adapters(wrapper, world)
    assert adapter.runtime_calls == adapter.component_calls == world.load_calls == 1
    assert timing["gpu_name"] is None
