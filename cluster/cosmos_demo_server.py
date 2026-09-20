"""Private Cosmos chunk worker with actual diffusion-step NDJSON updates."""
import argparse
import base64
import hashlib
import hmac
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
from plumb.adapters.worlds import Cosmos3NanoDiffusersAdapter, Cosmos3NanoDiffusersProfile

parser = argparse.ArgumentParser()
parser.add_argument("--token-file", required=True)
parser.add_argument("--port", type=int, default=8919)
args = parser.parse_args()
token = Path(args.token_file).read_text().strip()
profile = Cosmos3NanoDiffusersProfile(profile_id="cosmos-live-drawer-256-30", local_model_path="/scratch/lchow432/plumb/models/nvidia--Cosmos3-Nano", resolution_tier=256)
adapter = Cosmos3NanoDiffusersAdapter(profile)
runtime = adapter._load_runtime()
load_started = time.perf_counter()
pipeline = adapter._ensure_pipeline(runtime)
load_seconds = time.perf_counter() - load_started
lock = threading.Lock()
calls = 0

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def authorized(self):
        return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

    def do_GET(self):
        if not self.authorized():
            self.send_error(401); return
        if self.path != "/health":
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"ready": True, "world_model": "cosmos", "model": profile.model_id, "revision": profile.model_revision}).encode())

    def do_POST(self):
        global calls
        if not self.authorized():
            self.send_error(401); return
        if self.path != "/cosmos":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length < 8 * 1024 * 1024:
            self.send_error(422); return
        try:
            request = json.loads(self.rfile.read(length))
            raw = base64.b64decode(request["png_base64"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == request["sha256"].removeprefix("sha256:")
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            rows = request["actions"]
            assert len(rows) == 16 and all(len(row) == 10 and all(math.isfinite(float(v)) for v in row) for row in rows)
            assert request["prompt"] in {"Close the drawer", "Put the pot to the left of the purple item."}
            resolution = 480 if request["prompt"].startswith("Put the pot") else 256
        except Exception:
            self.send_error(422); return
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
        def emit(value):
            self.wfile.write((json.dumps(value) + "\n").encode()); self.wfile.flush()
        with lock:
            started = time.perf_counter()
            try:
                emit({"kind": "progress", "stage": "preparing", "step": 0, "total": 30})
                action = runtime.action_condition_cls(mode="forward_dynamics", chunk_size=16, domain_name="bridge_orig_lerobot", resolution_tier=resolution, raw_actions=runtime.torch.as_tensor(rows, dtype=runtime.torch.float32), image=image, view_point=profile.view_point)
                runtime.torch.cuda.reset_peak_memory_stats()
                def progress(pipe, index, timestep, values):
                    emit({"kind": "progress", "stage": "denoising", "step": index + 1, "total": 30})
                    return values
                result = pipeline(prompt=request["prompt"], action=action, fps=5.0, num_inference_steps=30, guidance_scale=1.0, generator=adapter._seeded_generator(runtime, int(request["seed"])), output_type="np", use_system_prompt=False, callback_on_step_end=progress, callback_on_step_end_tensor_inputs=[])
                runtime.torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                frames = adapter._frames_from_result(result)
                if len(frames) != 17:
                    raise ValueError("Cosmos did not return 17 frames including conditioning")
                timing = {"world_seconds": elapsed, "model_load_seconds": load_seconds if calls == 0 else None, "peak_gpu_bytes": int(runtime.torch.cuda.max_memory_allocated())}
                emit({"kind": "progress", "stage": "encoding", "step": 30, "total": 30})
                for index, frame in enumerate(frames[1:], start=1):
                    array = np.asarray(frame)
                    if np.issubdtype(array.dtype, np.floating):
                        if not np.isfinite(array).all(): raise ValueError("Nonfinite generated pixels")
                        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
                    png = io.BytesIO(); Image.fromarray(array).save(png, format="PNG")
                    data = png.getvalue()
                    emit({"kind": "frame", "index": index, "png_base64": base64.b64encode(data).decode(), "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "timing": timing})
                calls += 1
                emit({"kind": "terminal", "status": "completed", "timing": timing, "model": profile.model_id, "revision": profile.model_revision})
            except Exception as error:
                emit({"kind": "terminal", "status": "failed", "reason": type(error).__name__ + ": " + str(error)[:500]})

print(json.dumps({"status": "ready", "model": profile.model_id, "load_seconds": load_seconds}), flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
