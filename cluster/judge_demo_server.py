"""Authenticated local-only semantic judge with real per-sample stream updates."""
import argparse
import base64
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import transformers
import peft
from PIL import Image
from peft import PeftModel
from plumb.policies.judge import QwenRubricJudge, QwenJudgeProfile, JudgeRequest, ReferenceImage, JudgeInputProvenance
from plumb.policies.demo_tasks import POT_TASK_ID, POT_INSTRUCTION, POT_RUBRIC
from deploy.baseten.judge_demo.model.model import _tree_hash, _png, ADAPTER_TREE_SHA256, ADAPTER_ID, BASE_MODEL_ID, BASE_MODEL_REVISION
from deploy.baseten.training.train_judge_lora import _local_model_binding

parser = argparse.ArgumentParser()
parser.add_argument("--token-file", required=True)
parser.add_argument("--port", type=int, default=8921)
args = parser.parse_args()
token = Path(args.token_file).read_text().strip()
root = Path("/scratch/lchow432/plumb")
model_root = root / "models/qwen-judge-training-view-cc594-v1"
adapter_path = root / "experiments/judge-lora-pilot-v1-trained/adapters/epoch-02"
assert transformers.__version__ == "4.49.0" and peft.__version__ == "0.14.0"
assert _tree_hash(adapter_path) == ADAPTER_TREE_SHA256
binding = _local_model_binding(str(model_root), str(model_root / "PLUMB-TRAINING-MANIFEST.json"), {"model_id":BASE_MODEL_ID, "model_revision":BASE_MODEL_REVISION})
started = time.perf_counter()
processor = transformers.AutoProcessor.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
base = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(model_root, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter_path, adapter_name=ADAPTER_ID, is_trainable=False, local_files_only=True).to("cuda")
model.set_adapter(ADAPTER_ID); model.eval()
load_seconds = time.perf_counter() - started
profile = QwenJudgeProfile(profile_id="semantic-epoch02-cluster-demo", local_model_path=str(model_root), model_revision=BASE_MODEL_REVISION, processor_revision=BASE_MODEL_REVISION, asset_manifest_id="verified-local-training-manifest", asset_manifest_sha256=binding["model_manifest_sha256"])
lock = threading.Lock()
calls = 0

def receipt():
    layers = [m for m in model.modules() if hasattr(m, "lora_A")]
    assert layers and all(not m.disable_adapters and not m.merged and ADAPTER_ID in m.active_adapters for m in layers)
    return {"adapter_id":ADAPTER_ID, "adapter_tree_sha256":ADAPTER_TREE_SHA256, "active_adapter":ADAPTER_ID, "adapter_enabled":True, "verified_enabled_layer_count":len(layers), "base_model_id":BASE_MODEL_ID, "base_model_revision":BASE_MODEL_REVISION, "processor_revision":BASE_MODEL_REVISION}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def authorized(self): return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)
    def do_GET(self):
        if not self.authorized(): self.send_error(401); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"ready":True, "provider":"Trillium", "gpu_type":torch.cuda.get_device_name(), "allocation_id":os.environ.get("SLURM_JOB_ID"), "adapter_receipt":receipt()}).encode())
    def do_POST(self):
        global calls
        if not self.authorized(): self.send_error(401); return
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size < 32*1024*1024: self.send_error(422); return
        payload = json.loads(self.rfile.read(size))
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
        def emit(v): self.wfile.write((json.dumps(v)+"\n").encode()); self.wfile.flush()
        with lock:
            try:
                adapter_receipt = receipt()
                class StreamingJudge(QwenRubricJudge):
                    def _sample(self, request, index, seed):
                        emit({"kind":"progress", "stage":"assessing", "completed_samples":index, "sample_count":5})
                        result = super()._sample(request, index, seed)
                        emit({"kind":"progress", "stage":"assessing", "completed_samples":index+1, "sample_count":5})
                        return result
                judge = StreamingJudge(profile, model_factory=lambda p,r:model, processor_factory=lambda p,r:processor)
                frames = tuple(_png(f,Image) for f in payload["frames"])
                refs = tuple(ReferenceImage(_png(r,Image), r["source_uri"], r["sha256"].removeprefix("sha256:")) for r in payload["reference_images"])
                task = payload["task_id"]
                if task not in {"close_drawer", POT_TASK_ID}: raise ValueError("Unsupported task")
                fields = {"diagnostic_mode":True,"task_instruction":POT_INSTRUCTION,"task_rubric":POT_RUBRIC} if task == POT_TASK_ID else {"task_id":task}
                request = JudgeRequest(frames,tuple(payload["frame_timestamps"]),refs,provenance=JudgeInputProvenance(**payload["provenance"]),**fields)
                started_call = time.perf_counter()
                report = judge.evaluate(request,seeds=payload["seeds"])
                report.validate_aggregation()
                result = {"schema_version":1,"request_id":payload["request_id"],"status":"completed","report":report.as_dict(),"adapter_receipt":adapter_receipt,"timing":{"worker_inference_seconds":time.perf_counter()-started_call,"worker_load_seconds":load_seconds if calls==0 else None,"first_call_on_replica":calls==0,"gpu_peak_memory_bytes":report.gpu_peak_memory_bytes,"gpu_type":torch.cuda.get_device_name()}}
                calls += 1
                emit({"kind":"result","response":result})
            except Exception as error:
                emit({"kind":"result","response":{"schema_version":1,"request_id":payload.get("request_id"),"status":"failed","reason":str(error)[:1000]}})

print(json.dumps({"ready":True,"load_seconds":load_seconds,"adapter_receipt":receipt()}),flush=True)
ThreadingHTTPServer(("0.0.0.0",args.port),Handler).serve_forever()
