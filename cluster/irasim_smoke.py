"""Cluster-only, experimental original-IRASim one-step smoke with durable evidence."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, sys, time, traceback
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return {"path":str(path),"bytes":Path(path).stat().st_size,"sha256":"sha256:"+h.hexdigest()}
def versions():
    out={}
    for n in ("torch","torchvision","diffusers","huggingface_hub","transformers","accelerate","timm","omegaconf"):
        try: out[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: out[n]=None
    return out
def write(path,payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
def main():
    p=argparse.ArgumentParser()
    for name in ("repo","checkpoint","vae","scheduler","config","image","action","action-provenance","output","report"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--seed",type=int,default=0); args=p.parse_args(); started=time.time()
    base={"schema_version":1,"kind":"irasim_original_one_step_smoke","qualification":"experimental_unqualified_not_gate_b",
          "paths":{"repo":args.repo,"checkpoint":args.checkpoint,"vae":args.vae,"scheduler":args.scheduler,"config":args.config},
          "runtime":{"packages":versions(),"python":sys.version.replace("\n"," ")},"seed":args.seed}
    try:
        from PIL import Image
        import numpy as np, imageio.v3 as iio
        from plumb.adapters.irasim_runtime import IRASimOneStepAdapter, IRASimOneStepProfile
        action=json.loads(Path(args.action).read_text())
        provenance=json.loads(Path(args.action_provenance).read_text())
        if not isinstance(provenance,dict) or not provenance.get("source_uri") or not provenance.get("sha256"):
            raise ValueError("--action-provenance must name a source_uri and SHA-256 for a real source action.")
        if len(action)!=7: raise ValueError("action JSON must be one source-derived native 7-D action.")
        image=np.asarray(Image.open(args.image).convert("RGB"))
        profile=IRASimOneStepProfile("irasim-original-one-step",args.repo,args.checkpoint,args.vae,args.scheduler,args.config)
        adapter=IRASimOneStepAdapter(profile); result=adapter.generate_one_step(image,action,args.seed)
        frames=np.stack([((f.detach().float().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype("uint8") for f in result.frames])
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); iio.imwrite(str(out),frames,fps=5,codec="libx264")
        base.update({"status":"completed","artifacts":{"image":digest(args.image),"action":digest(args.action),"action_provenance":provenance,"video":digest(out)},
                     "native_action_scaled":result.native_action_scaled,"condition_preprocessing":result.condition_preprocessing,"timing":result.timing.as_dict(),"frame_count":len(result.frames)})
    except Exception as e:
        base.update({"status":"failed","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
    base["finished_at_unix"]=time.time(); base["total_seconds"]=base["finished_at_unix"]-started
    write(args.report,base); print(json.dumps(base,indent=2)); return 0 if base["status"]=="completed" else 2
if __name__=="__main__": raise SystemExit(main())
