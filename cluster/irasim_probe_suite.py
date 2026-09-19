"""Offline, bounded original-IRASim one-step same-start diagnostic suite.

This is not Gate B evidence or a policy evaluation. It compares one real,
source-provenance OpenVLA 7-D action to repeat/stationary/directed controls at
the same source image and seed, using the unchanged IRASim 16/3/1 model.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, math, os, subprocess, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from plumb.adapters.irasim_runtime import IRASimOneStepAdapter, IRASimOneStepProfile, IRASIM_COMMIT

LABEL="experimental_unqualified_same_start_one_step_diagnostic"

@dataclass(frozen=True)
class Case:
    name:str; action:Tuple[float,...]; kind:str; description:str

def build_cases(source_action:Sequence[float], delta:float) -> Tuple[Case,...]:
    if len(source_action)!=7: raise ValueError("source action must be 7-D")
    a=tuple(float(v) for v in source_action)
    stationary=(0.,0.,0.,0.,0.,0.,a[6])
    cases=[Case("original",a,"source_action","Source-provenance OpenVLA action."),
           Case("repeat",a,"fixed_seed_repeat","Exact original-action repeat."),
           Case("stationary",stationary,"held_gripper_stationary","Translation/rotation zero; original gripper held. Stationary command is not image/noise invariance.")]
    for axis,name in enumerate(("x","y","z")):
        for sign,label in ((1,"plus"),(-1,"minus")):
            row=[0.,0.,0.,0.,0.,0.,a[6]]; row[axis]=sign*float(delta)
            cases.append(Case("directed_%s_%s"%(label,name),tuple(row),"directed_axis","%s %.6f translation with source gripper held."%(label,sign*float(delta))))
    return tuple(cases)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return "sha256:"+h.hexdigest()
def normalized_sha(value):
    if not isinstance(value, str): raise ValueError("SHA-256 must be a string")
    digest=value.removeprefix("sha256:").lower()
    if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest): raise ValueError("invalid SHA-256")
    return digest
def validate_inputs(action,provenance,image_path,checkpoint_path,checkpoint_sha,seeds,delta,repo,out):
    if out.exists(): raise FileExistsError("refusing existing output directory before model work: %s"%out)
    if not isinstance(action,list) or len(action)!=7 or any(not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in action): raise ValueError("source action must be finite 7-D")
    if not 0<=float(action[6])<=1: raise ValueError("source gripper must be within [0,1]")
    if not isinstance(provenance,dict) or not provenance.get("source_uri") or not provenance.get("sha256"): raise ValueError("action provenance requires source_uri + sha256")
    # The supplied CLI files, not untrusted provenance paths, are the authority.
    actual_action=sha(provenance["_action_path"]); actual_image=sha(image_path)
    if normalized_sha(provenance.get("action_file_sha256"))!=normalized_sha(actual_action) or normalized_sha(provenance.get("image_sha256"))!=normalized_sha(actual_image): raise ValueError("provenance action/image SHA-256 does not match actual CLI files")
    if not isinstance(checkpoint_sha,str) or len(checkpoint_sha)!=64 or sha(checkpoint_path)!=("sha256:"+checkpoint_sha.lower()): raise ValueError("checkpoint SHA-256 mismatch")
    if len(seeds)<1 or len(seeds)>3 or len(set(seeds))!=len(seeds): raise ValueError("require 1-3 unique seeds")
    if not math.isfinite(delta) or not 0<float(delta)<=.03: raise ValueError("axis delta must be finite in (0, .03]")
    head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if head!=IRASIM_COMMIT: raise ValueError("IRASim source revision mismatch")
def immutable(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:f.write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
def raw_frames(result):
    import numpy as np
    return [((f.detach().float().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).round().astype("uint8") for f in result.frames]
def raw_metrics(reference,candidate):
    import numpy as np
    if len(reference)!=len(candidate):return {"status":"incomparable","reason":"frame count mismatch"}
    labels={"condition":0,"future":1,"last":len(reference)-1}; values={}
    for label,i in labels.items(): values[label]=float(np.abs(reference[i].astype("float32")-candidate[i].astype("float32")).mean())
    def motion(frames):return float(np.mean([np.abs(frames[i].astype("float32")-frames[i-1].astype("float32")).mean() for i in range(1,len(frames))]))
    return {"status":"ok","space":"raw_uint8_before_png_mp4_codec","pixel_mae_0_255":values,"motion_mean_consecutive_mae_0_255":{"reference":motion(reference),"candidate":motion(candidate)}}
def write_media(frames,base):
    import imageio.v3 as iio, numpy as np
    base=Path(base); targets=(base.with_suffix(".mp4"),base.with_name(base.name+"-condition.png"),base.with_name(base.name+"-future.png"))
    if any(t.exists() for t in targets): raise FileExistsError("refusing to overwrite IRASim media artifact")
    iio.imwrite(str(targets[0]),np.stack(frames),fps=5,codec="libx264")
    iio.imwrite(str(base.with_name(base.name+"-condition.png")),frames[0],extension=".png")
    iio.imwrite(str(base.with_name(base.name+"-future.png")),frames[1],extension=".png")
    return {"mp4":{"path":str(targets[0]),"sha256":sha(targets[0])},"condition_png":{"path":str(targets[1]),"sha256":sha(targets[1])},"future_png":{"path":str(targets[2]),"sha256":sha(targets[2])}}
def run_cases(adapter,image,cases,seeds,out,provenance):
    raw_by_seed={}; reports=[]
    for seed in seeds:
        raw_by_seed[seed]={}
        for case in cases:
            base=Path(out)/("seed-%s-%s"%(seed,case.name)); report_path=base.with_suffix(".json")
            try:
                result=adapter.generate_one_step(image,case.action,seed); frames=raw_frames(result); media=write_media(frames,base); raw_by_seed[seed][case.name]=frames
                report={"schema_version":1,"kind":"irasim_one_step_action_probe","status":"completed","qualification":LABEL,"seed":seed,"case":case.__dict__,"provenance":provenance,"native_action_scaled":result.native_action_scaled,"condition_preprocessing":result.condition_preprocessing,"timing":result.timing.as_dict(),"artifacts":media,"note":"No Gate B claim. Stationary command does not imply image/noise invariance."}
            except Exception as e:
                report={"schema_version":1,"kind":"irasim_one_step_action_probe","status":"failed","qualification":LABEL,"seed":seed,"case":case.__dict__,"provenance":provenance,"error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}}
            immutable(report_path,report); reports.append(report)
    for report in reports:
        seed=report["seed"]; name=report["case"]["name"]
        if report["status"]=="completed" and name!="original" and "original" in raw_by_seed[seed]:
            report["raw_comparison_to_original"]=raw_metrics(raw_by_seed[seed]["original"],raw_by_seed[seed][name])
            # immutable report already exists; a sidecar preserves immutable first record.
            immutable(Path(out)/("seed-%s-%s-metrics.json"%(seed,name)),{"qualification":LABEL,"case":name,"seed":seed,"raw_comparison_to_original":report["raw_comparison_to_original"]})
    return reports
def main():
    p=argparse.ArgumentParser()
    for name in ("repo","checkpoint","vae","scheduler","config","image","action","action-provenance","output-dir") :p.add_argument("--"+name,required=True)
    p.add_argument("--seeds",default="0,1");p.add_argument("--axis-delta",type=float,default=.005);p.add_argument("--checkpoint-sha256",required=True);a=p.parse_args()
    from PIL import Image
    import numpy as np
    action=json.loads(Path(a.action).read_text()); provenance=json.loads(Path(a.action_provenance).read_text())
    image=np.asarray(Image.open(a.image).convert("RGB")); seeds=tuple(int(x) for x in a.seeds.split(",") if x.strip()); out=Path(a.output_dir)
    provenance["_action_path"]=a.action
    validate_inputs(action,provenance,Path(a.image),Path(a.checkpoint),a.checkpoint_sha256,seeds,a.axis_delta,Path(a.repo),out)
    provenance.pop("_action_path");out.mkdir(parents=True,exist_ok=False)
    profile=IRASimOneStepProfile("irasim-one-step-probe",a.repo,a.checkpoint,a.vae,a.scheduler,a.config);adapter=IRASimOneStepAdapter(profile)
    provenance.update({"source_action_file":str(a.action),"source_action_sha256":sha(a.action),"image_file":str(a.image),"image_sha256":sha(a.image),
                       "checkpoint_sha256":sha(a.checkpoint),"config_sha256":sha(a.config),"source_revision":IRASIM_COMMIT,
                       "source_release":os.environ.get("PLUMB_SOURCE_RELEASE"),"inference_steps":50,"guidance_scale":1.0})
    reports=run_cases(adapter,image,build_cases(action,a.axis_delta),seeds,out,provenance)
    immutable(out/"suite.json",{"schema_version":1,"kind":"irasim_one_step_probe_suite","qualification":LABEL,"cases":[{"seed":r["seed"],"case":r["case"]["name"],"status":r["status"]} for r in reports],"note":"Same-start/same-seed single-step diagnostic only; no Gate B qualification."})
    return 0 if all(r["status"]=="completed" for r in reports) else 2
if __name__=="__main__":raise SystemExit(main())
