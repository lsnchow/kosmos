"""Safely convert reviewed IRASim EMA weights to safetensors on a CPU allocation.

This tool intentionally refuses an unsafe pickle fallback. It admits only the
globals observed in the pinned checkpoint preflight and aliases OmegaConf
metadata classes to an inert recorder, so their upstream methods never run.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, uuid
from pathlib import Path

APPROVED_GLOBALS=frozenset({
 "collections.defaultdict","omegaconf.listconfig.ListConfig","builtins.int","builtins.list",
 "omegaconf.nodes.AnyNode","typing.Any","builtins.dict","omegaconf.base.ContainerMetadata",
 "omegaconf.dictconfig.DictConfig","omegaconf.base.Metadata",
})
OMEGACONF_ALIASES=("omegaconf.listconfig.ListConfig","omegaconf.nodes.AnyNode","omegaconf.base.ContainerMetadata",
                  "omegaconf.dictconfig.DictConfig","omegaconf.base.Metadata","typing.Any")
class InertMetadata:
    """Inert pickle target: retain state without calling OmegaConf code."""
    def __init__(self,*args,**kwargs): self.args=args; self.kwargs=kwargs
    def __setstate__(self,state): self.state=state
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def validate_unsafe_globals(found):
    found=set(found); unknown=found-APPROVED_GLOBALS
    if unknown: raise RuntimeError("Rejecting checkpoint with unreviewed pickle globals: "+", ".join(sorted(unknown)))
    return sorted(found)
def safe_aliases():
    import collections
    aliases=[(InertMetadata,name) for name in OMEGACONF_ALIASES]
    aliases += [(collections.defaultdict,"collections.defaultdict"),(int,"builtins.int"),(list,"builtins.list"),(dict,"builtins.dict")]
    return aliases
def build_model(repo,config):
    from omegaconf import OmegaConf
    sys.path.insert(0,str(Path(repo)))
    from models import get_models
    args=OmegaConf.merge(OmegaConf.load(Path(repo)/"configs/base/data.yaml"),OmegaConf.load(Path(repo)/"configs/base/diffusion.yaml"),OmegaConf.load(config))
    if (int(args.num_frames),int(args.extras),int(args.mask_frame_num)) != (16,3,1): raise RuntimeError("Expected original Bridge frame_ada 16/3/1 config")
    args.latent_size=[int(v)//8 for v in args.video_size]
    return get_models(args)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True)
    p.add_argument("--repo",required=True); p.add_argument("--config",required=True); p.add_argument("--source-sha256",required=True); p.add_argument("--report",required=True); a=p.parse_args()
    import torch
    source=sha(a.checkpoint)
    report={"schema_version":1,"kind":"irasim_safe_ema_conversion","source":str(a.checkpoint),"source_sha256":source,"expected_source_sha256":a.source_sha256,"status":"failed"}
    try:
        if source != a.source_sha256.lower(): raise RuntimeError("source checksum mismatch")
        observed=validate_unsafe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(a.checkpoint)); report["approved_unsafe_globals"]=observed
        with torch.serialization.safe_globals(safe_aliases()): checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
        ema=checkpoint.get("ema") if isinstance(checkpoint,dict) else None
        if not isinstance(ema,dict) or not ema or any(not isinstance(k,str) or not torch.is_tensor(v) for k,v in ema.items()): raise RuntimeError("EMA is not a nonempty str->Tensor state dict")
        model=build_model(a.repo,a.config); model.load_state_dict(ema,strict=True)
        from safetensors.torch import save_file
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists(): raise FileExistsError("Refusing to overwrite an existing converted checkpoint")
        temporary=out.with_name(out.name+".partial-"+uuid.uuid4().hex)
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(temporary),metadata={"source_sha256":source,"format":"IRASim Bridge EMA strict verified"})
        temporary.replace(out)
        report.update({"status":"completed","output":str(out),"output_sha256":sha(out),"tensor_count":len(ema)})
    except Exception as e: report["error"]={"type":type(e).__name__,"message":str(e)}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["status"]=="completed" else 2
if __name__=="__main__": raise SystemExit(main())
