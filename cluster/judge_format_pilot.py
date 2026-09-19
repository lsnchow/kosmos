#!/usr/bin/env python3
"""Actual format-only LoRA diagnostic; never supervises judgement values."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time, traceback
from pathlib import Path
from typing import Any, Mapping, Sequence
from cluster import judge_teacher_pilot as v1
from cluster import judge_teacher_v2 as v2
from plumb.policies.judge import parse_rubric_json

PURPOSE="judge_json_structure_only_pilot"; SEED=20260919
CONFIG={"purpose":PURPOSE,"base_model_revision":v1.QWEN_REVISION,"lora_r":16,"lora_alpha":32,"lora_targets":["q_proj","v_proj"],"learning_rate":5e-5,"epochs":2,"batch_size":1,"seed":SEED,"semantic_supervision":"forbidden"}

def _sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def _canon(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _first_valid(report:Mapping[str,Any])->Mapping[str,Any]:
 samples=report.get('raw_judge_samples')
 if not isinstance(samples,list) or len(samples)!=5: raise ValueError('teacher report lacks five samples')
 for sample in sorted(samples,key=lambda x:x.get('sample_index',99)):
  attempts=sample.get('attempts') if isinstance(sample,Mapping) else None
  if not isinstance(attempts,list) or not attempts: continue
  final=attempts[-1]; raw=final.get('raw_output') if isinstance(final,Mapping) else None
  if not isinstance(raw,str): continue
  try: label=parse_rubric_json(raw).as_dict()
  except Exception: continue
  if not isinstance(final.get('parsed'),Mapping) or dict(final['parsed'])!=dict(label): continue
  return {"sample_index":sample['sample_index'],"raw_teacher_output":raw,"label":label}
 raise ValueError('no schema-valid final actual teacher sample')

def _report_for(teacher:Path,clip:Mapping[str,Any])->Mapping[str,Any]:
 paths=sorted((teacher/'rawteacher-v2'/clip['clip_id']).glob('**/attempt-*.json'),reverse=True)
 for path in paths:
  try: p=json.loads(path.read_text())
  except Exception: continue
  if p.get('status')=='completed_unqualified' and p.get('input_binding_sha256')==clip['input_binding_sha256'] and p.get('profile_config',{}).get('profile_id')==v2.PROFILE_ID:
   return {"path":path,"payload":p}
 raise ValueError('missing hash-matched completed v2 teacher report for '+clip['clip_id'])

def build_rows(dataset:Path,teacher:Path)->dict:
 from cluster.export_judge_v2_pilot import _load_v2_freeze, _collect_reports
 candidate_path,_=v1._regular_relative(dataset,'candidate-inputs.json','candidate manifest')
 candidate=v1.validate_candidate_manifest(dataset,v1._json(candidate_path,'candidate manifest'))
 frozen=_load_v2_freeze(teacher,candidate_path)
 reports=_collect_reports(teacher,{c['clip_id']:c for c in candidate['clips']},frozen)
 rows={"train":[],"development_validation":[]}; refs={}
 for clip in candidate['clips']:
  report_path,report_payload,_=reports[clip['clip_id']]
  record={'path':report_path,'payload':report_payload}; actual=_first_valid(report_payload['raw_judge_report'])
  split=clip['cohort']; row={"purpose":PURPOSE,"qualified":False,"label_source":"teacher_values_untrusted_masked","input_profile":"qwen_rubric_serving_messages_v1","clip_id":clip['clip_id'],"source_lineage_id":clip['source_lineage_id'],"cohort":"format_pilot_"+split,"frames":[x['path'] for x in clip['frames']],"frame_sha256":[x['sha256'] for x in clip['frames']],"frame_timestamps":[x['timestamp'] for x in clip['frames']],"timestamp_semantics":clip['timestamp_semantics'],"reference_images":[{"path":clip['reference']['path'],"sha256":clip['reference']['sha256'],"role":"scene_context_not_goal","provenance_uri":clip['reference']['source_uri']}],"instruction":v1._canonical_task().instruction,"rubric":v2.v2_rubric(),"raw_teacher_output":actual['raw_teacher_output'],"label":actual['label'],"raw_teacher_ref":{"uri":record['path'].resolve().relative_to(teacher.resolve()).as_posix(),"sha256":_sha(record['path']),"sample_index":actual['sample_index']}}
  row['rubric']=frozen['profile_config']['rubric']
  rows[split].append(row); refs[clip['clip_id']]=row['raw_teacher_ref']
 return {"rows":rows,"candidate_sha256":_sha(candidate_path),"teacher_refs":refs,'teacher_freeze_sha256':_sha(teacher/'freeze.json'),'teacher_profile_config_sha256':frozen['profile_config_sha256']}

def _loss(model,inputs,torch):
 audit=inputs.pop('format_structure_audit',None)
 if not isinstance(audit,Mapping) or audit.get('semantic_supervised_tokens',-1)!=0 or audit.get('supervised_syntax_tokens',0)<1: raise RuntimeError('structure-only audit unavailable or semantic labels exposed')
 output=model(**{k:v.to('cuda') if hasattr(v,'to') else v for k,v in inputs.items()}); loss=output.loss
 if loss is None or not bool(torch.isfinite(loss).item()): raise RuntimeError('nonfinite format loss')
 return loss,audit

def _write_new(path:Path,payload:Mapping[str,Any])->str:
 encoded=(json.dumps(payload,sort_keys=True,indent=2)+'\n').encode();
 with path.open('xb') as f: f.write(encoded); f.flush(); os.fsync(f.fileno())
 return _sha(path)

def load_input_manifest(path:Path)->dict:
 """Load the frozen format input and resolve/re-hash its source media rows."""
 payload=json.loads(path.read_text())
 if payload.get('schema')!='plumb_judge_format_pilot_input_v1': raise ValueError('unexpected format input manifest schema')
 root=Path(payload.get('source',{}).get('dataset_root','')).resolve(strict=True)
 result={'train':[],'development_validation':[]}
 for split in result:
  for row in payload.get('splits',{}).get(split,[]):
   if len(row.get('frames',[]))!=16 or len(row.get('frame_sha256',[]))!=16: raise ValueError('format input requires 16 frames and 16 hashes')
   frames=[]
   for rel,digest in zip(row['frames'],row['frame_sha256']):
    lexical=root/rel
    if Path(rel).is_absolute() or '..' in Path(rel).parts or lexical.is_symlink(): raise ValueError('format input frame path is unsafe')
    candidate=lexical.resolve(strict=True); candidate.relative_to(root)
    if _sha(candidate)!=digest: raise ValueError('format input frame hash mismatch')
    frames.append(str(candidate))
   refs=[]
   for ref in row['reference_images']:
    lexical=root/ref['path']
    if Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts or lexical.is_symlink(): raise ValueError('format input reference path is unsafe')
    candidate=lexical.resolve(strict=True); candidate.relative_to(root)
    if _sha(candidate)!=ref['sha256']: raise ValueError('format input reference hash mismatch')
    refs.append({**ref,'path':str(candidate)})
   result[split].append({**row,'frames':frames,'reference_images':refs})
 return result

def run(args)->dict:
 if not os.environ.get('SLURM_JOB_ID') or os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1': raise RuntimeError('allocated offline GPU required')
 dataset=Path(args.dataset_root).resolve(strict=True); teacher=Path(args.teacher_root).resolve(strict=True); output=Path(args.output_dir).resolve()
 if output.exists() or output==dataset or output==teacher: raise RuntimeError('output-dir must be fresh and separate')
 built=build_rows(dataset,teacher)
 from deploy.baseten.training import train_judge_lora as train
 output=train._claim_fresh_output_dir(str(output),dataset_dir=str(dataset),model_root=str(Path(args.model_root).resolve()))
 if train._paths_overlap(output,teacher): raise RuntimeError('format output overlaps teacher evidence')
 config_sha=_write_new(output/'config.json',CONFIG)
 input_manifest={"schema":"plumb_judge_format_pilot_input_v1","purpose":PURPOSE,"qualified":False,"candidate_manifest_sha256":built['candidate_sha256'],"source":{"dataset_root":str(dataset),"teacher_root":str(teacher)},"splits":{}}
 input_manifest['teacher_freeze_sha256']=built['teacher_freeze_sha256']
 input_manifest['teacher_profile_config_sha256']=built['teacher_profile_config_sha256']
 for split,rows in built['rows'].items():
  if len(rows)!=(12 if split=='train' else 4): raise RuntimeError('format input split count changed')
  input_manifest['splits'][split]=[{"purpose":r['purpose'],"qualified":False,"label_source":r['label_source'],"clip_id":r['clip_id'],"source_lineage_id":r['source_lineage_id'],"cohort":r['cohort'],"frames":r['frames'],"frame_sha256":r['frame_sha256'],"frame_timestamps":r['frame_timestamps'],"timestamp_semantics":r['timestamp_semantics'],"reference_images":r['reference_images'],"instruction":r['instruction'],"rubric":r['rubric'],"raw_teacher_ref":r['raw_teacher_ref'],"raw_teacher_output":r['raw_teacher_output'],"label":r['label'],"raw_teacher_output_sha256":hashlib.sha256(r['raw_teacher_output'].encode()).hexdigest()} for r in rows]
 input_manifest_sha=_write_new(output/'input_manifest.json',input_manifest)
 resolved=load_input_manifest(output/'input_manifest.json')
 if len({r['source_lineage_id'] for s in resolved.values() for r in s})!=16: raise RuntimeError('format input lineages are not unique')
 try: from cluster.judge_format_training import build_structure_only_collator
 except ImportError as e: raise RuntimeError('structure-only collator module unavailable') from e
 modules=train._load_frameworks(); torch=modules['torch']; transformers=modules['transformers']; peft=modules['peft']
 if not torch.cuda.is_available(): raise RuntimeError('allocated CUDA required')
 visible=[item.strip() for item in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if item.strip()]
 if len(visible)!=1: raise RuntimeError('format pilot requires exactly one allocated CUDA visible device')
 torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 student={"model_id":v1.QWEN_MODEL_ID,"model_revision":v1.QWEN_REVISION}; binding=train._local_model_binding(args.model_root,args.model_manifest,student)
 processor=transformers.AutoProcessor.from_pretrained(binding['model_root'],revision=v1.QWEN_REVISION,local_files_only=True,trust_remote_code=False)
 base=transformers.AutoModelForVision2Seq.from_pretrained(binding['model_root'],revision=v1.QWEN_REVISION,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16)
 model=peft.get_peft_model(base,peft.LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none')).to('cuda')
 collator=build_structure_only_collator(processor)
 def inputs_for(row):
  inputs=collator([row]); audit=getattr(collator,'last_audit',None); inputs['format_structure_audit']=audit; return inputs
 def evaluate(rows):
  model.eval(); values=[]; audits=[]
  with torch.no_grad():
   for row in rows:
    inputs=inputs_for(row); loss,checked=_loss(model,inputs,torch); values.append(float(loss.detach().cpu())); audits.append(checked)
  return sum(values)/len(values),audits
 before,dev_audit=evaluate(resolved['development_validation']); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5); steps=0; train_audit=[]; epochs=[]; started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,3):
  model.train()
  losses=[]
  for row in resolved['train']:
   inputs=inputs_for(row); loss,checked=_loss(model,inputs,torch); loss.backward()
   finite=False
   for parameter in model.parameters():
    if parameter.requires_grad and parameter.grad is not None:
     if not bool(torch.isfinite(parameter.grad).all().item()): raise RuntimeError('nonfinite LoRA gradient')
     finite = finite or float(torch.linalg.vector_norm(parameter.grad.float()).item())>0.0
   if not finite: raise RuntimeError('no finite nonzero LoRA gradient before optimizer step')
   opt.step(); opt.zero_grad(set_to_none=True); steps+=1; train_audit.append(checked); losses.append(float(loss.detach().cpu()))
  epoch_dev,epoch_audit=evaluate(resolved['development_validation']); epochs.append({'epoch':epoch,'train_syntax_loss':sum(losses)/len(losses),'development_validation_syntax_loss':epoch_dev,'development_validation_audit':epoch_audit})
 after=epochs[-1]['development_validation_syntax_loss']; adapter=output/'adapter'; model.save_pretrained(str(adapter)); processor.save_pretrained(str(adapter))
 gpu=subprocess.run(['nvidia-smi','--id='+visible[0],'--query-gpu=uuid,name','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
 return {"kind":"plumb_judge_json_structure_only_pilot","purpose":PURPOSE,"qualified":False,"status":"completed_unqualified_structure_only_pilot","config":CONFIG,"config_sha256":config_sha,"candidate_manifest_sha256":built['candidate_sha256'],"teacher_refs":built['teacher_refs'],"input_manifest":{"path":"input_manifest.json","sha256":input_manifest_sha},"dataset_files_sha256":{'input_manifest':input_manifest_sha,'config':config_sha},"source":{"dataset_root":str(dataset),'teacher_root':str(teacher),'candidate_manifest':str(dataset/'candidate-inputs.json')},"model_binding":binding,"selected_checkpoint":{"adapter_path":"adapter","adapter_tree_sha256":train._digest_directory(adapter),"selected_epoch":2},"train_rows":len(resolved['train']),"development_validation_rows":len(resolved['development_validation']),"dev_syntax_loss_before":before,"dev_syntax_loss_after":after,"optimizer_steps":steps,"epoch_reports":epochs,"structure_audit":{"train":train_audit,"dev_before":dev_audit},"gpu_identity":gpu,"gpu_peak_memory_bytes":int(torch.cuda.max_memory_allocated()),"wall_seconds":time.monotonic()-started,"timing_scope":"two format-only LoRA epochs plus paired syntax-only development losses; excludes source/model binding and frozen input creation","source_release":os.environ.get('PLUMB_SOURCE_RELEASE'),"training_started":True,"filled_gates":[]}

def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser(description=__doc__); [p.add_argument(x,required=True) for x in ('--dataset-root','--teacher-root','--model-root','--model-manifest','--output-dir')]; args=p.parse_args(argv)
 try:
  result=run(args); out=Path(args.output_dir)/'format-pilot-result.json'; _write_new(out,result); print(json.dumps(result,sort_keys=True)); return 0
 except Exception as e: print(json.dumps({'qualified':False,'status':'failed','error':{'type':type(e).__name__,'message':str(e)}})); return 2
if __name__=='__main__': raise SystemExit(main())
