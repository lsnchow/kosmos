from cluster import judge_format_pilot as pilot
from cluster import judge_teacher_v2 as v2
import json

def test_first_valid_uses_lowest_schema_valid_actual_raw_sample_only():
 def item(index,raw,parsed): return {'sample_index':index,'attempts':[{'raw_output':raw,'parsed':parsed}]}
 valid=json.dumps({'integrity':'artifact','collision':'visible','progress':5,'completion_evidence':'met','evidence_frame_indices':[0],'observable_reasons':'x'})
 report={'raw_judge_samples':[item(0,None,None),item(1,valid,json.loads(valid)),item(2,valid,json.loads(valid)),item(3,None,None),item(4,None,None)]}
 selected=pilot._first_valid(report)
 assert selected['sample_index']==1 and selected['raw_teacher_output']==valid

def test_format_config_never_declares_semantic_supervision():
 assert pilot.CONFIG['semantic_supervision']=='forbidden'
 assert pilot.CONFIG['lora_r']==16 and pilot.CONFIG['lora_alpha']==32
