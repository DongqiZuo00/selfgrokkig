import json, sys, hashlib
from pathlib import Path
import torch
exp=Path("/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness")
out=exp/"outputs/verge_review_20260908"
result={"torch":torch.__version__,"cuda_visible":__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),"roots":{}}
for role in ["solver","challenger"]:
 p=exp/f"checkpoints/verge_book_v2_initial_{role}/resume_u0000/state.pt"
 state=torch.load(p,map_location="cpu",weights_only=True)
 result["roots"][role]={"state_path":str(p),"keys":list(state),
    "runtime_state":state.get("runtime_state"),"optimizer_entries":len(state.get("optimizer",{}).get("state",{})),
    "optimizer_groups":[{k:v for k,v in g.items() if k!="params"} for g in state.get("optimizer",{}).get("param_groups",[])]}
result["source_sha256"]={n:hashlib.sha256((exp/"src"/n).read_bytes()).hexdigest() for n in ["common.py","verge_round_core.py","verge_repair_training.py"]}
(out/"checkpoint_state_evidence.json").write_text(json.dumps(result,indent=2,default=str))
print(json.dumps(result,indent=2,default=str))

