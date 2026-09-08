"""CPU evidence, not a new training implementation. No model loading/sampling."""
import copy, importlib.util, json, sys
from pathlib import Path
import torch
import common
from verge_round_core import condition_vector, thresholds, centered_advantages
from verge_repair_protocol import parse_typed_proposal
from test_verge_fresh_review import Tiny, equal_tree
out=Path(__file__).resolve().parents[2]
evidence={}
loop="""START start:
    NEXT pull
PULLER_RB pull:
    [R] loop
    [B] end
    [EMPTY] end
PAINTER_RED loop:
    NEXT loop
END end"""
cases=[{"input":x,"expected_output":"","expected_accepted":True,"check_output":True} for x in ["R","B",""]]
factory=common.CREATE_ROBOT_FACTORY(loop)
executions=[factory.process_robot(c["input"]) for c in cases]
passes=[int(e.finished and e.final_tape==c["expected_output"]) for e,c in zip(executions,cases)]
assert passes==[0,1,1] and "Maximum iterations" in executions[0].rejection_reason
observed=condition_vector(loop,{"ground_truth":cases})
f=sum(passes)/3 # Explicit single-class FIXTURE only; never a choice of real target classes.
desired=[1]+[int(f>=t) for t in sorted([1/3,.25,.5,.75])]+[int(f==1)]
assert observed==[1]+[0]*10 and desired==[1,1,1,1,0,0]
evidence["timeout_fixture"]={"per_test_pass":passes,"fixed_denominator":3,"f_single_explicit_fixture_class":f,
    "current_11_conditions":observed,"design_6_conditions":desired,
    "status":"CURRENT_LADDER_CONFLICT","note":"Production class assignment remains undefined."}
end="START start:\n    NEXT end\nEND end"
def rejected(call):
    try:
        call()
    except ValueError:
        return True
    return False
evidence["invalid_input_contracts"]={
    "empty_tests_rejected":rejected(lambda:common.verify_full(end,[])),
    "zero_test_thresholds_rejected":rejected(lambda:thresholds(0)),
    "missing_output_field_rejected":rejected(lambda:common.verify_full(end,[{"input":""}])),
    "status":"FIXED_IN_REVIEW_COPY_ONLY"}
assert all(v for k,v in evidence["invalid_input_contracts"].items() if k!="status")
stage={"stages":[{"family":"finite_map","colors":2,"min_input_length":1,"max_input_length":1,"outputs":["",""]}]}
evidence["constant_stage"]={"schema_accepted":parse_typed_proposal(json.dumps(stage),[])==stage,
    "status":"NONCONST_VALIDATOR_ABSENT_IN_THIS_PATH"}
evidence["optimizer"]={}
for name,base in [("baseline",out/"baseline/src"),("review_copy",out/"review_copy/src")]:
    spec=importlib.util.spec_from_file_location("training_"+name,base/"verge_repair_training.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.9,.95),weight_decay=0.)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:1/(i+1))
    items=[{"prompt_token_ids":[0],"completion_token_ids":[1,2]} for _ in range(8)]
    items[0]["completion_token_ids"]=[3,3];masks=[[1,1] for _ in items]
    module.train_step(model,None,opt,sched,items,centered_advantages([1]+[0]*7),masks,{"kl_beta":0.},0)
    weights=copy.deepcopy(model.state_dict());os=copy.deepcopy(opt.state_dict());ss=copy.deepcopy(sched.state_dict())
    metric=module.train_step(model,None,opt,sched,items,[0.]*8,masks,{"kl_beta":0.},1)
    evidence["optimizer"][name]={"optimizer_step":metric["optimizer_step"],
        "max_parameter_change":float((model.weight.detach()-weights["weight"]).abs().max()),
        "optimizer_state_identical":equal_tree(os,opt.state_dict()),
        "scheduler_state_identical":equal_tree(ss,sched.state_dict())}
assert evidence["optimizer"]["baseline"]["max_parameter_change"]>0
assert evidence["optimizer"]["review_copy"]["max_parameter_change"]==0
# The following are mathematical counterexamples, not measurements of VERGE.
rhos_a=[.99,.98,.97,.95,.88,.87];rhos_b=[.99,.98,.97,.95,.89,.01]
evidence["lead_counterexample"]={"A":rhos_a,"B":rhos_b,"same_b":5,"same_s":0,
    "lead_by_design":"B","higher_full_pass":"A"}
evidence["hint_counterexample"]={"tasks":32,"hint_solved_tasks":8,"samples_per_task":8,
    "hint_rate":.25,"no_hint_rate":0.,"nonconstant_reward_groups":0,
    "explanation":"Each hinted task has either eight successes or eight failures; validator rates pass but every update group is constant."}
evidence["nondegeneracy_counterexample"]={"class_A_pass_rates":[1.,.5],"class_B_pass_rates":[0.,0.],
    "f_for_two_distinct_behaviors":[0.,0.],"full_pass":[0,0]}
evidence["verification_replay"]=[]
from datasets import load_from_disk
exp=Path("/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness")
rows={str(r["id"]):r for r in load_from_disk(str(exp/"data/verge_mistral_round1/target_selection"))}
for branch in (0,1):
    p=exp/f"raw_results/verge_book_v2_verge_r02/branches/{branch}/target.jsonl"
    records=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    checked=records[:32]
    mismatches=[]
    for r in checked:
        got=common.verify_full(r["completion"],rows[str(r["instance_id"])]["ground_truth"]).reward
        if got!=r["reward"]:mismatches.append({"instance":r["instance_id"],"draw":r["rollout_index"]})
    evidence["verification_replay"].append({"source":str(p),"total_records":len(records),
        "checked":len(checked),"sample_rule":"first 32 in stored order, preselected before reading reward","mismatches":mismatches,
        "new_model_samples":0})
(out/"contract_diagnostics.json").write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2))
