"""Read-only VERGE evidence collection. Never load models or open sealed tests."""
import json, hashlib, sys, platform
from pathlib import Path
from datetime import datetime, timezone
EXP = Path("/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness")
OUT = EXP / "outputs/verge_review_20260908"
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
result={"collected_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"root":str(EXP)}
result["instructions"]=[]
for parent in [Path("/"),Path("/blue"),Path("/blue/du.j"),EXP.parents[1],EXP.parent,EXP]:
    for name in ["AGENTS.md","AGENTS.override.md"]:
        p=parent/name
        result["instructions"].append({"path":str(p),"exists":p.is_file(),"text":p.read_text() if p.is_file() else None})
result["runs"]=[]
for p in sorted((EXP/"raw_results").glob("verge_book_v2_*/archive.json")):
    a=read(p); branches=a["branch_endpoints"]; d=a["decision"]; cfg=a.get("config",{})
    row={"path":str(p.relative_to(EXP)),"sha256":sha(p),"keys":list(a),
         "decision":d,"challenger_update":a["challenger_update"],
         "branches":{k:{"checkpoint":v["checkpoint"],"counts":v["target"]["counts"],"rates":v["target"]["rates"],
           "rollouts":v["target"]["rollouts"],"training":v["training"]} for k,v in branches.items()}}
    # Independently recount saved endpoint vectors; these are old-condition outputs.
    for k,v in branches.items():
        vectors=v["target"]["keyed_vectors"]
        row["branches"][k]["recomputed_counts"]=[sum(vec[j] for vec in vectors.values()) for j in range(11)]
        row["branches"][k]["counts_match"]=row["branches"][k]["recomputed_counts"]==v["target"]["counts"]
    result["runs"].append(row)
result["completion_markers"]={}
for rel in ["raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json",
            "raw_results/verge_followon_v1/FOLLOWON_BLOCK_COMPLETE.json",
            "raw_results/verge_control_v1/CONTROL_BLOCK_COMPLETE.json"]:
    p=EXP/rel
    result["completion_markers"][rel]=read(p) if p.exists() else {"missing":True}
result["checkpoint_metadata"]={}
for role in ["solver","challenger"]:
    p=EXP/f"checkpoints/verge_book_v2_initial_{role}/resume_u0000"
    result["checkpoint_metadata"][role]={n:read(p/n) if (p/n).is_file() else None
        for n in ["adapter_config.json","verge_committed.json","trainer_state.json"]}
    result["checkpoint_metadata"][role]["files"]=[{"name":f.name,"bytes":f.stat().st_size} for f in p.iterdir() if f.is_file()]
model=EXP.parents[1]/"models/Ministral-3-3B-Instruct-2512-BF16"
result["model_metadata"]={}
for n in ["config.json","tokenizer_config.json"]:
    p=model/n
    data=read(p)
    result["model_metadata"][n]={"sha256":sha(p),"fields":{k:data.get(k) for k in ["_name_or_path","_commit_hash","model_type","architectures","transformers_version","tokenizer_class","chat_template","bos_token","eos_token","pad_token"]}}
from datasets import load_from_disk
splits={}
result["splits"]={}
for name in ["target_train","target_selection","target_completion"]:
    p=EXP/"data/verge_mistral_round1"/name
    rows=list(load_from_disk(str(p)))
    specs={json.dumps(json.loads(r["generator_config"]).get("official_parameter"),sort_keys=True) for r in rows}
    tuples={json.dumps(r["ground_truth"],sort_keys=True) for r in rows}
    inputs={str(c.get("input","")) for r in rows for c in r["ground_truth"]}
    splits[name]={"specs":specs,"tests":tuples,"inputs":inputs,"ids":{r["id"] for r in rows}}
    result["splits"][name]={"rows":len(rows),"distinct_specs":list(specs),
        "unique_test_suites":len(tuples),"unique_input_tapes":len(inputs),
        "case_counts":sorted({len(r["ground_truth"]) for r in rows}),
        "case_keys":sorted({k for r in rows for c in r["ground_truth"] for k in c}),
        "sample_generator_config":rows[0]["generator_config"],
        "content_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        "rows_with_hint_text":sum("hint" in "\n".join(m["content"] for m in r["messages"]).lower() for r in rows)}
result["split_overlap"]={}
for x,y in [("target_train","target_selection"),("target_train","target_completion"),("target_selection","target_completion")]:
    result["split_overlap"][x+" vs "+y]={k:len(splits[x][k]&splits[y][k]) for k in splits[x]}
result["sealed_test_read"]=False
OUT.mkdir(exist_ok=True)
(OUT/"field_evidence.json").write_text(json.dumps(result,indent=2))
print(json.dumps({"runs":len(result["runs"]),"endpoint_counts_match":all(b["counts_match"] for r in result["runs"] for b in r["branches"].values()),
 "splits":result["splits"],"split_overlap":result["split_overlap"],"saved":str(OUT/"field_evidence.json")},indent=2))

