"""Account for three-element control arrays without counting parents/batch twice."""
import argparse
import json
import subprocess
import time
from verge_control_controller import EXP, ROOT, LABELS, read, write
from verge_book_resources import FIELDS, parse_sacct, summarize


def registry(root=ROOT):
    registered, parents = {}, set()
    for r in range(6):
        path = root / "jobs" / f"round_{r:02d}.json"
        if not path.exists():
            continue
        record = read(path)
        if record.get("round") != r or record.get("labels") != list(LABELS):
            raise RuntimeError("Unrecognized control wave registry")
        for stage in ("workers","coordinator"):
            entries = record.get("historical_"+stage,[]) + ([record[stage]] if stage in record else [])
            for parent in entries:
                if not isinstance(parent,str) or not parent.isdigit() or parent in parents:
                    raise RuntimeError("Invalid or duplicated control parent job")
                parents.add(parent)
                if stage == "workers":
                    for i,label in enumerate(LABELS):
                        registered[f"{parent}_{i}"] = {"round":r,"arm":label,"stage":"worker",
                                                       "parent_job_id":parent}
                else:
                    registered[parent] = {"round":r,"arm":"all","stage":"coordinator",
                                          "parent_job_id":parent}
    return registered, sorted(parents,key=int)


def collect():
    registered, parents = registry()
    if not parents:
        return {"allocations":[],"registered_parent_jobs":0,"gpu_allocation_hours":0.,
                "cpu_allocation_core_hours":0.,"accounting_complete":False,
                "note":"No real direct-control jobs have been submitted","suite_complete":False}
    response = subprocess.run(["sacct","--noheader","--parsable2","--units=K","-j",",".join(parents),
        "--format="+",".join(FIELDS)],cwd=EXP,capture_output=True,text=True,check=True)
    result = summarize(parse_sacct(response.stdout),registered)
    result.update(registered_parent_jobs=len(parents),observed_at_unix=time.time(),
                  scope="Only explicitly registered direct-control jobs, including recorded recovery predecessors",
                  primary_and_v1_costs_kept_separately=True,suite_complete=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = collect()
    if args.save:
        write(ROOT / "resource_accounting_latest.json",result)
    print(json.dumps(result,indent=2))
