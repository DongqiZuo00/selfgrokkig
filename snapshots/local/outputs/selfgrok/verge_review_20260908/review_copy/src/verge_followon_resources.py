"""Allocation accounting for all registered follow-on elements and recoveries."""
import argparse
import json
import subprocess
import time
from verge_followon_controller import EXP,ROOT,version,read,write
from verge_book_resources import FIELDS,parse_sacct,summarize


def registry(root=ROOT):
    path=root/'jobs/block.json'
    if not path.exists():return {},[]
    record=read(path)
    if record.get('candidate_count')!=72 or record.get('indices')!=list(range(72)):
        raise RuntimeError('Unrecognized all-candidate registry')
    registered={};parents=set()
    workers=record.get('historical_workers',[])+([{'job_id':record['workers'],
        'indices':record.get('worker_indices',record['indices'])}] if 'workers' in record else [])
    coordinators=record.get('historical_coordinators',[])+([record['coordinator']] if 'coordinator' in record else [])
    for entry in workers:
        parent,indices=entry['job_id'],entry['indices']
        if (not isinstance(parent,str) or not parent.isdigit() or parent in parents or not indices
                or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<72 for i in indices)):
            raise RuntimeError('Invalid or repeated follow-on worker registration')
        parents.add(parent)
        for i in indices:registered[f'{parent}_{i}']={'version':version(i),'candidate_array_index':i,
                                                    'stage':'worker','parent_job_id':parent}
    for parent in coordinators:
        if not isinstance(parent,str) or not parent.isdigit() or parent in parents:
            raise RuntimeError('Invalid or repeated follow-on coordinator registration')
        parents.add(parent);registered[parent]={'stage':'coordinator','candidate_array_index':None,'parent_job_id':parent}
    return registered,sorted(parents,key=int)


def collect():
    registered,parents=registry()
    if not parents:return {'allocations':[],'registered_parent_jobs':0,'accounting_complete':False,
        'note':'No real follow-on jobs have been submitted','suite_complete':False}
    response=subprocess.run(['sacct','--noheader','--parsable2','--units=K','-j',','.join(parents),
        '--format='+','.join(FIELDS)],cwd=EXP,capture_output=True,text=True,check=True)
    result=summarize(parse_sacct(response.stdout),registered)
    result.update(registered_parent_jobs=len(parents),observed_at_unix=time.time(),
        scope='Only registered follow-on allocations, including historical recovery elements; parent/batch not counted twice',
        primary_v1_v2_and_control_costs_kept_separately=True,suite_complete=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    result=collect()
    if args.save:write(ROOT/'resource_accounting_latest.json',result)
    print(json.dumps(result,indent=2))
