"""Registered search allocation costs, including shared evaluations and recovery."""
import argparse
import json
import subprocess
import time
from verge_search_controller import EXP,ROOT,read,write,version
from verge_book_resources import FIELDS,parse_sacct,summarize


def registry(root=ROOT):
    registered={};parents=set()
    for path in sorted((root/'jobs').glob('round_*.json')):
        record=read(path);r=record.get('round');name=version(r)
        if path.name!=f'round_{r:02d}.json' or record.get('version')!=name or record.get('branches')!=[0,1,2,3]:
            raise RuntimeError('Unrecognized search allocation registry')
        workers=record.get('historical_workers',[])+([{'job_id':record['workers'],
            'indices':record.get('worker_indices',record['branches'])}] if 'workers' in record else [])
        for entry in workers:
            parent,indices=entry['job_id'],entry['indices']
            if (not isinstance(parent,str) or not parent.isdigit() or parent in parents or not indices
                    or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<4 for i in indices)):
                raise RuntimeError('Invalid or repeated search worker registration')
            parents.add(parent)
            for b in indices:registered[f'{parent}_{b}']={'version':version(r,b),'round':r,
                'search_branch_index':b,'stage':'worker','parent_job_id':parent}
        for stage in ('prepare','finish'):
            entries=record.get('historical_'+stage,[])+([record[stage]] if stage in record else [])
            for parent in entries:
                if not isinstance(parent,str) or not parent.isdigit() or parent in parents:
                    raise RuntimeError('Invalid or repeated search shared-evaluation registration')
                parents.add(parent);registered[parent]={'version':name,'round':r,'search_branch_index':None,
                    'stage':stage,'parent_job_id':parent}
    return registered,sorted(parents,key=int)


def collect():
    registered,parents=registry()
    if not parents:return {'allocations':[],'registered_parent_jobs':0,'accounting_complete':False,
        'note':'No real target-only search jobs have been submitted','suite_complete':False}
    response=subprocess.run(['sacct','--noheader','--parsable2','--units=K','-j',','.join(parents),
        '--format='+','.join(FIELDS)],cwd=EXP,capture_output=True,text=True,check=True)
    result=summarize(parse_sacct(response.stdout),registered)
    result.update(registered_parent_jobs=len(parents),observed_at_unix=time.time(),
        scope='Only registered target-only search allocations, including shared initial/selected completion and historical recoveries',
        primary_control_followon_costs_kept_separately=True,suite_complete=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    result=collect()
    if args.save:write(ROOT/'resource_accounting_latest.json',result)
    print(json.dumps(result,indent=2))
