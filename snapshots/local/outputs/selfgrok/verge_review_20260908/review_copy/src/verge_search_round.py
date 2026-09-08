"""One shared initial evaluation or one selected-checkpoint completion batch.

No Challenger or new curriculum sampling. Runtime namespace remains separately
gated; this program is not called by the live primary experiment.
"""
import argparse
import json
import os
import random
import time
from common import EXP_ROOT,read_json,atomic_json,stable_int
from verge_round_core import VERSION,config,profile
from verge_round_runtime import endpoint,rows,VLLMServerPool,VLLMRolloutClient
from verge_search_protocol import SUITE,prepared_round,select_round
from verge_search_worker_protocol import validate_round_launch
from verge_search_round_verify import validate_initial,validate_completion

ROOT=EXP_ROOT/'raw_results'/SUITE/'rounds'/VERSION


def port():return 20000+int(os.environ.get('SLURM_JOB_ID','0'))%25000


def keep(path,value):
    if path.exists() and read_json(path)!=value:raise RuntimeError('Frozen search round artifact changed: '+path.name)
    if not path.exists():atomic_json(path,value)


def prepare():
    cfg=config();validate_round_launch(EXP_ROOT,cfg);path=ROOT/'round_frozen.json'
    if path.exists():
        frozen=validate_round_launch(EXP_ROOT,cfg,require_prepared=True)
        validate_initial(cfg,frozen['initial'],ROOT/'initial');return
    with VLLMServerPool(gpus=(0,),base_port=port()) as urls:
        client=VLLMRolloutClient(urls[0],0)
        client.load_lora(VERSION+'_start',EXP_ROOT/cfg['solver_start'])
        initial=endpoint(client,ROOT/'initial',checkpoint=cfg['solver_start'],
            selection=rows(cfg['target_selection'])[:64],scope=rows(cfg['scope_dataset'])[:32])
        validate_initial(cfg,initial,ROOT/'initial');client.use_base()
    draw=random.Random(stable_int(cfg['random_stream_id'],'selector')).random()
    keep(path,prepared_round(cfg,initial,draw))
    print(json.dumps({'prepared_round':VERSION,'target_only_workers':4,'proposals':0}))


def finish():
    from verge_search_controller import completed_worker,completed_round
    cfg=config();frozen=validate_round_launch(EXP_ROOT,cfg,require_prepared=True);r=cfg['search_round']
    if (ROOT/'complete.json').exists():completed_round(r,deep=True);return
    validate_initial(cfg,frozen['initial'],ROOT/'initial')
    branches={b:completed_worker(r,b,deep=True) for b in range(4)}
    if any(b is None for b in branches.values()):raise RuntimeError('All four search branches are required')
    decision=select_round(frozen,branches);keep(ROOT/'decision.json',decision)
    path=ROOT/'completion_profile.json'
    if path.exists():completion=read_json(path)
    else:
        with VLLMServerPool(gpus=(0,),base_port=port()) as urls:
            client=VLLMRolloutClient(urls[0],0);selected=decision['selected']['checkpoint']
            client.load_lora(VERSION+'_retained',EXP_ROOT/selected)
            targets=rows(cfg['target_completion'])[:64];stream=cfg['random_stream_id']
            scored=client.score_rows(targets,1,stable_int(stream,42,'completion'),ROOT/'completion.jsonl',
                                    sampling_seed_key=stream+'_completion')
            completion=dict(profile(scored,targets),checkpoint=selected)
            validate_completion(cfg,decision,completion,ROOT);atomic_json(path,completion);client.use_base()
    validate_completion(cfg,decision,completion,ROOT)
    record={'protocol_version':VERSION,'search_round_complete':True,'search_block_complete':False,
        'suite_complete':False,'completed_branches':4,'round_loss_tokens':4*524288,'selected':decision['selected'],
        'official_test_opened':False,'challenger_updated':False,'completed_at':time.time(),
        'completion_full_successes':completion['counts'][-1],'completion_draws':64,
        'completion_used_for_selection':False,'all_candidate_artifacts_retained':True}
    atomic_json(ROOT/'complete.json',record)
    completed_round(r,deep=True)
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','finish'))
    {'prepare':prepare,'finish':finish}[parser.parse_args().action]()
