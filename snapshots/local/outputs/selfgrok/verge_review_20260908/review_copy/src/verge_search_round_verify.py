"""Validate existing shared-initial and selected-completion receipts; no draws."""
import math
from verge_search_sampling_contract import validate_batch
from verge_search_protocol import validate_profile


def profile_counts(profile,records):
    n=len(records);counts=profile['counts'];rates=profile['rates'];vectors=profile['keyed_vectors']
    keys=[f"{r['instance_id']}::{r['rollout_index']}" for r in records]
    if (len(set(keys))!=n or set(keys)!=set(vectors) or profile['rollouts']!=n
            or len(counts)!=11 or len(rates)!=11):raise ValueError('Profile does not cover its fixed endpoint draws')
    for key,row in zip(keys,records):
        v=vectors[key]
        if (len(v)!=11 or any(type(x) is not int or x not in (0,1) for x in v)
                or any(b>a for a,b in zip(v,v[1:])) or v[-1]!=row['reward']):
            raise ValueError('Condition profile contradicts its nested binary outcomes')
    actual=[sum(v[j] for v in vectors.values()) for j in range(11)]
    if (counts!=actual or any(not math.isclose(rate,count/n,rel_tol=0,abs_tol=1e-12) for count,rate in zip(counts,rates))):
        raise ValueError('Reported endpoint counts or rates differ from per-draw outcomes')


def validate_initial(cfg,initial,directory):
    if initial['checkpoint']!=cfg['solver_start']:raise ValueError('Initial endpoint uses another checkpoint')
    validate_profile(initial)
    records=validate_batch(cfg,directory,'target',64,8);profile_counts(initial['target'],records)
    scope=validate_batch(cfg,directory,'scope',32,4)
    if initial['scope']['full_pass_rate']!=sum(r['reward'] for r in scope)/len(scope):
        raise ValueError('Initial scope rate differs from original verifier outcomes')
    return {'shared_initial_verified':True,'additional_model_draws':0}


def validate_completion(cfg,decision,completion,directory):
    if completion.get('checkpoint')!=decision['selected']['checkpoint']:
        raise ValueError('Completion profile is not the selected checkpoint')
    records=validate_batch(cfg,directory,'completion',64,1);profile_counts(completion,records)
    return {'selected_completion_verified':True,'additional_model_draws':0}
