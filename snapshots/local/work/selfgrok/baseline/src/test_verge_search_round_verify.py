"""Synthetic shared-initial and selected-completion receipts; no model samples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from common import atomic_json
from verge_search_protocol import round_config,PLAN
from verge_search_sampling_contract import endpoint_request_seed
from verge_search_round_verify import validate_initial,validate_completion
from verge_independent_sampling import PROTOCOL

EXP=Path(__file__).resolve().parents[1]


def batch(path,cfg,kind,prompts,n):
    seed=endpoint_request_seed(cfg,kind)
    request={'prompt':[[1]]*prompts,'n':n,'seed':seed,'max_tokens':2048,'temperature':1.,'top_p':1.,'top_k':-1}
    records=[{'instance_id':str(i//n),'instance_index':i//n,'rollout_index':i%n,'prompt_token_ids':[1],
        'sampling_protocol':PROTOCOL,'sampling_parent_seed':seed+i//n*n,'sampling_child_seed':seed+i,
        'completion_token_ids':[2],'completion_tokens':1,'max_tokens':2048,'reward':0} for i in range(prompts*n)]
    response={'backend_protocol':PROTOCOL,'prompt_parent_seeds':[seed+i*n for i in range(prompts)],
        'child_seed_count':prompts*n,'choices':[{'index':i,'token_ids':[2]} for i in range(prompts*n)]}
    path.mkdir(parents=True,exist_ok=True)
    (path/(kind+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    atomic_json(path/(kind+'.response.json'),{'request':request,'response':response,'backend_protocol':PROTOCOL})
    return {'rollouts':prompts*n,'counts':[0]*11,'rates':[0.]*11,'bottleneck_index':0,
        'keyed_vectors':{f"{r['instance_id']}::{r['rollout_index']}":[0]*11 for r in records}}


class RoundReceiptTests(unittest.TestCase):
    def setUp(self):self.cfg=round_config(EXP,0,json.loads((EXP/PLAN).read_text()))

    def test_shared_initial_native_records_and_no_extra_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);target=batch(p,self.cfg,'target',64,8);batch(p,self.cfg,'scope',32,4)
            initial={'checkpoint':self.cfg['solver_start'],'target':target,'scope':{'full_pass_rate':0.,'mean_case_fraction':0.}}
            self.assertEqual(validate_initial(self.cfg,initial,p)['additional_model_draws'],0)
            initial['scope']['full_pass_rate']=.1
            with self.assertRaises(ValueError):validate_initial(self.cfg,initial,p)

    def test_completion_only_on_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);completion=batch(p,self.cfg,'completion',64,1)
            decision={'selected':{'checkpoint':self.cfg['solver_start']}}
            completion['checkpoint']=self.cfg['solver_start'];validate_completion(self.cfg,decision,completion,p)
            completion['checkpoint']='checkpoints/unselected'
            with self.assertRaisesRegex(ValueError,'selected'):validate_completion(self.cfg,decision,completion,p)

    def test_changed_profile_or_action_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);completion=batch(p,self.cfg,'completion',64,1)
            completion['checkpoint']=self.cfg['solver_start'];decision={'selected':{'checkpoint':self.cfg['solver_start']}}
            modified=copy.deepcopy(completion);modified['counts'][0]=1
            with self.assertRaisesRegex(ValueError,'counts'):validate_completion(self.cfg,decision,modified,p)
            path=p/'completion.response.json';saved=json.loads(path.read_text());saved['response']['choices'][0]['token_ids']=[3]
            atomic_json(path,saved)
            with self.assertRaisesRegex(ValueError,'native action'):validate_completion(self.cfg,decision,completion,p)


if __name__=='__main__':unittest.main()
