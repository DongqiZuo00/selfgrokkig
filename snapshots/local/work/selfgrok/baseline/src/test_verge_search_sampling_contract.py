"""Exercise the real native sampler request expression, without model requests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from common import stable_int
from verge_search_sampling_contract import training_request_seed,endpoint_request_seed


class SamplingTests(unittest.TestCase):
    def native_request(self,seed,key,n):
        source=(Path(__file__).resolve().parent/'verge_repair_sampling.py').read_text()
        f=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='score_rows')
        assignment=next(x for x in f.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='request' for t in x.targets))
        code=compile(ast.Expression(assignment.value),'actual_native_request_expression','eval')
        return eval(code,{'stable_int':stable_int,'seed':seed,'sampling_seed_key':key,'client':SimpleNamespace(model_name='synthetic'),
            'prompts':[[1]],'samples':n,'PROGRAM_STOPS':['synthetic-stop']})

    def test_training_seed_includes_sampler_key_transform(self):
        cfg={'training_random_stream_id':'verge_search_v1_train_r00_b2'};s=cfg['training_random_stream_id']
        request=self.native_request(stable_int(s,42,0,3),s+'_phase0_group3',8)
        self.assertEqual(training_request_seed(cfg,3),request['seed'])
        self.assertNotEqual(request['seed'],stable_int(s,42,0,3))

    def test_endpoint_seed_matches_each_actual_role(self):
        cfg={'random_stream_id':'verge_search_v1_paired_r00'};s=cfg['random_stream_id']
        for kind,role,n in [('target','selection',8),('scope','scope',4),('completion','completion',1)]:
            request=self.native_request(stable_int(s,42,role),s+'_'+role,n)
            self.assertEqual(endpoint_request_seed(cfg,kind),request['seed'])
            self.assertEqual(request['max_tokens'],2048)

    def test_training_branches_differ_but_endpoints_align(self):
        cfgs=[{'training_random_stream_id':f'verge_search_v1_train_r00_b{b}',
               'random_stream_id':'verge_search_v1_paired_r00'} for b in range(4)]
        self.assertEqual(len({training_request_seed(c,0) for c in cfgs}),4)
        self.assertEqual(len({endpoint_request_seed(c,'target') for c in cfgs}),1)


if __name__=='__main__':unittest.main()
