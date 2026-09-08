"""Copy the existing source endpoint as metadata; zero initial model calls."""
from common import EXP_ROOT,atomic_json,read_json
from verge_round_core import config,ROOT
from verge_followon_protocol import prepared_round,validate_prepared


def main():
    cfg=config()
    path=ROOT/'round_frozen.json'
    if path.exists():
        validate_prepared(cfg,read_json(path));return
    source=cfg['followon_source']
    endpoint=EXP_ROOT/'raw_results'/source['source_cohort']/'branches'/str(source['candidate_index'])/'complete.json'
    initial=read_json(endpoint)
    if initial['checkpoint']!=cfg['solver_start'] or initial['training']['train_tokens']!=524288:
        raise RuntimeError('Source endpoint no longer describes the completed candidate')
    atomic_json(path,prepared_round(cfg,initial))


if __name__=='__main__':main()
