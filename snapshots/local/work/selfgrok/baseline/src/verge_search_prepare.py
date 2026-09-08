"""One worker copies the shared round-start endpoint without new model calls."""
from common import EXP_ROOT,read_json,atomic_json
from verge_round_core import ROOT,config
from verge_search_worker_protocol import validate_launch,worker_prepared,validate_prepared


def main():
    cfg=config();shared=validate_launch(EXP_ROOT,cfg);path=ROOT/'round_frozen.json'
    expected=worker_prepared(cfg,shared['initial'])
    if path.exists():
        stored=read_json(path);validate_prepared(cfg,stored)
        if stored!=expected:raise RuntimeError('Worker initial endpoint differs from the shared round snapshot')
        return
    atomic_json(path,expected)


if __name__=='__main__':main()
