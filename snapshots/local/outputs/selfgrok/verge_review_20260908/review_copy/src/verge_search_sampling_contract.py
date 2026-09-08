"""Expected backend request seeds, including the native sampler's key transform."""
from common import stable_int,read_json,read_jsonl
from verge_independent_sampling import PROTOCOL,validate_response,validate_records


def training_request_seed(cfg,group):
    stream=cfg['training_random_stream_id']
    return stable_int(stable_int(stream,42,0,group),f'{stream}_phase0_group{group}')


def endpoint_request_seed(cfg,kind):
    if kind not in ('target','scope','completion'):raise ValueError('Unknown search evaluation role')
    role='selection' if kind=='target' else kind;stream=cfg['random_stream_id']
    return stable_int(stable_int(stream,42,role),f'{stream}_{role}')


def validate_batch(cfg,directory,kind,prompts,n):
    path=directory/(kind+'.jsonl');saved=read_json(path.with_suffix('.response.json'));records=read_jsonl(path)
    request=saved['request'];response=saved['response']
    if (saved.get('backend_protocol')!=PROTOCOL or len(records)!=prompts*n
            or len(request['prompt'])!=prompts or request['n']!=n
            or request['seed']!=endpoint_request_seed(cfg,kind)):
        raise ValueError('Endpoint draw stream or batch size differs from the search contract')
    if (request.get('max_tokens')!=2048 or request.get('temperature')!=1.
            or request.get('top_p')!=1. or request.get('top_k')!=-1):
        raise ValueError('Search endpoint changed native completion or sampling settings')
    validate_response(request,response);validate_records(request,records)
    choices=sorted(response['choices'],key=lambda c:c['index'])
    for record,choice in zip(records,choices):
        ids=record.get('completion_token_ids')
        if (not isinstance(ids,list) or not 0<len(ids)<=2048 or record.get('completion_tokens')!=len(ids)
                or record.get('max_tokens')!=2048 or choice.get('token_ids')!=ids
                or record.get('reward') not in (0,1)):
            raise ValueError('Endpoint native action or binary verifier receipt changed')
    return records
