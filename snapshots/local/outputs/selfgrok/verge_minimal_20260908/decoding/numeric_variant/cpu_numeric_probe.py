"""Real-tokenizer CPU equivalence and mask probe; no weights/model samples."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from numeric_grammar import PROGRAM_PREFIX, PAINTERS, PULLERS, rename_identifiers, POLICY_ID
from numeric_bridge import NumericGrammarService
from dsl_grammar import example_program
from hf_grammar_bridge import HFGrammarLogitsProcessor, masked_chosen_logp


def graph_equivalence(parser_path):
    spec = importlib.util.spec_from_file_location('numeric_actual_vendor_parser', parser_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kinds = PAINTERS + tuple(PULLERS)
    checks, executions = 0, 0
    rename = lambda value: re.sub(r'^n([0-9]+)$', r'\1', value)
    for count in range(31):
        for style in ('linear', 'self_loop', 'mixed'):
            source = example_program(tuple(kinds[i % 6] for i in range(count)), self_loop=style == 'self_loop')
            if style == 'mixed':
                destinations = ['start', 'end', 'NONE'] + [f'n{i}' for i in range(count)]
                position = [0]
                def change(match):
                    target = destinations[position[0] % len(destinations)]
                    position[0] += 1
                    return match[1] + target
                source = re.sub(r'(    (?:NEXT|\[(?:R|B|Y|G|EMPTY)\]) )[^\n]+', change, source)
            numeric = rename_identifiers(source)
            left = module.create_robot_factory('\n'.join(source.splitlines()[1:-1]))
            right = module.create_robot_factory('\n'.join(numeric.splitlines()[1:-1]))
            def signature(factory, convert):
                return {convert(key): (node.node_type.value,
                       tuple((route.condition, convert(route.target)) for route in node.routes))
                        for key, node in factory.nodes.items()}
            assert len(left.nodes) == len(right.nodes) == count + 2
            assert signature(left, rename) == signature(right, lambda value: value)
            for tape in ('', 'R', 'B', 'Y', 'G', 'RBYG', 'RRBB'):
                a, b = left.process_robot(tape), right.process_robot(tape)
                assert (a.finished, a.final_tape, a.rejection_reason) == (b.finished, b.final_tape, b.rejection_reason)
                assert [rename(node) for node in a.path] == b.path
                executions += 1
            checks += 1
    return {'graph_pairs': checks, 'paired_executions': executions,
            'all_2_to_32_node_counts_checked': True, 'all_types_routes_and_loops_preserved': True,
            'vendor_parser_path': str(parser_path), 'vendor_parser_sha256': hashlib.sha256(parser_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-python', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--parser', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    if not args.output.resolve().is_relative_to(directory) or args.output.exists():
        raise ValueError('use a new output inside numeric_variant')
    torch.set_num_threads(1)
    equivalence = graph_equivalence(args.parser)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, fix_mistral_regex=True)
    kinds = PAINTERS + tuple(PULLERS)
    counts = (0, 1, 2, 3, 4, 5, 6, 30)
    fixtures = [rename_identifiers(example_program(tuple(kinds[i % 6] for i in range(n)), self_loop=True)) for n in counts]
    ids = [tokenizer.encode(text[len(PROGRAM_PREFIX):], add_special_tokens=False) + [tokenizer.eos_token_id] for text in fixtures]
    prompt_ids = tokenizer.encode(PROGRAM_PREFIX, add_special_tokens=False)
    with NumericGrammarService(args.worker_python, args.model, log_name='cpu_probe_worker.stderr.log') as service:
        vocab_digest = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False,
                                               separators=(',', ':')).encode()).hexdigest()
        assert service.metadata['hf_vocab_sha256'] == vocab_digest
        assert service.metadata['decoding_policy'] == POLICY_ID
        native = torch.tensor([prompt_ids] * 8)
        processor = HFGrammarLogitsProcessor(service)
        hashes = [[] for _ in range(8)]
        started = time.monotonic()
        for position in range(max(map(len, ids))):
            scores = processor(native, torch.zeros((8, service.metadata['vocab_size'])))
            allowed = torch.isfinite(scores).cpu().numpy()
            if position == 0:
                first = {str(token): bool(allowed[0, token]) for token in (1048, 1049, 1110, 10460, 1474, tokenizer.eos_token_id)}
                assert first['1048'] and first['1049'] and not first['1110']
                assert first['10460'] and first['1474'] and not first[str(tokenizer.eos_token_id)]
            picked = []
            for row, tokens in enumerate(ids):
                if position < len(tokens):
                    token = tokens[position]
                    assert allowed[row, token]
                    hashes[row].append(hashlib.sha256(np.packbits(allowed[row], bitorder='little').tobytes()).hexdigest())
                else:
                    token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                picked.append(token)
            native = torch.cat((native, torch.tensor(picked)[:, None]), dim=1)
        mask_seconds = time.monotonic() - started
        replay_started, mask_bytes = time.monotonic(), 0
        for row, tokens in enumerate(ids):
            packed, terminated = service.replay(tokens)
            assert terminated
            assert hashes[row] == [hashlib.sha256(item.astype('<i4', copy=False).tobytes()).hexdigest() for item in packed]
            mask_bytes += packed.nbytes
        replay_seconds = time.monotonic() - replay_started
        packed, terminated = service.replay(ids[1][:3])
        assert not terminated
        try:
            service.replay(ids[1] + [tokenizer.eos_token_id])
        except ValueError:
            post_eos_rejected = True
        else:
            raise AssertionError('post-EOS padding entered replay')
        packed, _ = service.replay(ids[1])
        allowed = np.unpackbits(packed.view(np.uint8), axis=-1, bitorder='little')[:, :service.metadata['vocab_size']]
        logits = torch.zeros((len(ids[1]), service.metadata['vocab_size']), requires_grad=True)
        logp = masked_chosen_logp(logits, torch.tensor(ids[1]), packed)
        np.testing.assert_allclose(logp.detach().numpy(), -np.log(allowed.sum(-1)), atol=1e-6, rtol=1e-6)
        (-logp.mean()).backward()
        assert torch.isfinite(logits.grad).all()
        assert torch.all(logits.grad[torch.from_numpy(~allowed.astype(bool))] == 0)
        result = {**service.metadata, 'grammar_mode': service.metadata['mode'],
                  'mode': 'CPU_numeric_fixtures_not_model_samples', 'graph_equivalence': equivalence,
                  'batch_size': 8, 'suffix_tokens_per_row_including_eos': list(map(len, ids)),
                  'total_native_tokens': sum(map(len, ids)), 'max_decode_steps': max(map(len, ids)),
                  'first_native_mask': first, 'different_eos_and_padding_checked': True,
                  'all_generation_masks_equal_training_replay': True, 'post_eos_replay_rejected': post_eos_rejected,
                  'cap_truncated_prefix_replay_allowed': True, 'masked_logp_matches_sampling': True,
                  'autograd_finite_and_disallowed_gradient_zero': True,
                  'batch_mask_rpc_seconds': mask_seconds, 'replay_seconds': replay_seconds,
                  'packed_mask_bytes': mask_bytes, 'gpu_used': False, 'model_weights_loaded': False,
                  'new_model_samples': 0, 'numeric_GPU_effectiveness_established': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
