"""Matched greedy generation; labels never enter the model input.

python -m test.generate --model models/Qwen3-4B-Base --cache data/paired_cache.pt \
    --split test --output outputs/base_test.jsonl
For full QASPER, --inputs is JSONL containing id, paper_id and prompt_ids.
All compared checkpoints must use the same input file and generation settings.
"""
import argparse
import hashlib
import json
from pathlib import Path


def groups(jobs, max_sequences=8, token_budget=262144, cap=32768):
    batch = []
    for row in sorted(jobs, key=lambda item: len(item['prompt_ids'])):
        length = len(row['prompt_ids'])
        if length >= 32768 or not row['prompt_ids']:
            raise ValueError('prompt must be nonempty and leave room in the 32768 context')
        candidate = batch + [row]
        cost = len(candidate) * max(len(item['prompt_ids']) + min(cap, 32768-len(item['prompt_ids']))
                                    for item in candidate)
        if batch and (len(candidate) > max_sequences or cost > token_budget):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def key(row):
    return (row['pair_id'], row['variant']) if 'pair_id' in row else row['id']


def main():
    import torch
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from evidence_opd.train.stop_contract import generation_options, VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache')
    source.add_argument('--inputs')
    parser.add_argument('--split', choices=['development', 'test'], default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.max_new_tokens <= 32768:
        parser.error('max-new-tokens must be between 1 and 32768')
    path = Path(args.cache or args.inputs)
    if args.cache:
        pairs = torch.load(path, map_location='cpu', weights_only=False)[args.split]
        jobs = [dict(pair_id=pair['pair_id'], paper_id=pair['source']['paper_id'],
                     variant=variant, prompt_ids=prompt)
                for pair in pairs for variant, prompt in pair['evaluation_prompts'].items()]
    else:
        with path.open(encoding='utf8') as stream:
            jobs = [json.loads(line) for line in stream if line.strip()]
    expected = {key(row) for row in jobs}
    if not jobs or len(expected) != len(jobs):
        raise ValueError('input jobs must be nonempty and unique')
    model = Path(args.model).resolve()
    identity = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        split=args.split if args.cache else None, model=str(model),
        weights=[(file.name,file.stat().st_size,file.stat().st_mtime_ns)
                 for file in sorted(model.glob('*.safetensors'))],
        checkpoint_manifest=(model/'checkpoint_manifest.json').read_text()
                            if (model/'checkpoint_manifest.json').exists() else None,
        stop_protocol=VERSION, max_new_tokens=args.max_new_tokens, context=32768,
        decoding='greedy', seed=42, dtype='bfloat16', batch_size=8,
        backend='vllm', backend_version=vllm_version,
        termination_assisted=True, budget_policy='min(requested,32768-prompt_tokens)')
    if not identity['weights']:
        raise ValueError('model must be a local safetensors checkpoint')
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open(encoding='utf8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['signature'] != signature or key(row) in done or key(row) not in expected:
                    raise ValueError('stale, duplicate or extra output')
                done.add(key(row))
    output.with_suffix('.identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
    if done == expected:
        return
    engine = LLM(model=str(model), dtype='bfloat16', max_model_len=32768,
                 tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=.70,
                 max_num_seqs=8, enable_prefix_caching=False, trust_remote_code=False)
    with output.open('a', encoding='utf8', buffering=1) as stream:
        for batch in groups([row for row in jobs if key(row) not in done], cap=args.max_new_tokens):
            caps = [min(args.max_new_tokens,32768-len(row['prompt_ids'])) for row in batch]
            params = [SamplingParams(temperature=0.,top_p=1.,max_tokens=cap,seed=42,
                                     **generation_options(model)) for cap in caps]
            results = engine.generate([{'prompt_token_ids':row['prompt_ids']} for row in batch],
                                      sampling_params=params,use_tqdm=False)
            if len(results) != len(batch):
                raise ValueError('generation coverage differs')
            for row, result, cap in zip(batch,results,caps):
                generated = result.outputs[0]
                prediction = {k:v for k,v in row.items() if k in {'id','pair_id','variant','paper_id'}}
                prediction.update(text=generated.text,generated_tokens=len(generated.token_ids),
                    hit_cap=generated.finish_reason=='length',effective_response_cap=cap,
                    prompt_tokens=len(row['prompt_ids']),signature=signature,
                    finish_reason=generated.finish_reason,termination_is_assisted=True)
                stream.write(json.dumps(prediction,ensure_ascii=False)+'\n')
                done.add(key(row))
            print(json.dumps({'completed':len(done),'expected':len(expected)}),flush=True)
    if done != expected:
        raise ValueError('incomplete prediction coverage')


if __name__ == '__main__':
    main()
