"""Unassisted EOS/format/F1 gate; production stopping rules are deliberately disabled."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from evidence_opd.config import load_config
    from evidence_opd.train.development_guard import prepare_probe,summarize_probe
    from evidence_opd.train.stop_contract import EOS_IDS,VERSION,inspect_completion,repeated
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--baseline');parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args();config=load_config(args.config);cache=ROOT/config['data']['cache']
    payload=torch.load(cache,map_location='cpu',weights_only=False);checkpoint=Path(args.checkpoint).resolve()
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    jobs,identity=prepare_probe(payload,config['training']['development_guard'],hashlib.sha256(cache.read_bytes()).hexdigest(),ROOT)
    identity.update(stop_protocol=VERSION,checkpoint=str(checkpoint),completion_guard=False,decoding='greedy; native EOS union only')
    engine=LLM(model=str(checkpoint),dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
        gpu_memory_utilization=.70,max_num_seqs=8,max_num_batched_tokens=8192,enable_chunked_prefill=True,
        enable_prefix_caching=False,disable_custom_all_reduce=True,disable_log_stats=True)
    tokens=[];details=[];cap=config['training']['development_guard']['response_tokens']
    for begin in range(0,len(jobs),8):
        batch=jobs[begin:begin+8]
        caps=[min(cap,32768-len(j['prompt_ids'])) for j in batch]
        outputs=engine.generate([{'prompt_token_ids':j['prompt_ids']} for j in batch],
            [SamplingParams(temperature=0.,max_tokens=c,stop_token_ids=list(EOS_IDS),detokenize=False) for c in caps],use_tqdm=False)
        for job,out,maximum in zip(batch,outputs,caps):
            assert out.prompt_token_ids==job['prompt_ids'];answer=out.outputs[0];ids=list(answer.token_ids)
            if answer.finish_reason=='stop' and (not ids or ids[-1] not in EOS_IDS):
                actual=answer.stop_reason if answer.stop_reason is not None else tokenizer.eos_token_id
                if actual not in EOS_IDS or len(ids)>=maximum:raise ValueError('cannot recover actual natural EOS')
                ids.append(actual)
            tokens.append(ids);text=tokenizer.decode(ids,skip_special_tokens=True)
            details.append(dict(pair_id=job['pair_id'],variant=job['variant'],valid_completion=inspect_completion(text).valid,
                repetition=repeated(text),text=text,finish_reason=answer.finish_reason))
    summary,rows=summarize_probe(jobs,tokens,tokenizer,set(EOS_IDS),cap)
    summary['valid_completion_rate']=sum(r['valid_completion'] for r in details)/len(details)
    summary['detected_repetition_rate']=sum(r['repetition'] for r in details)/len(details)
    settings=config['sft'];failures=[]
    if summary['eos_rate']<settings['minimum_natural_eos_rate']:failures.append('natural_eos_rate')
    if summary['valid_completion_rate']<settings['minimum_valid_completion_rate']:failures.append('valid_complete_answer_rate')
    if summary['detected_repetition_rate']>settings['maximum_repetition_rate']:failures.append('repetition')
    if args.baseline:
        baseline=json.loads(Path(args.baseline).read_text())['summary']
        for key in ['original_answer_f1','counterfactual_answer_f1']:
            if summary[key]<baseline[key]-.05:failures.append(key+'_below_base_probe')
    report=dict(passed=not failures,failures=failures,summary=summary,identity=identity,
        rows=details,probe_rows=rows,tokens=tokens,split='development',test_used=False,termination_is_assisted=False,
        purpose='Small fixed behavior gate, not the full benchmark or a convergence claim')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('UNASSISTED_STOP_GATE '+json.dumps(dict(passed=report['passed'],summary=summary,failures=failures)),flush=True)
    if args.require_pass and failures:raise SystemExit(3)


if __name__=='__main__':main()
