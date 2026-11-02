#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time
import traceback
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_opd.train.stop_contract import generation_options,END_MARKER,STUDENT_EOS,boundary_reason


def fingerprint(tensor):
    value=tensor[:4,:8] if tensor.ndim==2 else tensor[:16]
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def validate_generation_policy(command,version,requested=None):
    if command!='generate' or version is None or version!=requested:
        raise ValueError('stale rollout policy')


def main():
    parent=os.getppid()
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent or parent==1:raise RuntimeError('rollout controller no longer exists')
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True)
    parser.add_argument('--socket',required=True);parser.add_argument('--utilization',type=float,default=.84)
    parser.add_argument('--max-sequences',type=int,default=32);args=parser.parse_args()
    import torch
    from vllm import LLM,SamplingParams
    if not .5<=args.utilization<=.90:raise ValueError('vLLM must retain headroom for runtime allocations')
    engine=LLM(model=args.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
               gpu_memory_utilization=args.utilization,max_num_seqs=args.max_sequences,
               max_num_batched_tokens=8192,enable_chunked_prefill=True,enable_prefix_caching=False,
               enforce_eager=False,trust_remote_code=False,skip_tokenizer_init=False,
               disable_custom_all_reduce=True,disable_log_stats=True)
    path=Path(args.socket)
    if path.exists():
        if not stat.S_ISSOCK(path.stat().st_mode):raise ValueError('refusing to replace a non-socket file')
        path.unlink()
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(path));os.chmod(path,0o600)
    listener.listen(1)
    connection,_=listener.accept();stream=connection.makefile('rwb');version=None
    print(json.dumps({'stage':'vllm_engine_ready','utilization':args.utilization}),flush=True)
    for line in stream:
        try:
            request=json.loads(line);command=request['command']
            if command=='ping':result={'ok':True,'policy_version':version}
            elif command=='reload':
                weights=Path(request['path']).resolve()
                if weights.parent!=path.parent.resolve() or weights.name!='weights.pt':
                    raise ValueError('weights must be the controller-owned atomic snapshot')
                state=torch.load(weights,map_location='cpu',mmap=True,weights_only=True)
                expected={k:fingerprint(state[k]) for k in ['model.embed_tokens.weight','model.norm.weight','lm_head.weight']}
                def load(model):
                    model.load_weights(state.items())
                    return {'model.embed_tokens.weight':fingerprint(model.model.embed_tokens.weight),
                            'model.norm.weight':fingerprint(model.model.norm.weight),
                            'lm_head.weight':fingerprint(model.lm_head.weight)}
                with torch.no_grad():observed=engine.apply_model(load)
                torch.cuda.synchronize()
                if any(value!=expected for value in observed):raise ValueError('vLLM loaded-weight fingerprint differs')
                del state
                version=int(request['policy_version'])
                result={'ok':True,'policy_version':version,'weight_fingerprints':expected}
            elif command=='generate':
                validate_generation_policy(command,version,request.get('policy_version'))
                prompts=request['prompts'];seeds=request['seeds'];cap=request['cap'];eos=set(request['eos'])
                caps=request.get('caps',[cap]*len(prompts))
                if (len(prompts)!=len(seeds) or len(caps)!=len(prompts) or
                    not all(0<c<=cap and len(p)+c<=32768 for p,c in zip(prompts,caps))):
                    raise ValueError('invalid prompt coverage or model context limit')
                guarded=request['temperature']>0
                parameters=[SamplingParams(temperature=request['temperature'],top_p=request['top_p'],
                    max_tokens=c,seed=int(seed),**generation_options(args.model,guard=guarded)) for seed,c in zip(seeds,caps)]
                started=time.monotonic()
                generated=engine.generate([{'prompt_token_ids':p} for p in prompts],parameters,use_tqdm=False)
                tokens=[];reasons=[]
                for prompt,response,effective_cap in zip(prompts,generated,caps):
                    if response.prompt_token_ids!=prompt:raise ValueError('vLLM prompt ordering differs')
                    output=response.outputs[0];ids=list(output.token_ids)
                    if output.finish_reason=='stop' and (not ids or ids[-1] not in eos):
                        stop=output.stop_reason
                        if guarded and (stop==END_MARKER or boundary_reason(output.text)):
                            # This EOS is a rule-enforced boundary, not natural EOS.
                            if len(ids)<effective_cap:ids.append(STUDENT_EOS)
                        else:
                            if stop is None:stop=engine.llm_engine.model_config.hf_config.eos_token_id
                            if not isinstance(stop,int) or stop not in eos or len(ids)>=effective_cap:
                                raise ValueError('cannot recover actual sampled EOS')
                            ids.append(stop)
                    if not ids or len(ids)>effective_cap:raise ValueError('invalid response length')
                    tokens.append(ids);reasons.append(output.finish_reason)
                result={'ok':True,'policy_version':version,'tokens':tokens,'finish_reasons':reasons,
                        'seconds':time.monotonic()-started,
                        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            else:raise ValueError('unknown rollout command')
        except Exception as exc:
            traceback.print_exc();result={'ok':False,'error':str(exc)}
        stream.write(json.dumps(result).encode()+b'\n');stream.flush()
    connection.close();listener.close();path.unlink(missing_ok=True)


if __name__=='__main__':main()
