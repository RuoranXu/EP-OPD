import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
import torch.distributed as dist
from evidence_opd.config import load_config
from evidence_opd.models import load_student,load_tokenizer
from evidence_opd.train.optimizer import build_optimizer,build_scheduler
from evidence_opd.train.checkpoint import save_student_weights
from evidence_opd.train.short_supervision import backward_clean
from evidence_opd.train.stop_contract import VERSION
from trainer.sft_stage import sft_identity




def rows_from(pairs):
    return [(p['evaluation_prompts'][v],p['clean_targets'][v]) for p in pairs for v in p['evaluation_prompts']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--size',required=True,choices=['4b','1p7b'])
    parser.add_argument('--smoke-updates',type=int,default=0);args=parser.parse_args()
    config=load_config(args.config);settings=config['sft'];rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('SFT requires four replicas')
    torch.set_num_threads(1);local_rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=1800))
    torch.manual_seed(42)
    config['models']['student']=config['models']['student_tokenizer']=config['models']['base_student']
    config['training']['learning_rate']=float(settings['learning_rate'])
    tokenizer=load_tokenizer(config['models']['student']);model=load_student(config,device)
    optimizer=build_optimizer(model,config)
    cache=Path(config['data']['cache']);payload=torch.load(cache,map_location='cpu',weights_only=False)
    training=rows_from(payload['train'])
    globalbatch=int(settings['global_batch_rows']);assert globalbatch%world==0
    steps=math.ceil(len(training)/globalbatch)*int(settings['epochs'])
    scheduler=build_scheduler(optimizer,steps,.05,'cosine')
    folder=Path(config['checkpoint']['local_root'])/f'sft_{args.size}_seed42'
    if args.smoke_updates:folder=folder.with_name(folder.name+'_smoke')
    if rank==0:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'run_manifest.json').write_text(json.dumps(dict(protocol=VERSION,initialization='fresh Base',
            identity=sft_identity(config),
            train_rows=len(training),train_pairs=len(payload['train']),epochs=settings['epochs'],maximum_updates=steps,
            global_batch_rows=globalbatch,learning_rate=settings['learning_rate'],content_weight=1.,eos_weight=2.,
            eos_weight_independent_of_length=True,gradient_splits=['train'],
            cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            synchronization='one gradient all-reduce per update across four full replicas; FP32 CPU master optimizer'),indent=2))
    dist.barrier();start=time.monotonic();step=0
    metric=(folder/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    for epoch in range(int(settings['epochs'])):
        order=list(range(len(training)));random.Random(42+epoch).shuffle(order)
        for begin in range(0,len(order),globalbatch):
            batch=[training[i] for i in order[begin:begin+globalbatch]]
            # Empty final ranks participate with zero gradients; fixed global divisor
            # keeps the final short update correctly normalized.
            local=batch[rank::world];optimizer.zero_grad(set_to_none=True);loss=0.;tick=time.monotonic()
            for prompt,target in sorted(local,key=lambda x:len(x[0])):
                stats=backward_clean(model,[prompt],[target],tokenizer,device,len(batch),1.,2.,128)
                loss+=stats['content']+stats['eos']
            # All ranks share identical initialization and update order. Explicit
            # synchronization supports chunked LM-head/backbone recomputation.
            for parameter in model.parameters():
                if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
            scalar=torch.tensor(loss,device=device);dist.all_reduce(scalar)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite SFT gradient')
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            record=dict(step=step,total_updates=steps,epoch=epoch+1,documents=len(batch),loss=float(scalar),grad_norm=float(norm),
                lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,unix_time=time.time())
            if rank==0:
                metric.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                (folder/'progress.json').write_text(json.dumps(record))
            if args.smoke_updates and step>=args.smoke_updates:break
        if args.smoke_updates and step>=args.smoke_updates:break
    if rank==0:
        destination=folder/'final'
        save_student_weights(model,tokenizer,destination,dict(optimizer_step=step,objective='short_complete_SFT',protocol=VERSION))
        (folder/'training_complete.json').write_text(json.dumps(dict(completed_updates=step,maximum_updates=steps,
            smoke=bool(args.smoke_updates),checkpoint=str(destination))))
        metric.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
