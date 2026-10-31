#!/usr/bin/env python3
import argparse
from collections import deque
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import shutil
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from evidence_opd.config import load_config
from evidence_opd.data import collate_replay
from evidence_opd.distributed import DistributedContext
from evidence_opd.train.stop_contract import semantic_stop_kl,VERSION
from evidence_opd.train.stop_training import sanitize_batch,backward_replay
from evidence_opd.train.short_supervision import backward_clean
from evidence_opd.losses.paired import paired_margin
from evidence_opd.models import load_student, load_tokenizer, assert_vocabulary_alignment
from evidence_opd.rollout import generate_responses
from evidence_opd.rollout.vllm_client import VLLMRollout,packet_batches
from evidence_opd.rollout.length_budget import batch_cap,check_response_health
from evidence_opd.teacher.hf_tp_worker import create_teacher_mesh, load_tp_teacher
from evidence_opd.teacher.response_logits import selected_response_hidden
from evidence_opd.teacher.nccl_transport import broadcast_replay_batch
from evidence_opd.train.optimizer import build_optimizer, build_scheduler, ep_learning_rate
from evidence_opd.train.checkpoint import save_student_weights
from evidence_opd.train.budget import checkpoint_steps as budget_checkpoint_steps, validate_epoch_extension, resume_scheduler, recovery_total_updates
from evidence_opd.train.development_guard import prepare_probe,summarize_probe,generate_probe,assess,improves,exchange_probe_tokens

__all__ = ['main', 'semantic_stop_kl', 'sanitize_batch', 'backward_replay']


def updates(records, seed, epochs, pairs_per_update):
    """Shuffle papers, then questions; keep each epoch's final update explicit."""
    by_paper = {}
    for row in records:
        by_paper.setdefault(row['source']['paper_id'],[]).append(row)
    rng = random.Random(seed)
    result = []
    for epoch in range(epochs):
        papers = sorted(by_paper)
        rng.shuffle(papers)
        ordered = []
        for p in papers:
            rows = list(by_paper[p])
            rng.shuffle(rows)
            ordered.extend(rows)
        result.extend((epoch, ordered[i:i+pairs_per_update]) for i in range(0,len(ordered),pairs_per_update))
    return result


def microbatches(rows, max_documents, token_budget, response_cap, context_window=None):
    rows = sorted(rows,key=lambda p:max(len(p['student_prompts'][v]) for v in ['Dorig','Dcf']))
    batch = []
    for row in rows:
        candidate = batch+[row]
        effective_cap=batch_cap(candidate,response_cap,context_window) if context_window else response_cap
        longest = max(max(len(p['student_prompts'][v]),len(p['teacher_prompts'][v]))
                      for p in candidate for v in ['Dorig','Dcf']) + effective_cap
        if batch and (len(candidate)*2 > max_documents or len(candidate)*2*longest > token_budget):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def margin_loss(model, pairs, tokenizer, device, margin):
    prompts, answers = [], []
    for pair in pairs:
        for v in ['Dorig','Dcf']:
            for answer in pair['answer_ids']:
                prompts.append(pair['student_prompts'][v])
                answers.append(answer)
    width = max(map(len,answers))
    response = torch.full((len(answers),width),tokenizer.pad_token_id,device=device,dtype=torch.long)
    mask = torch.zeros_like(response,dtype=torch.bool)
    for i,answer in enumerate(answers):
        response[i,:len(answer)] = torch.tensor(answer,device=device)
        mask[i,:len(answer)] = True
    batch = collate_replay(prompts,response,mask,tokenizer.pad_token_id,device,0)
    hidden = selected_response_hidden(model,batch)
    logits = model.lm_head(hidden).float()
    values = logits.log_softmax(-1).gather(1,response[mask].unsqueeze(1)).squeeze(1)
    row_ids = mask.nonzero(as_tuple=True)[0]
    scores = values.new_zeros(len(answers)).scatter_add(0,row_ids,values) / mask.sum(-1)
    scores = scores.reshape(len(pairs),2,2)
    return paired_margin(scores,margin), scores.detach()


def canonical_score(model, prompt, answer, tokenizer, device):
    response=torch.tensor([answer],dtype=torch.long,device=device)
    mask=torch.ones_like(response,dtype=torch.bool)
    batch=collate_replay([prompt],response,mask,tokenizer.pad_token_id,device,0)
    hidden=selected_response_hidden(model,batch)
    logits=model.lm_head(hidden).float()
    return logits.log_softmax(-1).gather(1,response.reshape(-1,1)).mean()


def backward_margin(model, pairs, tokenizer, device, margin, scale):
    """Exact hinge gradient with one live canonical-answer graph at a time."""
    if any(float(getattr(model.config,k,0.) or 0.) for k in ['attention_dropout','hidden_dropout','hidden_dropout_prob']):
        raise ValueError('streamed margin recomputation requires zero model dropout')
    with torch.no_grad():
        losses,scores=margin_loss(model,pairs,tokenizer,device,margin)
        scalar=float(losses.sum()*scale)
    if not math.isfinite(scalar):raise FloatingPointError('nonfinite paired-margin loss')
    if scalar==0.:return scalar,0
    leaf=scores.detach().requires_grad_(True)
    coefficients=torch.autograd.grad(paired_margin(leaf,margin).sum()*scale,leaf)[0]
    active=0
    for i,pair in enumerate(pairs):
        for j,variant in enumerate(['Dorig','Dcf']):
            for k,answer in enumerate(pair['answer_ids']):
                coefficient=coefficients[i,j,k]
                if float(coefficient)==0.:continue
                score=canonical_score(model,pair['student_prompts'][variant],answer,tokenizer,device)
                (score*coefficient).backward()
                active+=1
    return scalar,active


def backward_margin_bounded(model,pairs,tokenizer,device,margin,scale,token_budget=32768):
    padded=len(pairs)*4*(max(len(p['student_prompts'][v]) for p in pairs for v in ['Dorig','Dcf'])
                         +max(len(a) for p in pairs for a in p['answer_ids']))
    if padded>token_budget:
        return backward_margin(model,pairs,tokenizer,device,margin,scale)
    losses,scores=margin_loss(model,pairs,tokenizer,device,margin)
    value=losses.sum()*scale;scalar=float(value.detach())
    if not math.isfinite(scalar):raise FloatingPointError('nonfinite paired-margin loss')
    if scalar==0.:return scalar,0
    leaf=scores.requires_grad_(True)
    active=int(torch.autograd.grad(paired_margin(leaf,margin).sum(),leaf)[0].ne(0).sum())
    value.backward()
    return scalar,active


def recovery(model, optimizer, scheduler, path, step, seed, data_hash, config, total_updates=None, budget_extension=None):
    temporary = path.with_suffix('.tmp.pt')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'completed_updates':step,'seed':seed,
                'data_hash':data_hash,'config':config,'total_updates':total_updates,
                'budget_extension':budget_extension,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state()},temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/ep_opd_20261023.yaml')
    parser.add_argument('--size',choices=['4b','1p7b'],default='4b')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke-updates',type=int,default=0)
    parser.add_argument('--run-prefix',default='ep_opd')
    args=parser.parse_args()
    config=load_config(args.config)
    sft_folder=Path(config['models']['student']).resolve().parent
    if config['training'].get('require_sft_gate',True):
        gate=json.loads((sft_folder/'behavior_gate.json').read_text())
        if not gate.get('passed'):raise ValueError('SFT termination/format gate did not pass')
        if Path(gate['identity']['checkpoint']).resolve()!=Path(config['models']['student']).resolve():
            raise ValueError('SFT gate belongs to another checkpoint')
    config['training']['learning_rate']=ep_learning_rate(config,args.size)
    backend=config.get('rollout',{}).get('backend','hf')
    if backend not in {'hf','vllm'}:raise ValueError('unknown rollout backend')
    context_window=config['data'].get('context_window')
    if context_window and (backend!='vllm' or context_window!=32768):
        raise ValueError('adaptive response budgets currently require vLLM and the native 32768 context')
    rank=int(os.environ['RANK'])
    audit_path=Path(config.get('model_audit','outputs/model_audit.json'))
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=config['communication']['timeout_seconds']))
    if dist.get_world_size()!=4:
        raise ValueError('EP-OPD pipeline requires exactly four GPUs')
    compute_group=dist.new_group([0,1,2])
    # Token packets are small. CPU Gloo avoids a GPU-3 NCCL spin kernel
    # competing with the separate vLLM CUDA context during prefetch.
    rollout_group=dist.new_group([0,3],backend='gloo' if backend=='vllm' else 'nccl')
    context=DistributedContext(rank,local,4,device,rollout_group)
    mesh=create_teacher_mesh()
    seed=args.seed
    torch.manual_seed(seed)
    if not args.run_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_prefix):
        raise ValueError('run prefix must contain only letters, digits, underscore or hyphen')
    name=f'{args.run_prefix}_{args.size}_seed{seed}'+('_smoke' if args.smoke_updates else '')
    output=Path(config['checkpoint']['local_root'])/name
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
    cache_path=Path(config['data']['cache'])
    data_hash=hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if config['training'].get('require_sft_gate',True) and gate['identity']['cache_sha256']!=data_hash:
        raise ValueError('SFT gate belongs to another training cache')
    payload=torch.load(cache_path,map_location='cpu',weights_only=False)
    records=payload['train']
    guard_config=config['training'].get('development_guard')
    guard_jobs=guard_identity=baseline_tokens=baseline_summary=None
    guard_dir=output/'development_guard'
    if guard_config:
        if backend!='vllm':raise ValueError('development guard requires the existing vLLM rollout engine')
        guard_jobs,guard_identity=prepare_probe(payload,guard_config,data_hash,ROOT)
        guard_identity['base_checkpoint']=config['models']['student']
        guard_identity['signature']=hashlib.sha256(json.dumps({k:v for k,v in guard_identity.items() if k!='signature'},sort_keys=True).encode()).hexdigest()
    if len(records)<config['data']['minimum_train_pairs']:
        raise ValueError('minimum accepted training pair count not met')
    plan=updates(records,seed,config['training']['epochs'],config['data']['global_batch_pairs'])
    if args.smoke_updates:
        plan=plan[:args.smoke_updates]
    checkpoint_steps=budget_checkpoint_steps(len(plan),config['checkpoint'].get('selection_every_fraction',.25))
    student=teacher=tokenizer=optimizer=scheduler=rollout_client=None
    sync_folder=Path(config['checkpoint'].get('sync_root','outputs/rollout_sync'))/name
    start=0
    budget_extension=None
    if rank in [0,3]:
        assert_vocabulary_alignment(config)
        tokenizer=load_tokenizer(config['models']['student_tokenizer'])
        if rank==0:
            student=load_student(config,device)
            optimizer=build_optimizer(student,config)
            state_path=output/'recovery.pt'
            if state_path.exists():
                state=torch.load(state_path,map_location='cpu',weights_only=False)
                if state['data_hash']!=data_hash or state['seed']!=seed:
                    raise ValueError('recovery identity differs')
                start=state['completed_updates']
                prior_manifest=json.loads((output/'run_manifest.json').read_text())
                old_total=recovery_total_updates(state,len(records),prior_manifest['updates'],bool(args.smoke_updates))
                budget_extension=state.get('budget_extension')
                if old_total!=len(plan):
                    if os.environ.get('EP_ALLOW_BUDGET_EXTENSION')!='1':
                        raise ValueError('budget extension requires explicit campaign authorization')
                    budget_extension=validate_epoch_extension(state['config'],config,start,old_total,len(plan))
                    original_plan=updates(records,seed,state['config']['training']['epochs'],
                                          state['config']['data']['global_batch_pairs'])
                    def plan_hash(values):
                        return hashlib.sha256(json.dumps([(e,[p['pair_id'] for p in rows]) for e,rows in values],
                                                         separators=(',',':')).encode()).hexdigest()
                    budget_extension['old_plan_sha256']=plan_hash(original_plan)
                    budget_extension['new_plan_prefix_sha256']=plan_hash(plan[:old_total])
                    if budget_extension['old_plan_sha256']!=budget_extension['new_plan_prefix_sha256']:
                        raise ValueError('extended training changes the already scheduled data order')
                student.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                scheduler=resume_scheduler(optimizer,state['scheduler'],len(plan),old_total,start,
                                           config['training']['warmup_ratio'])
                if budget_extension is not None:
                    budget_extension['next_learning_rates']=scheduler.get_last_lr()
                    budget_extension['scheduler_kind']=getattr(scheduler,'schedule_kind','legacy_cosine')
                torch.set_rng_state(state['rng'])
                torch.cuda.set_rng_state(state['cuda_rng'])
                del state
            else:
                scheduler=build_scheduler(optimizer,len(plan),config['training']['warmup_ratio'],'cosine')
            manifest={'config':config,'data_hash':data_hash,'seed':seed,'updates':len(plan),
                      'cache_metadata':payload['metadata'],'rank_roles':['student','teacher_tp0','teacher_tp1','rollout'],
                      'objective':'semantic_STOP_RKL_before_teacher_stop + 0.5 paired_hinge + 0.5 clean_content_CE + 2 canonical_EOS_CE + 0.5 semantic_terminal_CE', 'initialization':'verified short-completion SFT', 'stop_protocol':VERSION,
                      'weights_sync':'each update','rollout_queue_bound':2,'resumed_at_update':start,
                      'rollout_backend':backend,
                      'budget_extension':budget_extension,'checkpoint_updates':sorted(checkpoint_steps),
                      'rollout_transport':'cpu_gloo' if backend=='vllm' else 'cuda_nccl',
                      'margin_implementation':'bounded_batched_or_full_prefix_score_recomputation',
                      'pair_backward_padded_token_budget':config['training'].get('pair_backward_padded_token_budget',32768),
                      'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                          ['trainer/ep_opd.py','workers/rollout_service.py','src/evidence_opd/rollout/vllm_client.py',
                           'src/evidence_opd/train/stop_contract.py','src/evidence_opd/train/stop_training.py','src/evidence_opd/train/short_supervision.py',
                           'src/evidence_opd/rollout/length_budget.py',
                           'src/evidence_opd/train/budget.py','src/evidence_opd/train/optimizer.py',
                           'src/evidence_opd/train/fp32_master.py']}}
            if guard_config:
                manifest['development_guard']=guard_identity
                manifest['source_sha256']['src/evidence_opd/train/development_guard.py']=hashlib.sha256(
                    (ROOT/'src/evidence_opd/train/development_guard.py').read_bytes()).hexdigest()
            if context_window:
                manifest['response_budget']={'context_window':context_window,'response_ceiling':config['data']['max_response_tokens'],
                    'policy':'min(response ceiling, context minus longest teacher/student prefix in padded microbatch)',
                    'cached_prompt_packing_unchanged':True,'response_health':config['training'].get('response_health')}
            if audit_path.exists():
                manifest['model_identity_audit']=json.loads(audit_path.read_text())
            (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        else:
            if backend=='vllm':
                visible=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
                rollout_client=VLLMRollout(config['models']['student'],sync_folder,physical_gpu=visible[local],
                    utilization=config['rollout'].get('gpu_memory_utilization',.84),
                    max_sequences=config['rollout'].get('max_sequences',32))
                if guard_config:
                    baseline_path=guard_dir/'baseline.json'
                    if baseline_path.exists():
                        prior=json.loads(baseline_path.read_text())
                        if prior['identity']!=guard_identity:raise ValueError('fixed Base development probe identity changed')
                        baseline_tokens=prior['tokens']
                    else:
                        eos_ids=sorted({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}-{None})
                        baseline_tokens=generate_probe(rollout_client,guard_jobs,guard_config,eos_ids,initial_base=True)
            else:
                student=AutoModelForCausalLM.from_pretrained(config['models']['student'],local_files_only=True,
                          dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    elif rank in [1,2]:
        teacher=load_tp_teacher(config,context,mesh)
    header=torch.tensor([start],device=device,dtype=torch.long)
    dist.broadcast(header,src=0)
    start=int(header.item())
    guard_history=[]
    if guard_config:
        history_path=guard_dir/'history.jsonl'
        if history_path.exists():guard_history=[json.loads(x) for x in history_path.read_text().split('\n') if x]
        received=exchange_probe_tokens(rank,baseline_tokens,rollout_group)
        if rank==0:
            baseline_tokens=received
            eos_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}-{None}
            baseline_summary,baseline_rows=summarize_probe(guard_jobs,baseline_tokens,tokenizer,eos_ids,guard_config['response_tokens'])
            guard_dir.mkdir(exist_ok=True)
            baseline_path=guard_dir/'baseline.json'
            baseline_path.write_text(json.dumps({'identity':guard_identity,'summary':baseline_summary,
                'rows':baseline_rows,'tokens':baseline_tokens},ensure_ascii=False,indent=2))
    guard_checked_updates={r['completed_updates'] for r in guard_history}
    if rank==0:
        metrics=(output/'metrics.jsonl').open('a',buffering=1)
        metric_history=[json.loads(x) for x in (output/'metrics.jsonl').read_text().split('\n') if x.strip()]
    cap=config['data']['max_response_tokens']
    chunk=config['communication']['kl_chunk_tokens']
    vocab=json.loads((Path(config['models']['teacher'])/'config.json').read_text())['vocab_size']
    for step in range(start,len(plan)):
        step_started=time.monotonic()
        epoch,pairs=plan[step]
        batches=list(microbatches(pairs,config['data']['microbatch_max_documents'],
                                 config['data']['microbatch_padded_token_budget'],cap,context_window))
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        if rank in [0,3]:
            if backend=='vllm':
                if rank==0:
                    sync_folder.mkdir(parents=True,exist_ok=True,mode=0o700)
                    temporary=sync_folder/'weights.tmp.pt'
                    torch.save(student.state_dict(),temporary);temporary.replace(sync_folder/'weights.pt')
                dist.barrier(group=rollout_group)
                if rank==3:
                    receipt=rollout_client.reload(sync_folder/'weights.pt',step)
                    (sync_folder/'policy_sync.json').write_text(json.dumps(receipt,indent=2))
            else:
                # Both copies have identical parameter order; no stale rollouts cross this boundary.
                for p in student.parameters():
                    dist.broadcast(p.detach(),src=0,group=rollout_group)
        if guard_config and step%guard_config['interval_updates']==0 and step not in guard_checked_updates:
            if rank==3:
                eos_ids=sorted({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}-{None})
                probe_tokens=generate_probe(rollout_client,guard_jobs,guard_config,eos_ids,version=step)
            else:probe_tokens=None
            received=exchange_probe_tokens(rank,probe_tokens,rollout_group)
            if rank==0:
                eos_ids={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}-{None}
                score,probe_rows=summarize_probe(guard_jobs,received,tokenizer,eos_ids,guard_config['response_tokens'])
                report={'completed_updates':step,'summary':score,'unix_time':time.time()}
                guard_history.append(report)
                decision=assess(guard_history,baseline_summary,guard_config)
                report['decision']=decision
                with (guard_dir/'history.jsonl').open('a',buffering=1) as handle:handle.write(json.dumps(report)+'\n')
                (guard_dir/f'probe-{step:05d}.json').write_text(json.dumps(
                    dict(report,identity=guard_identity,rows=probe_rows),ensure_ascii=False,indent=2))
                best_path=guard_dir/'best.json'
                best=json.loads(best_path.read_text()) if best_path.exists() else None
                if step>0 and improves(score,best,baseline_summary):
                    snapshot=output/f'guard-checkpoint-{step:05d}'
                    if not snapshot.exists():
                        save_student_weights(student,tokenizer,snapshot,{'optimizer_step':step,'data_hash':data_hash,
                            'seed':seed,'size':args.size,'objective':'ep_opd','guard_probe':True})
                    retained=best.get('retained',[]) if best else []
                    retained.append(snapshot.name)
                    while len(retained)>2:
                        obsolete=output/retained.pop(0)
                        saved=json.loads((obsolete/'checkpoint_manifest.json').read_text())
                        if (obsolete.is_symlink() or obsolete.resolve().parent!=output.resolve() or
                            not obsolete.name.startswith('guard-checkpoint-') or saved.get('guard_probe') is not True):
                            raise ValueError('refusing to remove an unowned guard checkpoint')
                        shutil.rmtree(obsolete)
                    best_path.write_text(json.dumps(dict(report,path=str(snapshot),retained=retained),indent=2))
                print('DEVELOPMENT_GUARD '+json.dumps(report),flush=True)
                if decision['stop']:
                    (guard_dir/'stop.json').write_text(json.dumps(dict(report,baseline=baseline_summary),indent=2))
                    raise RuntimeError('fixed development behavior deteriorated repeatedly; recovery and best weights preserved')
            guard_checked_updates.add(step)
            dist.barrier()
        if rank==0:
            optimizer.zero_grad(set_to_none=True)
        queued=deque()
        started=time.monotonic()
        sums={'rkl':0.,'pair':0.,'tokens':0,'capped':0,'documents':0,'active_margin_scores':0,'invalid_kd_documents':0,'clean_content_ce':0.,'clean_eos_ce':0.,'teacher_stop':0.,'teacher_early_stops':0,'kd_tokens':0}
        if context_window:sums.update(eos_responses=0,maximum_generated_tokens=0)
        rollout_iterator=None
        if rank==3 and backend=='vllm':
            eos=sorted({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}-{None})
            rollout_iterator=iter(packet_batches(rollout_client,batches,step,seed,cap,eos,
                config['rollout'].get('group_documents',32),config['rollout']['temperature'],config['rollout']['top_p'],
                context_window=context_window))
        for index,batch in enumerate(batches):
            def progress(phase):
                if rank==0:
                    p=output/'progress.json';temporary=p.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'completed_updates':step,'optimizer_update':step+1,
                        'total_updates':len(plan),'microbatch':index+1,'microbatches':len(batches),
                        'phase':phase,'unix_time':time.time(),'rollout_backend':backend}))
                    temporary.replace(p)
            prompts=[p['student_prompts'][v] for p in batch for v in ['Dorig','Dcf']]
            teacher_prompts=[p['teacher_prompts'][v] for p in batch for v in ['Dorig','Dcf']]
            if rank==3:
                if backend=='vllm':
                    actual,values,details=next(rollout_iterator)
                    if actual!=index:raise ValueError('rollout packet ordering differs')
                    packet=torch.full((len(prompts),cap),-1,dtype=torch.long)
                    for j,tokens in enumerate(values):packet[j,:len(tokens)]=torch.tensor(tokens)
                else:
                    responses,mask=generate_responses(student,tokenizer,prompts,config,device,
                                                      seed+1000003*step+index)
                    packet=responses.masked_fill(~mask,-1).contiguous()
                work=dist.isend(packet,dst=0,group=rollout_group)
                queued.append((work,packet))
                if len(queued)>=2:
                    queued.popleft()[0].wait()
                continue
            source=responses=mask=valid=None
            if rank==0:
                progress('waiting_for_guarded_rollout')
                packet=torch.empty((len(prompts),cap),dtype=torch.long,device='cpu')
                dist.recv(packet,src=3,group=rollout_group)
                raw_mask=packet.ge(0);lengths=raw_mask.sum(-1)
                effective_cap=batch_cap(batch,cap,context_window)
                raw_endings=packet[torch.arange(len(prompts)),lengths-1]
                ended=raw_endings.eq(151643)|raw_endings.eq(151645)
                sums['tokens']+=int(lengths.sum());sums['documents']+=len(prompts)
                sums['capped']+=int((lengths.eq(effective_cap)&~ended).sum())
                sums['eos_responses']+=int(ended.sum())
                sums['maximum_generated_tokens']=max(sums['maximum_generated_tokens'],int(lengths.max()))
                responses,mask,valid,details=sanitize_batch(tokenizer,batch,packet,device)
                sums['invalid_kd_documents']+=int((~valid).sum())
                source=collate_replay(teacher_prompts,responses,mask,tokenizer.pad_token_id,device,step,
                                      selected_mask=mask&valid[:,None])
                with (output/'replay_audit.jsonl').open('a',buffering=1) as handle:
                    for j,detail in enumerate(details):
                        handle.write(json.dumps(dict(update=step+1,pair_id=batch[j//2]['pair_id'],
                            variant=['Dorig','Dcf'][j%2],kd_accepted=bool(valid[j]),**detail))+'\n')
            teacher_batch=broadcast_replay_batch(context,source,group=compute_group)
            progress('semantic_stop_kl')
            stats=backward_replay(context,teacher_batch,teacher,student,prompts,responses,mask,valid,tokenizer,
                2*len(pairs),chunk,vocab,compute_group,config['training']['teacher_stop_weight'],
                config['training']['teacher_stop_threshold'])
            if rank==0:
                for key,value in stats.items():sums[key]+=value
                del source,teacher_batch,responses,mask
                progress('clean_complete_answer_supervision')
                for pair in batch:
                    for variant in ['Dorig','Dcf']:
                        clean=backward_clean(student,[pair['student_prompts'][variant]],
                            [pair['clean_targets'][variant]],tokenizer,device,2*len(pairs),
                            config['training']['clean_content_weight'],config['training']['clean_eos_weight'],128)
                        sums['clean_content_ce']+=clean['content'];sums['clean_eos_ce']+=clean['eos']
                # Split margin forward into pair-sized pieces when long prefixes would double padding.
                answer_width=max(len(a) for p in batch for a in p['answer_ids'])
                progress('paired_margin')
                margin_budget=config['training'].get('pair_backward_padded_token_budget',32768)
                for pair_group in microbatches(batch,max(2,config['data']['microbatch_max_documents']//2),
                                               margin_budget//2,answer_width):
                    scalar,active=backward_margin_bounded(student,pair_group,tokenizer,device,
                        config['training']['pair_margin'],config['training']['pair_weight']/len(pairs),margin_budget)
                    sums['pair']+=scalar
                    sums['active_margin_scores']+=active
            else:
                del teacher_batch
        if rollout_iterator is not None:
            try:
                next(rollout_iterator)
                raise ValueError('extra rollout packets after complete training coverage')
            except StopIteration:pass
        for work,packet in queued:
            work.wait()
        if rank==0:
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),config['training']['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite gradient norm')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            elapsed=time.monotonic()-started
            row=dict(sums,step=step+1,total_updates=len(plan),epoch=epoch,grad_norm=float(norm),
                     lr=optimizer.param_groups[0]['lr'],seconds=elapsed,tokens_per_second=sums['tokens']/elapsed,
                     pairs=len(pairs),policy_version=step+1,rollout_backend=backend,
                     peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                     peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
            checkpoint_started=time.monotonic()
            if (step+1)%config['training']['recovery_every_updates']==0:
                recovery(student,optimizer,scheduler,output/'recovery.pt',step+1,seed,data_hash,config,
                         total_updates=len(plan),budget_extension=budget_extension)
            if step+1 in checkpoint_steps:
                path=output/f'checkpoint-{step+1:05d}'
                if not path.exists():
                    save_student_weights(student,tokenizer,path,{'optimizer_step':step+1,'data_hash':data_hash,
                                         'seed':seed,'size':args.size,'objective':'ep_opd'})
            row['checkpoint_write_seconds']=time.monotonic()-checkpoint_started
            row['wall_seconds_including_sync_and_checkpoint']=time.monotonic()-step_started
            row['unix_time']=time.time()
            metrics.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            metric_history.append(row)
            health=check_response_health(metric_history,config['training'].get('response_health'))
            if len(metric_history)>=2 and all(r['invalid_kd_documents']/r['documents']>.5 for r in metric_history[-2:]):
                health=dict(reason='majority_invalid_KD_prefixes',updates=[r['step'] for r in metric_history[-2:]])
            if health:
                (output/'response_health_stop.json').write_text(json.dumps(health,indent=2))
                raise RuntimeError('response health gate stopped training after saving recovery: '+json.dumps(health))
        dist.barrier()
    if rank==0:
        (output/'training_complete.json').write_text(json.dumps({'completed_updates':len(plan),'data_hash':data_hash}))
        metrics.close()
    if rollout_client is not None:rollout_client.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__': main()
