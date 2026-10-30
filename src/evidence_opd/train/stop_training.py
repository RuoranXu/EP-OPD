"""Sanitized replay and teacher-stop horizons, bound by the new training entry only."""
import torch
import torch.distributed as dist
from evidence_opd.data import collate_replay
from evidence_opd.teacher.response_logits import selected_response_hidden
from evidence_opd.teacher.nccl_transport import token_chunks,send_logits,receive_logits
from .stop_contract import make_replay,semantic_stop_kl,stop_nll,teacher_stop_positions,STUDENT_EOS


def sanitize_batch(tokenizer,pairs,packet,device):
    values=[];valid=[];details=[]
    for i,(pair,variant) in enumerate((p,v) for p in pairs for v in ['Dorig','Dcf']):
        raw=packet[i][packet[i].ge(0)].tolist()
        tokens,accepted,meta=make_replay(tokenizer,raw,pair['clean_targets'][variant],pair['retained_block_ids'])
        # A fallback target is for CE only, never a supposed on-policy sample.
        values.append(tokens);valid.append(accepted);details.append(meta)
    width=max(map(len,values));response=torch.full((len(values),width),tokenizer.pad_token_id,device=device,dtype=torch.long)
    mask=torch.zeros_like(response,dtype=torch.bool)
    for i,tokens in enumerate(values):response[i,:len(tokens)]=torch.tensor(tokens,device=device);mask[i,:len(tokens)]=True
    valid=torch.tensor(valid,device=device,dtype=torch.bool)
    return response,mask,valid,details


def backward_replay(context,teacher_batch,teacher,student,prompts,responses,mask,valid,tokenizer,
                    normalizer,chunk,vocab,group,stop_weight=.5,threshold=.5):
    """Teacher-selected STOP ends the KD horizon; no later token can receive KD gradient.

    Teacher logits are projected twice from one frozen backbone forward: a first
    pass finds the causal stop position, the second transfers only retained rows.
    Invalid/repetitive trajectories were removed before the teacher forward.
    """
    rank=context.rank;device=context.device
    stats=dict(rkl=0.,teacher_stop=0.,teacher_early_stops=0,kd_tokens=0)
    if not teacher_batch.selected_tokens:return stats
    selected=teacher_batch.selected_mask
    rows,positions=selected.nonzero(as_tuple=True)
    horizon=torch.full((selected.shape[0],),selected.shape[1],device=device,dtype=torch.long)
    if rank in [1,2]:
        with torch.inference_mode():
            hidden=selected_response_hidden(teacher,teacher_batch)
            for a,b in token_chunks(len(hidden),chunk):
                logits=teacher.lm_head(hidden[a:b])
                first=teacher_stop_positions(logits,rows[a:b],positions[a:b],len(horizon),threshold)
                horizon=torch.minimum(horizon,torch.where(first.ge(0),first,horizon))
                del logits
    dist.broadcast(horizon,src=1,group=group)
    kept=positions.le(horizon[rows])
    # Each response's first semantic STOP is terminal, including when the teacher
    # requests it before the student's evidence boundary.
    effective=selected.clone();effective[selected]=kept
    if rank in [1,2]:hidden=hidden[kept]
    else:
        adjusted=responses.clone();active=mask.clone()
        for row,end in enumerate(horizon.tolist()):
            if end<selected.shape[1] and valid[row]:
                adjusted[row,end]=STUDENT_EOS;active[row,end+1:]=False
                stats['teacher_early_stops']+=int(end<int(mask[row].sum())-1)
        replay=collate_replay(prompts,adjusted,active,tokenizer.pad_token_id,device,
                              teacher_batch.policy_version,selected_mask=effective)
        hidden=selected_response_hidden(student,replay);gradient=torch.empty_like(hidden)
    row,pos=effective.nonzero(as_tuple=True)
    counts=effective.sum(-1).clamp_min(1)
    terminal=pos.eq(counts[row]-1)
    weights=1./counts[row]/normalizer
    for a,b in token_chunks(int(effective.sum()),chunk):
        if rank in [1,2]:
            with torch.inference_mode():logits=teacher.lm_head(hidden[a:b])
            if rank==1:send_logits(logits)
            del logits
        else:
            projection=hidden[a:b].detach().requires_grad_(True)
            logits=student.lm_head(projection);reference=receive_logits(b-a,vocab,device)
            kl=semantic_stop_kl(logits,reference)
            boundary=stop_nll(logits)*terminal[a:b]*stop_weight/normalizer
            loss=(kl*weights[a:b]).sum()+boundary.sum()
            loss.backward();gradient[a:b].copy_(projection.grad)
            stats['rkl']+=float((kl.detach()*weights[a:b]).sum())
            stats['teacher_stop']+=float(boundary.detach().sum())
    if rank==0:
        hidden.backward(gradient);stats['kd_tokens']=int(effective.sum())
    return stats
