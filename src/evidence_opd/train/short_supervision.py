"""Chunked clean-completion supervision with EOS weighting independent of response length."""
import torch
from evidence_opd.data import collate_replay
from evidence_opd.teacher.response_logits import selected_response_hidden
from .stop_contract import STUDENT_EOS


def backward_clean(model,prompts,answers,tokenizer,device,normalizer,content_weight=.5,eos_weight=2.,chunk_tokens=256):
    if not answers:return dict(content=0.,eos=0.)
    if any(not a or a[-1]!=STUDENT_EOS for a in answers):raise ValueError('clean target lacks canonical EOS')
    width=max(map(len,answers));response=torch.full((len(answers),width),tokenizer.pad_token_id,device=device,dtype=torch.long)
    mask=torch.zeros_like(response,dtype=torch.bool)
    for i,values in enumerate(answers):response[i,:len(values)]=torch.tensor(values,device=device);mask[i,:len(values)]=True
    replay=collate_replay(prompts,response,mask,tokenizer.pad_token_id,device,0)
    hidden=selected_response_hidden(model,replay);gradient=torch.empty_like(hidden)
    row,pos=mask.nonzero(as_tuple=True);lengths=mask.sum(-1);last=pos.eq(lengths[row]-1)
    target=response[mask]
    # All ordinary response tokens share one per-document content weight; EOS has its own full weight.
    weights=content_weight/(lengths[row]-1).clamp_min(1)/normalizer
    weights=torch.where(last,torch.full_like(weights,eos_weight/normalizer),weights)
    stats=dict(content=0.,eos=0.)
    for start in range(0,len(hidden),chunk_tokens):
        end=min(start+chunk_tokens,len(hidden));local=hidden[start:end].detach().requires_grad_(True)
        logits=model.lm_head(local).float()
        loss=torch.nn.functional.cross_entropy(logits,target[start:end],reduction='none')
        value=(loss*weights[start:end]).sum();value.backward();gradient[start:end].copy_(local.grad)
        stats['content']+=float((loss.detach()*weights[start:end]*~last[start:end]).sum())
        stats['eos']+=float((loss.detach()*weights[start:end]*last[start:end]).sum())
    hidden.backward(gradient)
    return stats
