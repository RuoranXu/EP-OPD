"""Explicit epoch extensions with continuous learning rates and exact recovery."""
import copy
import math

import torch

from .optimizer import build_scheduler


def checkpoint_steps(total_steps, fraction=.25):
    count=round(1/fraction) if 0<fraction<=1 else 0
    if total_steps<1 or count<1 or not math.isclose(count*fraction,1.,abs_tol=1e-9):
        raise ValueError('checkpoint fraction must partition the full training budget')
    return {max(1,math.ceil(total_steps*i/count)) for i in range(1,count+1)}


def recovery_total_updates(state,records_count,manifest_total,smoke=False):
    total=state.get('total_updates')
    if total is None:
        # The manifest may already describe an extension that crashed before
        # its first recovery. Legacy optimizer state still owns the old plan.
        total=manifest_total if smoke else state['config']['training']['epochs']*math.ceil(
            records_count/state['config']['data']['global_batch_pairs'])
    if not 0<state['completed_updates']<=total:raise ValueError('invalid recovery training budget')
    return total


def validate_epoch_extension(saved_config, requested_config, completed, old_total, new_total):
    before=copy.deepcopy(saved_config);after=copy.deepcopy(requested_config)
    old_epochs=before['training'].pop('epochs');new_epochs=after['training'].pop('epochs')
    before.get('checkpoint',{}).pop('selection_every_fraction',None)
    after.get('checkpoint',{}).pop('selection_every_fraction',None)
    if before!=after:
        raise ValueError('epoch extension cannot silently change data, models, loss, batch or optimizer')
    if not(0<completed<old_total<new_total and new_epochs>old_epochs):
        raise ValueError('epoch extension requires a positive unfinished recovery and a larger budget')
    return {'old_epochs':old_epochs,'new_epochs':new_epochs,'completed_updates':completed,
            'old_total_updates':old_total,'new_total_updates':new_total,
            'schedule_policy':'preserve next learning rate, then cosine decay over remaining updates',
            'optimizer_rng_cursor_preserved':True}


class AnchoredCosineLR(torch.optim.lr_scheduler.LRScheduler):
    schedule_kind='anchored_cosine_extension_v1'

    def __init__(self,optimizer,total_steps,anchor_update,anchor_lrs):
        if not 0<anchor_update<total_steps or not all(lr>0 for lr in anchor_lrs):
            raise ValueError('extension requires a positive current learning rate before completion')
        self.total_steps=total_steps
        self.anchor_update=anchor_update
        self.anchor_lrs=list(anchor_lrs)
        self.schedule_kind=type(self).schedule_kind
        super().__init__(optimizer)

    def get_lr(self):
        progress=min(max(self.last_epoch/(self.total_steps-self.anchor_update),0.),1.)
        return [lr*.5*(1+math.cos(math.pi*progress)) for lr in self.anchor_lrs]


def resume_scheduler(optimizer,saved,total_steps,old_total,completed,warmup_ratio):
    next_lrs=list(saved['_last_lr'])
    if len(next_lrs)!=len(optimizer.param_groups):raise ValueError('scheduler group count differs')
    kind=saved.get('schedule_kind')
    expected_epoch=completed-saved['anchor_update'] if kind==AnchoredCosineLR.schedule_kind else completed
    if saved['last_epoch']!=expected_epoch:raise ValueError('scheduler and data cursor differ')
    for group,lr in zip(optimizer.param_groups,next_lrs):group['lr']=lr
    if total_steps>old_total:
        scheduler=AnchoredCosineLR(optimizer,total_steps,completed,next_lrs)
    elif total_steps<old_total:
        raise ValueError('cannot shorten a recovered training budget')
    elif kind==AnchoredCosineLR.schedule_kind:
        scheduler=AnchoredCosineLR(optimizer,total_steps,saved['anchor_update'],saved['anchor_lrs'])
        scheduler.load_state_dict(saved)
    else:
        scheduler=build_scheduler(optimizer,total_steps,warmup_ratio,'cosine')
        scheduler.load_state_dict(saved)
    # LambdaLR initialization and legacy CPU-master snapshots can carry a stale
    # optimizer LR. The scheduler records the actual rate for the next update.
    for group,lr in zip(optimizer.param_groups,next_lrs):group['lr']=lr
    return scheduler
