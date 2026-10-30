from __future__ import annotations

from typing import Iterator

import torch
import torch.distributed as dist

from evidence_opd.data import ReplayBatch
from evidence_opd.distributed import DistributedContext


def broadcast_replay_batch(
    context: DistributedContext, source_batch: ReplayBatch | None, group=None
) -> ReplayBatch:
    if context.is_student:
        if source_batch is None:
            raise ValueError("rank 0 must provide the replay batch")
        batch, width = source_batch.input_ids.shape
        response_width = source_batch.prediction_positions.shape[1]
        header = torch.tensor(
            [batch, width, response_width, source_batch.policy_version],
            dtype=torch.long,
            device=context.device,
        )
    else:
        header = torch.empty(4, dtype=torch.long, device=context.device)
    dist.broadcast(header, src=0, group=group)
    batch_size, width, response_width, policy_version = [int(value) for value in header.tolist()]

    if context.is_student:
        result = source_batch
        selected_u8 = result.selected_mask.to(torch.uint8)
    else:
        result = ReplayBatch(
            input_ids=torch.empty((batch_size, width), dtype=torch.long, device=context.device),
            attention_mask=torch.empty((batch_size, width), dtype=torch.long, device=context.device),
            prediction_positions=torch.empty(
                (batch_size, response_width), dtype=torch.long, device=context.device
            ),
            selected_mask=torch.empty(
                (batch_size, response_width), dtype=torch.bool, device=context.device
            ),
            policy_version=policy_version,
        )
        selected_u8 = torch.empty(
            (batch_size, response_width), dtype=torch.uint8, device=context.device
        )
    dist.broadcast(result.input_ids, src=0, group=group)
    dist.broadcast(result.attention_mask, src=0, group=group)
    dist.broadcast(result.prediction_positions, src=0, group=group)
    dist.broadcast(selected_u8, src=0, group=group)
    result.selected_mask = selected_u8.bool()
    return result


def token_chunks(total: int, chunk_tokens: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, chunk_tokens):
        yield start, min(start + chunk_tokens, total)


def send_logits(logits: torch.Tensor, destination: int = 0) -> None:
    dist.send(logits.to(dtype=torch.bfloat16).contiguous(), dst=destination)


def receive_logits(
    token_count: int, vocab_size: int, device: torch.device, source: int = 1
) -> torch.Tensor:
    buffer = torch.empty(
        (token_count, vocab_size), dtype=torch.bfloat16, device=device
    )
    dist.recv(buffer, src=source)
    return buffer


@torch.no_grad()
def update_ema_across_ranks(
    local_model,
    context: DistributedContext,
    decay: float,
) -> None:
    if context.rank not in (0, 1):
        return
    for parameter in local_model.parameters():
        if context.rank == 0:
            transfer = parameter.detach()
        else:
            transfer = torch.empty_like(parameter)
        dist.broadcast(transfer, src=0, group=context.ema_group)
        if context.rank == 1:
            parameter.mul_(decay).add_(transfer, alpha=1.0 - decay)

