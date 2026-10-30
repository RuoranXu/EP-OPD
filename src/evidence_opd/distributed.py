from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    ema_group: dist.ProcessGroup

    @property
    def is_student(self) -> bool:
        return self.rank == 0

    @property
    def is_teacher_leader(self) -> bool:
        return self.rank == 1

    @property
    def is_teacher(self) -> bool:
        return self.rank in (1, 2)


def initialize_distributed(timeout_seconds: int = 600, eager_nccl_init: bool = True) -> DistributedContext:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        # Bind the rank to its logical CUDA device before NCCL creates the
        # default communicator. This is required whenever visible physical GPU
        # indices are non-contiguous (for example 0,1,3).
        timeout = timedelta(seconds=timeout_seconds)
        kwargs = {"device_id": device} if eager_nccl_init else {}
        dist.init_process_group(backend="nccl", timeout=timeout, **kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 3:
        raise ValueError(f"this backend requires exactly 3 ranks, received {world_size}")
    ema_group = dist.new_group(
        ranks=[0, 1], backend="nccl", timeout=timedelta(seconds=timeout_seconds)
    )
    return DistributedContext(rank, local_rank, world_size, device, ema_group)


def close_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
