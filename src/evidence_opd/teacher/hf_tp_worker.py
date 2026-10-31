from __future__ import annotations

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForCausalLM

from evidence_opd.distributed import DistributedContext


def create_teacher_mesh() -> DeviceMesh:
    # All ranks construct the same mesh; only ranks 1 and 2 load/use the TP model.
    return DeviceMesh("cuda", [1, 2], mesh_dim_names=("tp",))


def load_tp_teacher(
    config: dict[str, Any], context: DistributedContext, mesh: DeviceMesh
):
    if not context.is_teacher:
        return None
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"],
        local_files_only=True,
        trust_remote_code=bool(config["models"].get("trust_remote_code", False)),
        dtype=torch.bfloat16,
        tp_plan="auto",
        device_mesh=mesh,
        attn_implementation=config["teacher"].get("attention_backend", "sdpa"),
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    return model

