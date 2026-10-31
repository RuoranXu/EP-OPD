from __future__ import annotations

import math
from typing import Any

import torch


def ep_learning_rate(config: dict[str, Any], size: str) -> float:
    if size not in {'4b', '1p7b'}:
        raise ValueError('unknown EP student size')
    legacy = 1e-5 if size == '4b' else 1.5e-5
    value = float(config['training'].get('learning_rate_by_size', {}).get(size, legacy))
    if not math.isfinite(value) or value <= 0:
        raise ValueError('invalid EP learning rate')
    return value


def build_optimizer(model, config: dict[str, Any]):
    training = config["training"]
    kwargs = {
        "lr": float(training["learning_rate"]),
        "betas": tuple(float(value) for value in training["betas"]),
        "weight_decay": float(training["weight_decay"]),
    }
    precision = training.get('optimizer_precision', 'model_dtype')
    if precision == 'fp32_master_cpu':
        from .fp32_master import FP32MasterAdamW
        return FP32MasterAdamW(model.parameters(), **kwargs)
    if precision != 'model_dtype':
        raise ValueError(f'unknown optimizer precision: {precision}')
    if bool(training.get("fused_adamw", False)):
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, schedule: str = "linear"):
    if schedule not in {"linear", "cosine"}:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        denominator = max(total_steps - warmup_steps, 1)
        if schedule == "cosine":
            progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(remaining) / float(denominator)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)

