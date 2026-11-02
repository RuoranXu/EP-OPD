from __future__ import annotations

import torch


def exact_kl_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    divergence: str,
    temperature: float = 1.0,
    temperature_squared_scale: bool = False,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_logp = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if divergence == "reverse_kl":
        values = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    elif divergence == "forward_kl":
        values = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    else:
        raise ValueError(f"unsupported divergence: {divergence}")
    if temperature_squared_scale:
        values = values * (temperature * temperature)
    return values

