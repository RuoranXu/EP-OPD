from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass
class ReplayBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor
    selected_mask: torch.Tensor
    policy_version: int

    @property
    def selected_tokens(self) -> int:
        return int(self.selected_mask.sum().item())


class TokenizedCorpus:
    def __init__(self, path: str | Path) -> None:
        records = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.by_id = {str(row["id"]): row for row in records}
        if len(self.by_id) != len(records):
            raise ValueError("tokenized corpus contains duplicate IDs")

    def get_many(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        return [self.by_id[str(item)] for item in ids]


def read_batch_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collate_prompts_left(
    prompts: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        start = width - len(prompt)
        input_ids[row, start:] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[row, start:] = 1
    return input_ids, attention


def collate_replay(
    prompts: list[list[int]],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pad_id: int,
    device: torch.device,
    policy_version: int,
    selected_mask: torch.Tensor | None = None,
) -> ReplayBatch:
    batch, response_width = responses.shape
    lengths = [len(prompt) + response_width for prompt in prompts]
    width = max(lengths)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((batch, width), dtype=torch.long, device=device)
    positions = torch.empty((batch, response_width), dtype=torch.long, device=device)
    responses = responses.to(device=device, dtype=torch.long)
    response_mask = response_mask.to(device=device, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        prompt_length = prompt_tensor.numel()
        input_ids[row, :prompt_length] = prompt_tensor
        input_ids[row, prompt_length : prompt_length + response_width] = responses[row]
        active = int(response_mask[row].sum().item())
        attention[row, : prompt_length + active] = 1
        positions[row] = prompt_length - 1 + torch.arange(response_width, device=device)
    if selected_mask is None:
        selected_mask = response_mask
    else:
        selected_mask = selected_mask.to(device=device, dtype=torch.bool) & response_mask
    return ReplayBatch(input_ids, attention, positions, selected_mask, int(policy_version))

