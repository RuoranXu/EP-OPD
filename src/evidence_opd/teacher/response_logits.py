from __future__ import annotations

import torch

from evidence_opd.data import ReplayBatch


def position_ids_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(dim=-1) - 1
    return positions.clamp_min_(0)


def selected_response_hidden(model, batch: ReplayBatch) -> torch.Tensor:
    output = model.model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=position_ids_from_attention(batch.attention_mask),
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    batch_indices, response_indices = batch.selected_mask.nonzero(as_tuple=True)
    positions = batch.prediction_positions[batch_indices, response_indices]
    return hidden[batch_indices, positions]

