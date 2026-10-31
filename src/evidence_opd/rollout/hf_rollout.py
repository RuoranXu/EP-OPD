from __future__ import annotations

from typing import Any

import torch

from evidence_opd.data import collate_prompts_left


def _active_through_eos(tokens: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    mask = torch.ones_like(tokens, dtype=torch.bool)
    alive = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    for position in range(tokens.shape[1]):
        mask[:, position] = alive
        is_eos = torch.zeros_like(alive)
        for eos_id in eos_ids:
            is_eos |= tokens[:, position].eq(eos_id)
        alive &= ~is_eos
    return mask


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompts: list[list[int]],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention = collate_prompts_left(prompts, tokenizer.pad_token_id, device)
    maximum = int(config["data"]["max_response_tokens"])
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)
    old_training = model.training
    old_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention,
        do_sample=bool(config["rollout"]["do_sample"]),
        temperature=float(config["rollout"]["temperature"]),
        top_p=float(config["rollout"]["top_p"]),
        top_k=int(config["rollout"].get("top_k", 0)),
        max_new_tokens=maximum,
        eos_token_id=sorted(eos_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    response = generated[:, input_ids.shape[1] :]
    if response.shape[1] < maximum:
        response = torch.nn.functional.pad(
            response, (0, maximum - response.shape[1]), value=tokenizer.pad_token_id
        )
    response = response[:, :maximum]
    response_mask = _active_through_eos(response, eos_ids)
    model.config.use_cache = old_use_cache
    model.train(old_training)
    return response, response_mask

