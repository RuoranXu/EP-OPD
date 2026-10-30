"""Equation (2): length-normalized canonical-answer preferences, no teacher gradients."""
import torch


def paired_margin(scores: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    """scores[pair, document, answer]; return a loss for each pair (sum of two hinges)."""
    if scores.ndim != 3 or scores.shape[1:] != (2, 2):
        raise ValueError("expected scores with shape [pairs, 2 documents, 2 answers]")
    return torch.relu(margin - scores[:, 0, 0] + scores[:, 0, 1]) + torch.relu(
        margin - scores[:, 1, 1] + scores[:, 1, 0]
    )


def document_token_weights(mask: torch.Tensor, pairs_in_update: int) -> torch.Tensor:
    """Eq. (5): mean tokens within each document, half-sum documents, mean pairs."""
    lengths = mask.sum(-1).clamp_min(1)
    return (torch.ones_like(mask, dtype=torch.float32) /
            (lengths[:, None] * (2 * pairs_in_update)))[mask]
