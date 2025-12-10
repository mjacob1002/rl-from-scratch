import torch
import math

# def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
#     sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
#     mask = cumulative_probs <= top_p
#     mask[:, :] = False
#     mask[:, 0] = True
#     return probs * mask

def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for tokens to keep
    sorted_mask = cumulative_probs <= top_p
    # Always include at least one token
    sorted_mask[..., 0] = True

    # Map mask back to original order
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

    # Zero out probabilities not in top-p set, then re-normalize
    filtered_probs = probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    return filtered_probs