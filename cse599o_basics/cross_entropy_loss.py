import torch
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert targets != None, "Targets cannot be None"
    assert logits != None, "Logits cannot be None"
    maximum_logit, _ = torch.max(logits, dim=-1, keepdim=True)
    normalized_logits = logits - maximum_logit
    exp_logits = torch.exp(normalized_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=-1)
    #print(f"targets: {targets.shape}, device: {targets.device}, dtype: {targets.dtype}")
    target_unsqueezed = targets.unsqueeze(-1)
    #print(f"Normalized logits: {normalized_logits.shape}, device: {normalized_logits.device}, dtype: {normalized_logits.dtype}")
    #print(f"Target unsqueezed: {target_unsqueezed.shape}, device: {target_unsqueezed.device}, dtype: {target_unsqueezed.dtype}")
    numerator = torch.gather(normalized_logits, -1, target_unsqueezed).squeeze(-1)
    log_prob = numerator - torch.log(sum_exp_logits)
    return torch.mean(-log_prob)
