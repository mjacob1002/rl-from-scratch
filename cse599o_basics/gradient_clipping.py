import torch
from collections.abc import Iterable    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    # compute the total norm of the gradients
    total_l2_norm = torch.sqrt(sum(grad.pow(2).sum() for grad in grads))
    if total_l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * (max_l2_norm / (total_l2_norm + eps))

class MyGradientClipping:

    def __init__(self, max_l2_norm: float, eps: float = 1e-6):
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def __call__(self, parameters: Iterable[torch.nn.Parameter]):
        gradient_clipping(parameters, self.max_l2_norm, self.eps)