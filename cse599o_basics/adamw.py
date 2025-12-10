from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                t = state.get('t', 1)
                grad = p.grad
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)
                p.data = p.data * (1 - weight_decay * lr)
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss

                

