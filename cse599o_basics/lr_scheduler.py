import torch
import math

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t <= cosine_cycle_iters:
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return lr_min

