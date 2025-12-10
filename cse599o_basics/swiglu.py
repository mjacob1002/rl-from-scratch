from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int

from torch import Tensor
from einops import  reduce, rearrange, repeat

from cse599o_basics.silu import MySiLU
from cse599o_basics.linear import MyLinear

class MySwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, w1_weight: Float[Tensor, " d_ff d_in"] | None = None, w2_weight: Float[Tensor, " d_model d_ff"] | None = None, w3_weight: Float[Tensor, " d_ff d_model"] | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MySwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = MyLinear(d_model, d_ff, w1_weight, device=device, dtype=dtype)
        self.w2 = MyLinear(d_ff, d_model, w2_weight, device=device, dtype=dtype)
        self.w3 = MyLinear(d_model, d_ff, w3_weight, device=device, dtype=dtype)
        self.silu = MySiLU()
        self.device = device
        self.dtype = dtype
        
    
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1x = self.w1(in_features)
        w3x = self.w3(in_features)
        silu_w1x = self.silu(w1x)
        return self.w2(silu_w1x * w3x)




