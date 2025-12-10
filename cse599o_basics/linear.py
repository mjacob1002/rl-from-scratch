from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cse599o_basics.tokenizer import BPETokenizer


class MyLinear(nn.Module):

    def __init__(self, d_in: int, d_out: int, weights: Float[Tensor, " d_out d_in"] | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MyLinear, self).__init__()  # Crucial line!
        self.d_in = d_in
        self.d_out = d_out
        # the init weights are in the form of 
        self.weight = weights 
        if self.weight is None:
            self.weight = torch.nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
            sigma = (2 / (d_in + d_out)) ** 0.5
            torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b = 3 * sigma)
        self.device = device
        self.dtype = dtype

    def forward(self, in_features: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return in_features @ self.weight.T





