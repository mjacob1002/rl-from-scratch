from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import Module

class MyEmbedding(Module):

    def __init__(self, vocab_size: int, d_model: int, weights: Float[Tensor, " vocab_size d_model"] | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MyEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = weights
        if self.weight is None:
            self.weight = torch.nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
            torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b = 3)
        self.device = device
        self.dtype = dtype


    def forward(self, token_ids: Int[Tensor, "..."]) -> Tensor:
        try:
            res = self.weight[token_ids]
            return res
        except Exception as e:
            raise ValueError(f"The indexing didn't work: {e}")

