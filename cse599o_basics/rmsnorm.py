from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int

from torch import Tensor
from einops import  reduce, rearrange, repeat

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        #Construct the RMSNorm module.
        #This function should accept the following parameters:
        d_model: int # Hidden dimension of the model
        eps: float = 1e-5 # Epsilon value for numerical stability
        device: torch.device | None # Device to store the parameters on
        dtype: torch.dtype | None # Data type of the parameters
        super(RMSNorm,self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""
        squared_rms_mean = reduce (x ** 2, "b h w -> b h 1", "mean")
        rms = torch.sqrt(squared_rms_mean + self.eps)
        # how is this interpreted by einops
        x_rms_norm = x / repeat(rms, "b s 1 -> b s d", d=self.d_model) * self.weight
        return x_rms_norm.to(self.dtype)





