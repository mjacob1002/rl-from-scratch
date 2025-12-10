from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int

from torch import Tensor
from einops import  reduce, rearrange, repeat

class MySiLU(torch.nn.Module):

    def __init__(self):
        super(MySiLU, self).__init__()

    def forward(self, in_features: Float[Tensor, " ..." ]) -> Float[Tensor, " ..."]:
        return in_features * torch.sigmoid(in_features)
    

