from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class MySoftmax(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        # get the max value across the specified dim to normalize with
        max_value_on_dim, _ = torch.max(in_features, dim=self.dim, keepdim=True)

        normalized = in_features - max_value_on_dim

        exp_tensor = torch.exp(normalized)
        sum_on_dimension = torch.sum(exp_tensor, dim=self.dim, keepdim=True)
        return exp_tensor / sum_on_dimension
