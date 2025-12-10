from __future__ import annotations

import os
import math
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import Module

from cse599o_basics.softmax import MySoftmax

class MySDPA(torch.nn.Module):

    """Scaled Dot Product Attention"""

    def __init__(self):
        super().__init__()

    def forward(self, Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... queries keys"],
                mask: Bool[Tensor, " ... queries keys"] | None = None,
                ) -> Float[Tensor, " ... queries d_v"]:
        K_transposed = K.transpose(-1, -2)
        QK = Q @ K_transposed
        # to determine what to actually mask out, flip the bits
        actual_mask = ~mask
        # -inf makes sure that we don't attend aka when softmax is done, we get 0
        QK_scaled = QK.masked_fill_(actual_mask, -torch.inf)
        QK_scaled /= math.sqrt(Q.shape[-1])
        self.softmax = MySoftmax(dim=-1)
        QK_softmax = self.softmax(QK_scaled)
        return QK_softmax @ V
