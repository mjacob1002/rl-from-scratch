from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.linear import MyLinear
from cse599o_basics.embedding import MyEmbedding
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.silu import MySiLU
from cse599o_basics.swiglu import MySwiGLU
from cse599o_basics.softmax import MySoftmax
from cse599o_basics.scaled_dot_product_attention import MySDPA

class MyRoPE(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        # get the posible ks
        self.ks = torch.arange(1, d_k / 2 + 1, device=device)
        self.i_values = torch.arange(0, max_seq_len, device=device)
        self.theta_denom = 1 / (self.theta ** ((2 * self.ks - 2) / self.d_k))
        # take the outer product because it makes the shapes nice and gives every (i, k) pair
        self.theta_i_k = torch.outer(self.i_values, self.theta_denom)
        self.sin_buffer = torch.sin(self.theta_i_k)
        self.cos_buffer = torch.cos(self.theta_i_k)
        if device is not None:
            self.sin_buffer = self.sin_buffer.to(device)
            self.cos_buffer = self.cos_buffer.to(device)
        # these are for non-trainable parameters, so they won't get updated by graidents
        self.register_buffer("sin", self.sin_buffer, persistent=False)
        self.register_buffer("cos", self.cos_buffer, persistent=False)


    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]):
        """
            Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
            return a tensor of the same shape.
            Notes:
            - Accept x with an arbitrary number of batch dimensions.
            - token_positions has shape (..., seq_len) and gives absolute
            positions per token along the sequence dimension.
            - Use token_positions to slice (precomputed) cos/sin tensors
            along the sequence dimension.
        """
        # Get cos and sin for the given token positions
        # Shape: (..., sequence_length, d_k // 2)
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        
        # Apply rotation to each pair of dimensions
        res = x.clone()
        for k in range(self.d_k // 2):
            # Extract the k-th dimension: (..., sequence_length)
            cos_k = cos[..., k]
            sin_k = sin[..., k]
            
            # Get the pair of values to rotate
            x1 = x[..., 2 * k]
            x2 = x[..., 2 * k + 1]
            
            # Apply rotation: R(theta) @ [x1, x2]^T
            # [cos  -sin] [x1]   [x1*cos - x2*sin]
            # [sin   cos] [x2] = [x1*sin + x2*cos]
            res[..., 2 * k] = x1 * cos_k - x2 * sin_k
            res[..., 2 * k + 1] = x1 * sin_k + x2 * cos_k
        
        return res
