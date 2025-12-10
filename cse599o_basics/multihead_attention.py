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
from cse599o_basics.rope import MyRoPE
import einops
from einops import rearrange

class MyMultiheadAttention(torch.nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = MyLinear(d_model, d_model, device=device, dtype=dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = MyRoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.my_sdpa = MySDPA()
        self.dtype = dtype
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"], apply_rope: bool = True,token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, " ... sequence_length d_model"]:
        # this is the concatenated q, k, v for all heads, where each head can be found in d_model // num_heads 
        *batch_dims, seq_len, _ = x.shape
        q, k , v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        dim_head = self.d_model // self.num_heads
        q = einops.rearrange(q, "... s  (h d) -> ... h s d", h=self.num_heads)
        k = einops.rearrange(k, "... s  (h d) -> ... h s d", h=self.num_heads)
        v = einops.rearrange(v, "... s  (h d) -> ... h s d", h=self.num_heads)
        if apply_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        # causal attention mask to make sure that future tokens are not attended to
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        res = self.my_sdpa(q, k, v, mask)
        res = einops.rearrange(res, "... h s d -> ... s (h d)")
        return self.output_proj(res)



            
        

     