from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import time

from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.linear import MyLinear
from cse599o_basics.embedding import MyEmbedding
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.silu import MySiLU
from cse599o_basics.swiglu import MySwiGLU
from cse599o_basics.softmax import MySoftmax
from cse599o_basics.scaled_dot_product_attention import MySDPA
from cse599o_basics.rope import MyRoPE
from cse599o_basics.multihead_attention import MyMultiheadAttention
import einops
from einops import rearrange
# comment out wand for the autograder?
#import wandb


class MyTransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.attn = MyMultiheadAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = MySwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        #start = time.time()
        x_out = self.ln1(x)
        #ln1_end = time.time()
        # start_attn_time = torch.cuda.Event(enable_timing=True)
        # end_attn_time = torch.cuda.Event(enable_timing=True)
        # start_attn_time.record()
        x_out = self.attn(x_out)
        # end_attn_time.record()
        # torch.cuda.synchronize()
        # attn_time = start_attn_time.elapsed_time(end_attn_time)
        #wandb.log({"attn_time": attn_time})
       #attn_end = time.time()
        x_res = x + x_out
        #res_end = time.time()
        x_res_normed = self.ln2(x_res)
        #ln2_end = time.time()
        # start_ffn_time = torch.cuda.Event(enable_timing=True)
        # end_ffn_time = torch.cuda.Event(enable_timing=True)
        # start_ffn_time.record()
        x_out_2 = self.ffn(x_res_normed)
        # end_ffn_time.record()
        # torch.cuda.synchronize()
        #ffn_time = start_ffn_time.elapsed_time(end_ffn_time)
        #wandb.log({"ffn_time": ffn_time})
        #ffn_end = time.time()
        x_out_2 = x_res + x_out_2
        #end = time.time()
        #print(f"LN1 time: {ln1_end - start}; ATTN time: {attn_end - ln1_end}; RES time: {res_end - attn_end}; FFN time: {ffn_end - res_end}; Total time: {end - start}")
        return x_out_2

class MyTransformerLM(torch.nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embeddings = MyEmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.layers = torch.nn.ModuleList([
            MyTransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = MyLinear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        start = time.time()
        x = self.token_embeddings(x)
        #token_embeddings_end = time.time()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(f"Layer {i} time: {end_layer - start_layer}")
        #layers_end = time.time()
        x_post_final = self.ln_final(x)
        #ln_final_end = time.time()
        x_post_final= self.lm_head(x_post_final)
        #lm_head_end = time.time()
        #end = time.time()
        #print(f"Token embeddings time: {token_embeddings_end - start}; Layers time: {layers_end - token_embeddings_end}; LN final time: {ln_final_end - layers_end}; LM head time: {lm_head_end - ln_final_end}; Total time: {end - start}")
        return x_post_final
