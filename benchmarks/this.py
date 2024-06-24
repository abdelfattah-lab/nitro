import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import time
from typing import Optional, Tuple
import math
import intel_npu_acceleration_library

from dataclasses import dataclass

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

class ReducedAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads # 32
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size # 32
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size # 32
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 1
        self.head_dim = args.dim // args.n_heads # 128

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
    ):
        start_pos = start_pos[0]
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = xk
        values = xv

        keys = repeat_kv(
            keys, self.n_rep
        ) 
        values = repeat_kv(
            values, self.n_rep
        ) 

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(
            1, 2
        )
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# torch implementation
attention = ReducedAttention(args=ModelArgs())
x = torch.rand([1,8,4096])
start_pos = torch.tensor([4])

npu = torch.compile(attention, backend="npu")
npu(x, start_pos)

# import openvino as ov
# core = ov.Core()

# ov_model = ov.convert_model(attention,
#                             example_input=[x, start_pos])

# utils.visualize_example(ov_model, "attention")
# # #### Dynamic ####
# for device in ["CPU", "GPU"]:

#     # Compilation
#     start = time.time()
#     model = core.compile_model(ov_model, device_name=device)
#     print(f"{device} Compile time: {time.time() - start}")

#     # Inference, 1000 times
#     start = time.time()
#     for _ in range(1000):
#         x = torch.rand([1,8,4096])
#         start_pos = torch.tensor([4])
#         output = model([x, start_pos])
#     print(f"{device} Inference time: {time.time() - start}")

# ov_model = ov.convert_model(attention,
#                             example_input=[x, start_pos],
#                             input=[[1,8,4096],[1]])
# #### Static ####
# for device in ["CPU", "GPU", "NPU"]:

#     # Compilation
#     start = time.time()
#     model = core.compile_model(ov_model, device_name=device)
#     print(f"{device} Compile time: {time.time() - start}")

#     # Inference, 1000 times
#     start = time.time()
#     for _ in range(1000):
#         x = torch.rand([1,8,4096])
#         start_pos = torch.tensor([4])
#         output = model([x, start_pos])
#     print(f"{device} Inference time: {time.time() - start}")
