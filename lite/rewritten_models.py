import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from helpers import repeat_kv, precompute_freqs_cis, apply_rotary_emb_rectangular, print_shape
from typing import Tuple, List

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: int = 1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        hidden_dim = 14336 # NOT SURE WHY THIS COMPUTES TO A DIFFERENT VALUE 
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """
    Modified Attention Block. Cache is replaced as an I/O for the forward.
    """
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,            # [B, L, D]
        mask: torch.Tensor,         # [MB, ML, L, ML]
        freqs_cis: torch.Tensor,    # []
        cache_k: torch.Tensor,      # [MB, ML, KVH, H]
        cache_v: torch.Tensor       # [MB, ML, KVH, H]
    ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb_rectangular(xq, xk, freqs_cis=freqs_cis)

        cache_k = torch.cat((cache_k[:, seqlen:], xk), dim=1)
        cache_v = torch.cat((cache_v[:, seqlen:], xv), dim=1)

        keys = cache_k[:bsz]
        values = cache_v[:bsz]

        keys = repeat_kv(keys, self.n_rep) 
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # HMM????
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(output)
        return out, cache_k, cache_v

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        out, cache_k, cache_v = self.attention(self.attention_norm(x), mask, freqs_cis, cache_k, cache_v)
        h = x + out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, cache_k, cache_v

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                cache_k: List[torch.Tensor],
                cache_v: List[torch.Tensor]
                ):
        h = self.tok_embeddings(x)
        cache_k_out = []
        cache_v_out = []
        
        for i in range(self.n_layers):
            h, cache_k_out_, cache_v_out_ = self.layers[i](h, mask, freqs_cis, cache_k[i], cache_v[i])
            cache_k_out.append(cache_k_out_.unsqueeze(0))
            cache_v_out.append(cache_v_out_.unsqueeze(0))
        
        cache_k_out = torch.concat(cache_k_out, dim=0)
        cache_v_out = torch.concat(cache_v_out, dim=0)

        h = self.norm(h)
        out = h
        out = self.output(h).float()
        return out, cache_k_out, cache_v_out