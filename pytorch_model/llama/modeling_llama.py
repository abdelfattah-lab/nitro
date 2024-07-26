import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from pytorch_model.helpers import repeat_kv, apply_rotary_emb_rectangular
from pytorch_model.llama.config import ModelArgs

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        x_squared = x.pow(2)
        x_means = x_squared.mean(-1, keepdim=True)
        x_means += self.eps
        x_denom = torch.sqrt(x_means)
        return torch.div(x, x_denom)

    def forward(self, x:torch.Tensor):
        output = self._norm(x)
        # propagate = output.clone().detach()
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
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """
    Modified Attention Block. Cache is replaced as an I/O for the forward.
    """
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
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

        # It might be more efficient to use torch.roll. However, the Torchscript doesn't quite convert successfully.
        cache_k = torch.cat((cache_k[:, seqlen:], xk), dim=1)
        cache_v = torch.cat((cache_v[:, seqlen:], xv), dim=1)

        keys = cache_k[:bsz]
        values = cache_v[:bsz]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(1, 2)

        # Scaled Dot Product Attention
        # output = nn.functional.scaled_dot_product_attention(xq, keys, values, mask, 0, False)

        # Manual
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, values)
        
        # Output
        output = output.transpose(1, 2).view(bsz, seqlen, -1)

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
        x_norm = self.attention_norm(x)
        out, cache_k, cache_v = self.attention(x_norm, mask, freqs_cis, cache_k, cache_v)
        h = x + out
        h_norm = self.ffn_norm(h)
        out = h + self.feed_forward(h_norm)
        return out, cache_k, cache_v

class Llama(nn.Module):
    def __init__(self, params: ModelArgs, offset:int=0, chunk_size:int=-1):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Including statuses
        self.include_embedding = False
        self.include_transformer = False
        self.include_output = False

        # Parameters
        self.offset = offset
        self.chunk_size = chunk_size
        if self.chunk_size == -1:
            self.chunk_size = self.n_layers


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
                params: dict):

        if self.include_embedding:
            x = self.embedding(x, mask, freqs_cis, params)
        
        if self.include_transformer:
            x, cache_k_outs, cache_v_outs = self.transformer_chunk(x, mask, freqs_cis, params)
        
        if self.include_output:
            x = self.output_chunk(x, mask, freqs_cis, params)

        out = x

        # Preparing the outputs for appropriate naming
        outputs = {"logit" if self.include_output else "x" : out}
        if self.include_transformer:
            for i in range(self.chunk_size):
                outputs[f"cache_k_{i + self.offset}_out"] = cache_k_outs[i]
                outputs[f"cache_v_{i + self.offset}_out"] = cache_v_outs[i]

        # Return the output along with the cache tensors
        return outputs

    def embedding(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                params: dict):
        
        return self.tok_embeddings(x)
    
    def transformer_chunk(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                params: dict):
        
        cache_k_outs = []
        cache_v_outs = []
        for i in range(self.offset, self.offset + self.chunk_size):    
            x, cache_k_out_, cache_v_out_ = self.layers[i](x, mask, freqs_cis, params[f'cache_k_{i}'], params[f'cache_v_{i}'])
            cache_k_outs.append(cache_k_out_)
            cache_v_outs.append(cache_v_out_)

        return x, cache_k_outs, cache_v_outs
    
    def output_chunk(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                params: dict):
        
        return self.output(self.norm(x))