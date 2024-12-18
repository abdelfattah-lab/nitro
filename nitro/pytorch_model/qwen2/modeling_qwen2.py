import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from nitro.pytorch_model.utils.model_utils import repeat_kv, apply_rotary_emb_rectangular
from nitro.pytorch_model.qwen2.config import Qwen2Args

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
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(
        self,
        args: Qwen2Args
    ):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def forward(self, x:torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Attention(nn.Module):
    """
    Modified Attention Block. Cache is replaced as an I/O for the forward.
    """
    def __init__(self, args:  Qwen2Args):
        super().__init__()
        self.n_kv_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,            # [B, L, D]
        mask: torch.Tensor,         # [MB, ML, L, ML]
        freqs_cis: torch.Tensor,    # []
        cache_k: torch.Tensor,      # [MB, ML, KVH, H]
        cache_v: torch.Tensor       # [MB, ML, KVH, H]
    ):
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)        

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

        out = self.o_proj(output)
        return out, cache_k, cache_v

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args:   Qwen2Args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        x_norm = self.input_layernorm(x)
        out, cache_k, cache_v = self.self_attn(x_norm, mask, freqs_cis, cache_k, cache_v)
        h = x + out
        h_norm = self.post_attention_layernorm(h)
        out = h + self.mlp(h_norm)
        return out, cache_k, cache_v

class Qwen2Model(nn.Module):
    def __init__(self, params:Qwen2Args, offset:int=0):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        # Including statuses
        self.include_embedding = False
        self.include_transformer = False
        self.include_output = False

        # Parameters
        self.offset = offset
        self.chunk_size = -1
        if self.chunk_size == -1:
            self.chunk_size = self.n_layers


        self.embed_tokens = nn.Embedding(
            params.vocab_size, params.hidden_size
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        self.lm_head = nn.Linear(
            params.hidden_size, params.vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                kv_caches: dict):

        if self.include_embedding:
            x = self.embedding(x, mask, freqs_cis, kv_caches)
        
        if self.include_transformer:
            x, cache_k_outs, cache_v_outs = self.transformer_chunk(x, mask, freqs_cis, kv_caches)
        
        if self.include_output:
            x = self.output_chunk(x, mask, freqs_cis, kv_caches)

        out = x

        # Preparing the outputs for appropriate naming
        outputs = {"logits" if self.include_output else "x" : out}
        if self.include_transformer:
            for i in range(self.chunk_size):
                outputs[f"cache_k_{i + self.offset}_out"] = cache_k_outs[i]
                outputs[f"cache_v_{i + self.offset}_out"] = cache_v_outs[i]

        # Return the output along with the cache tensors
        return outputs

    def embedding(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                kv_caches: dict):
        
        return self.embed_tokens(x)
    
    def transformer_chunk(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                kv_caches: dict):
        
        cache_k_outs = []
        cache_v_outs = []
        for i in range(self.offset, self.offset + self.chunk_size):    
            x, cache_k_out_, cache_v_out_ = self.layers[i](x, mask, freqs_cis, kv_caches[f'cache_k_{i}'], kv_caches[f'cache_v_{i}'])
            cache_k_outs.append(cache_k_out_)
            cache_v_outs.append(cache_v_out_)

        return x, cache_k_outs, cache_v_outs
    
    def output_chunk(self, x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: torch.Tensor,
                kv_caches: dict):
        
        return self.lm_head(self.norm(x))
    
    def set_chunk_size(self, num:int) -> None:
        self.chunk_size = num