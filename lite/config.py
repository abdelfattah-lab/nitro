import torch
from dataclasses import dataclass
from helpers import precompute_freqs_cis_rect

@dataclass
class ModelArgs:
    dim: int = 4096 # 2048, 4096 FAILS.
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256        # manually changed this value just now.
    multiple_of: int = 256          # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: int = 1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 1         # Any higher batch number fails.
    max_seq_len: int = 512 # LENGTH OF THE CACHE. 

args = ModelArgs()

# 
DIM = args.dim
B = args.max_batch_size
L = 1
ML = args.max_seq_len
KVH = args.n_kv_heads
NH = args.n_heads # TODO: CURRENT HARD-CODED FIX.
HD = args.dim // args.n_heads
TYPE = torch.float32
RT = args.rope_theta
LAYERS = args.n_layers

freqs_cis = precompute_freqs_cis_rect(
    args.dim // args.n_heads,
    args.max_seq_len * 2,
    args.rope_theta)

pos = 5
freqs_cis = freqs_cis[5:5+L]

inputs = {
    "x"         : torch.randint(0, 128256, [B, L]),
    # "x"         : torch.randn([B, L, DIM]),
    "mask"      : torch.full([B, NH, L, ML], float('-inf'),  dtype=TYPE),
    "freqs_cis" : freqs_cis,
    "cache_k"   : torch.zeros([LAYERS, B, ML, KVH, HD], dtype=TYPE),
    "cache_v"   : torch.zeros([LAYERS, B, ML, KVH, HD], dtype=TYPE),
}

inputs["mask"][:,:,:,-1:] = 0

input_shapes = {k: v.shape for k, v in inputs.items()}
