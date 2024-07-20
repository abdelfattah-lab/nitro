from dataclasses import dataclass

@dataclass
class ModelArgs:
    model:str = "meta-llama/Meta-Llama-3-8B"

    dim:int = 4096
    n_layers:int = 32
    n_heads:int = 32
    n_kv_heads:int = 8
    vocab_size:int = 128256
    multiple_of:int = 1024
    ffn_dim_multiplier:int = 1.3
    norm_eps:int = 5e-5
    rope_theta:int = 500000
    max_batch_size:int = 1
    max_seq_len:int = 128
    chunk_size:int = 16
    