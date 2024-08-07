from dataclasses import dataclass
from pytorch_model.utils.config_utils import get_args_aux
from typing import Optional

@dataclass
class LlamaArgs:
    _name_or_path: str
    architectures: list[str]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: Optional[None]
    rope_theta: float
    tie_word_embeddings: bool
    torch_dtype: str
    transformers_version: str
    use_cache: bool
    vocab_size: int

    max_batch_size:int = 1
    max_seq_len:int = 128

import os
from pathlib import Path

def get_llama_args(model_name:str) -> LlamaArgs:
    """
    Obtains the model.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path = Path(path) / "config.json"

    args = get_args_aux(model_name, LlamaArgs, path)
    return args