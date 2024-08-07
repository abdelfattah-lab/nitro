from .llama.modeling_llama import LlamaModel
from .llama.config import get_llama_args

from .qwen2.modeling_qwen2 import Qwen2Model

__all__ = [
    'LlamaModel',
    'get_llama_args',
    'Qwen2Model',
]