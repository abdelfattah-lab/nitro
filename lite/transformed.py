from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,

    LlamaModel,
    LlamaForCausalLM,

    LlamaConfig
)
import torch.nn as nn
import torch

import intel_npu_acceleration_library

llama_config = LlamaConfig()

