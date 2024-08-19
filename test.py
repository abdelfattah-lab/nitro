# from nitro.pytorch_model.llama.modeling_llama import LlamaModel
import torch
import torch.nn as nn
from nitro.pytorch_model.llama.modeling_llama import LlamaModel
from nitro.pytorch_model.llama.config import LlamaArgs
from nitro.converter.converter import ConversionConfig

from nitro.converter.input_generators import generate_auto

llama_args = LlamaArgs(
    _name_or_path="meta-llama/Meta-Llama-3-8B",
    architectures=["LlamaForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=128000,
    eos_token_id=128001,
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=14336,
    max_position_embeddings=8192,
    model_type="llama",
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=5e-05,
    rope_scaling=None,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
    transformers_version="4.42.4",
    use_cache=True,
    vocab_size=128256
)

conversion_args = ConversionConfig()
torch_model = LlamaModel(llama_args)

inputs = generate_auto(llama_args, conversion_args, "x", "position_ids", "mask", "kv_caches")

import openvino as ov
ov_model = ov.convert_model(torch_model, example_input=inputs)
model = ov.compile_model(ov_model, device_name="CPU")

inputs = generate_auto(llama_args, conversion_args, "x", "position_ids", "mask", "kv_caches")

true_inputs = {}
true_inputs["x"] = inputs["x"]
true_inputs["position_ids"] = torch.tensor([1])
true_inputs["mask"] = inputs["mask"]

for key, val in inputs["kv_caches"].items():
    true_inputs[key] = val

