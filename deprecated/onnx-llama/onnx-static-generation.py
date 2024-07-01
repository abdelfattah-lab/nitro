from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes

core = ov.Core()

TOKEN_SEQUENCE_LENGTH = 64

# Model conversion from Optimum model to Llama model
port_to_shape = {
    "input_ids":[1,1],
    "attention_mask":[1,1],
    "position_ids":[1,1],
}

for i in range(32):
    port_to_shape[f"past_key_values.{i}.key"] = [1,8,TOKEN_SEQUENCE_LENGTH,128]
    port_to_shape[f"past_key_values.{i}.value"] = [1,8,TOKEN_SEQUENCE_LENGTH,128]

# Produce some example inputs
example_input={
    "input_ids":torch.randint(0,128002, [1, 1]),
    "attention_mask":torch.zeros([1, 1]),
    "position_ids":torch.arange(0, 1).reshape(1, 1),
}

for key in port_to_shape:
    example_input[key] = torch.rand(port_to_shape[key])

assert port_to_shape.keys() == example_input.keys()

model = ov.convert_model(f"models/llama3_onnx/model.onnx",
                         example_input=example_input
)
model.reshape(port_to_shape) # provide static input shape

# Save
ov.save_model(model, f"models/llama3_onnx/openvino_model_static.xml")