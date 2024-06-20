from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes
import numpy as np
import openvino_tokenizers
from openvino_tokenizers import add_greedy_decoding
from config import *
import time

core = ov.Core()

# Additional inputs
text_input = ["Two plus two is"]
USE_GREEDY = False

# Read tokenizer, detokenizer models
ov_tokenizer = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3/openvino_tokenizer.xml")
ov_detokenizer = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3/openvino_detokenizer.xml")
tokenizer, detokenizer = core.compile_model(ov_tokenizer), core.compile_model(ov_detokenizer)

model = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3/openvino_model.xml")

if USE_GREEDY:
    model = add_greedy_decoding(model)

device = "HETERO:CPU,GPU,NPU"

operation_mappings = {
    "MatMul":"GPU",
    # "Swish":"GPU",
    # "Softmax":"GPU"
}

for node in model.get_ops():
    op = node.get_type_name()
    if op in operation_mappings:
        node.get_rt_info()["affinity"] = "CPU"
    else:
        node.get_rt_info()["affinity"] = "CPU"

# Compile model
print("Compiling...")
compiled_model = core.compile_model(model, device_name=device)

# Inference Setup
print("Inference...")


model_input = {
   name.any_name: output
   for name, output in tokenizer(text_input).items()
}

model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]
model_input["beam_idx"] = np.array([0], dtype=np.int32)

max_infer = 20
prompt_size = model_input["input_ids"].shape[-1]

for idx in range(prompt_size, prompt_size+max_infer):
    print(" > iteration", idx)
    # get a prediction for the last token on the first inference
    output = compiled_model(model_input)

    # processing to get the token, in a tensor shape of [-1, 1]
    if USE_GREEDY: # the output has tensor shape (1, prompt_size). use last one.
        new_token = output["token_ids"][:,-1:]
    else: # process logits 
        logits = output[0][0,-1]
        print(logits)

        logits /= 0.2 # temperature regulation.

        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        probabilities = np.nan_to_num(probabilities, nan=1)
        selected_index = np.random.choice(len(logits), p=probabilities)
        new_token = np.array([[selected_index]])
    
    # append new token to input_ids, and re-coup
    model_input["input_ids"] = np.hstack((model_input["input_ids"].data, new_token))
    model_input["attention_mask"] = np.hstack((model_input["attention_mask"].data, [[1]]))
    model_input["position_ids"] = np.hstack((model_input["position_ids"].data, [[model_input["position_ids"].data.shape[-1]]]))

   
ov_token_ids = model_input["input_ids"]
ov_output = detokenizer(ov_token_ids)["string_output"]

print(f"Prompt:\n{text_input[0]}")
print()
print(f"Output:\n{ov_output}")