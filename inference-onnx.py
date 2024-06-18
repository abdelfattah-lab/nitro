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
ov_tokenizer = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3_onnx/openvino_tokenizer.xml")
ov_detokenizer = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3_onnx/openvino_detokenizer.xml")
tokenizer, detokenizer = core.compile_model(ov_tokenizer), core.compile_model(ov_detokenizer)

model = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3_onnx/openvino_model.xml")

# Heterogeneous configuration
device = "HETERO:CPU,GPU,NPU"

operation_mappings = {
    "MatMul":"NPU",
    # "Swish":"GPU",
    # "Softmax":"GPU"
}

for node in model.get_ops():
    op = node.get_type_name()
    if op in operation_mappings:
        node.get_rt_info()["affinity"] = "GPU"
    else:
        node.get_rt_info()["affinity"] = "CPU"

# Compile model
if USE_GREEDY:
    model = add_greedy_decoding(model)

print("Compiling...")
start = time.time()
compiled_model = core.compile_model(model, device_name=device)
print(f" > Finished compilation: {time.time() - start} seconds.")
# Inference Setup
print("Inference...")

def pad_inputs(array:np.ndarray, length, token, length_list):
    """
    Pads [array] with [token] up to length [length].
    """
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D")
    rows, cols = array.shape
    padded_array = np.full((rows, length), token)
    for i in range(rows):
        min_length = min(cols, length)
        padded_array[i, :min(cols, min_length)] = array[i, :min(cols, min_length)]
        if len(length_list) < i+1:
            length_list.append(min_length)
    return padded_array

pad_mappings = {
    "input_ids":128001,
    "attention_mask":0
}

length_list = []

model_input = {
   name.any_name: pad_inputs(output, length=TOKEN_SEQUENCE_LENGTH, token=pad_mappings[name.any_name], length_list=length_list) 
   for name, output in tokenizer(text_input).items()
}

model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]
model_input["position_ids"][0,length_list[0]:] = -1
# model_input["beam_idx"] = np.array([0], dtype=np.int32)

max_infer = 20
prompt_size = model_input["input_ids"].shape[-1]

# infer_request = compiled_model.create_infer_request()
# infer_request.reset_state()

for idx in range(prompt_size, prompt_size+max_infer):
    print(" > Iteration", idx)
    start = time.time()
    # get a prediction for the last token on the first inference

    output = compiled_model(model_input)

    # processing to get the token, in a tensor shape of [-1, 1]
    if USE_GREEDY: # the output has tensor shape (1, prompt_size). use last one.
        new_token = output["token_ids"][:,length_list[0]-1:] # TODO: THIS IS NOT CORRECT.
    else: # process logits 
        logits = output["logits"][0,length_list[0]-1] # index of the most recent
        print(logits)
        logits /= 0.2 # temperature regulation.

        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        if np.isnan(probabilities).any():
            new_token = np.argmax(logits)
        else:
            new_token = np.random.choice(len(logits), p=probabilities)
    
    # append new token to input_ids, and re-coup\
    for i, index in enumerate(length_list):
        print(model_input)
        model_input["input_ids"][i][index] = new_token
        model_input["attention_mask"][i][index] = 1
        model_input["position_ids"][i][index] = index
        length_list[i] += 1 # increment
        # TODO: truncation
    
    print(f" >> Finished iteration. Time: {time.time() - start} seconds")
   
ov_token_ids = model_input["input_ids"]
ov_output = detokenizer(ov_token_ids)["string_output"]

print(f"Prompt:\n{text_input[0]}")
print()
print(f"Output:\n{ov_output}")