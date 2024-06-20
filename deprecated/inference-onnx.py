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
ov_tokenizer = core.read_model(f"{MODEL_DIRECTORY}/openvino_tokenizer.xml")
ov_detokenizer = core.read_model(f"{MODEL_DIRECTORY}/openvino_detokenizer.xml")
tokenizer, detokenizer = core.compile_model(ov_tokenizer), core.compile_model(ov_detokenizer)

model = core.read_model(f"{MODEL_DIRECTORY}/openvino_model_modified.xml")

# Heterogeneous configuration
device = "HETERO:CPU,GPU,NPU"

operation_mappings = {
    "Q":"GPU",
    "K":"CPU",
    "V": "CPU",
}

import xml.etree.ElementTree as ET

tree = ET.parse(f"{MODEL_DIRECTORY}/openvino_model.xml")
root = tree.getroot()

# Caching purposes
idx_to_name = {} # layer id in xml -> name
name_to_idx = {}
for layer in sorted(root.findall('.//layer'), key=lambda x: int(x.get('id'))): # traverse layers in the .xml file
    id, name, type = int(layer.get('id')), layer.get('name'), layer.get('type')
    idx_to_name[id] = name
    name_to_idx[name] = id

# Store nodes into q, k, and v groups
q_group, k_group, v_group = set(), set(), set()

# Extremely hard coded
for i in range(2, 31): # transformer blocks 0-31
    print(i)
    
    q = name_to_idx[f"/model/layers.{i}/self_attn/q_proj/MatMul"]
    for j in range(q, q+50):
        q_group.add(idx_to_name[j])
    k = name_to_idx[f"/model/layers.{i}/self_attn/k_proj/MatMul"]
    for j in range(k, k+80):
        k_group.add(idx_to_name[j])
    v = name_to_idx[f"/model/layers.{i}/self_attn/v_proj/MatMul"]
    for j in range(v, v+30):
        if j in idx_to_name:
            v_group.add(idx_to_name[j])

q_count = k_count = v_count = 0
for node in model.get_ops():
    op = node.get_friendly_name()
    if op in q_group:
        node.get_rt_info()["affinity"] = operation_mappings["Q"]
        q_count += 1
    elif op in k_group:
        node.get_rt_info()["affinity"] = operation_mappings["K"]
        k_count += 1
    elif op in v_group:
        node.get_rt_info()["affinity"] = operation_mappings["V"]
        v_count += 1
    else:
        node.get_rt_info()["affinity"] = "CPU"

print("Q count:", q_count)
print("K count:", k_count)
print("V count:", v_count)

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