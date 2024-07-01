from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes
from nncf import compress_weights, CompressWeightsMode
import time
from pathlib import Path
import openvino_tokenizers
import numpy as np

# Parameters
TOKEN_SEQUENCE_LENGTH = 64
PROMPT = ["What is the meaning of"]
SAMPLE = False
COMPRESS = True
SAVE_COMPRESS = True
TEMPERATURE = 0.7

# Load and compile models
core = ov.Core()
directory = Path("models/llama3_onnx/")

ov_model = core.read_model(directory / "openvino_model_static.xml")

if COMPRESS:
    ov_model = compress_weights(ov_model, mode=CompressWeightsMode.INT8_ASYM)
    if SAVE_COMPRESS:
        ov.save_model(ov_model, directory / "openvino_model_static_compressed.xml")

ov_tokenizer = core.read_model(directory / "openvino_tokenizer.xml")
ov_detokenizer = core.read_model(directory / "openvino_detokenizer.xml")
tokenizer, detokenizer = core.compile_model(ov_tokenizer), core.compile_model(ov_detokenizer)

start = time.time()
compiled_model = ov.compile_model(ov_model, device_name="CPU")
print("Compilation time:", time.time() - start, "\n")

# Produce starting inputs
inputs={
    "input_ids":torch.tensor([[1]]),
    "attention_mask":torch.tensor([[1]]),
    "position_ids":torch.tensor([[1]]),
}
for i in range(32):
    inputs[f"past_key_values.{i}.key"] = torch.zeros([1,8,TOKEN_SEQUENCE_LENGTH,128])
    inputs[f"past_key_values.{i}.value"] = torch.zeros([1,8,TOKEN_SEQUENCE_LENGTH,128])

token_sequence = []
times = []
# Start the pre-fill stage: deconstructed to be sequential
prompt_sequence = tokenizer(PROMPT)['input_ids'][0]
for idx, token in enumerate(prompt_sequence[:-1]):
    token_sequence.append(token)
    inputs["input_ids"] = torch.tensor([[token]])
    inputs["position_ids"] = torch.tensor([[idx]])

    # Run inference
    start = time.time()
    outputs = compiled_model(inputs)
    times.append(time.time() - start)

    # Update key values based on outputs. We ignore the output token in this stage.
    for i in range(32):
        inputs[f"past_key_values.{i}.key"] = outputs.get(f"present.{i}.key")[:, :, 1:, :]
        inputs[f"past_key_values.{i}.value"] = outputs.get(f"present.{i}.value")[:, :, 1:, :]

# Start the decode stage
idx = len(prompt_sequence) - 1
token = prompt_sequence[-1]
token_sequence.append(token)

for idx in range(idx, TOKEN_SEQUENCE_LENGTH):
    inputs["input_ids"] = torch.tensor([[token]])
    inputs["position_ids"] = torch.tensor([[idx]])

    start = time.time()
    outputs = compiled_model(inputs)
    times.append(time.time() - start)

    logits = outputs.get("logits") / TEMPERATURE
    if SAMPLE:
        exps = np.exp(logits - np.max(logits))
        probabilities = (exps / np.sum(exps, axis=-1, keepdims=True)).flatten()
        token = np.random.choice(np.arange(probabilities.size), p=probabilities)
    else:
        token = np.argmax(logits)
    token_sequence.append(token)

    for i in range(32):
        inputs[f"past_key_values.{i}.key"] = outputs.get(f"present.{i}.key")[:, :, 1:, :]
        inputs[f"past_key_values.{i}.value"] = outputs.get(f"present.{i}.value")[:, :, 1:, :]

token_sequence = np.array(token_sequence).reshape(1, -1)
text_result = detokenizer(token_sequence)["string_output"]
print(f"Generated:\n{text_result[0]}")
print(np.mean(np.array(times)))