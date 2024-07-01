from model_playground import get_memory_usage, monitor_memory_usage
from config import args, input_shapes, inputs # PARAMS DEFINED HERE
import openvino as ov
import openvino.runtime.passes as passes
import time
from nncf import compress_weights
import torch
import psutil
import threading, multiprocessing
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import ast
import torch.nn as nn
import numpy as np

from helpers import precompute_freqs_cis_rect
model = "llama.xml"
tokenizer = "openvino_tokenizer.xml"
detokenizer = "openvino_detokenizer.xml"
device = "GPU"

prompt = ["Hello, how are you?"]

# Compilation
core = ov.Core()
model = core.compile_model(model, device_name=device)
tokenizer, detokenizer = core.compile_model(tokenizer), core.compile_model(detokenizer)
tokens = tokenizer(prompt)["input_ids"].squeeze()
tokens = list(tokens)

# Instantiate the mask and KV-cache
mask = torch.full(eval(model.input("mask").get_shape().to_string()), float("-inf"))
k_cache = torch.zeros(eval(model.input("cache_k").get_shape().to_string()))
v_cache = torch.zeros(eval(model.input("cache_k").get_shape().to_string()))
freqs_cis = precompute_freqs_cis_rect(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta)

def update_mask(step:int, mask):
    mask[:,:,:,-1-step:] = 0

tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

def embedding(tok_embeddings, token:int):
    return tok_embeddings(token)

next_token = tokens.pop() # we use the last token as the first token in decode stage

# Sequential pre-fill stage
for i, t in enumerate(tokens):
    print(i, t)
    update_mask(i, mask)
    fs = freqs_cis[i:i+1]
    # x = embedding(tok_embeddings, torch.tensor(t).view(1, -1)) # HARD-CODED TO BE B = L = 1: [B,L,DIM]
    output = model({
        "cache_v": v_cache,
        "cache_k": k_cache,
        "mask": mask,
        "freqs_cis": fs,
        "x": torch.tensor([[t]])
    })
    logits = output[0]
    print(logits)
    k_cache = output[1]
    v_cache = output[2]

print(" --- DECODE STAGE --- ")
start_idx = len(tokens)
for i in range(start_idx, start_idx+10):
    print(i, next_token)
    tokens.append(next_token)

    update_mask(i, mask)
    fs = freqs_cis[i:i+1]
    # x = embedding(tok_embeddings, torch.tensor(t).view(1, -1)) # HARD-CODED TO BE B = L = 1: [B,L,DIM]
    output = model({
        "cache_v": v_cache,
        "cache_k": k_cache,
        "mask": mask,
        "freqs_cis": fs,
        "x": torch.tensor([[next_token]])
    })
    logits = output[0]
    k_cache = output[1]
    v_cache = output[2]
    print(k_cache[0,0,-20:,0,0])
    next_token = np.argmax(logits.squeeze()) # pick the next token

outputs = detokenizer(np.array(tokens).reshape(1, -1))
print(outputs["string_output"])