from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from threading import Thread
import intel_npu_acceleration_library
import torch
import time
import sys

from intel_npu_acceleration_library.backend import MatMul
model_id = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)


print("Compile model for the NPU")
model = intel_npu_acceleration_library.compile(model)

query = "What is the meaning of life?"
prefix = tokenizer(query, return_tensors="pt")["input_ids"]


generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
