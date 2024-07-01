from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache, LlamaForCausalLM
import openvino.properties.device
import time
import torch

model_id = "meta-llama/Llama-2-7b-hf"

model:LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_id)

model.generation_config.cache_implementation = "static"

# inference
tokenizer = AutoTokenizer.from_pretrained(model_id)
compiled_model = torch.compile(model)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cpu")

outputs = compiled_model.generate(**input_ids)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))