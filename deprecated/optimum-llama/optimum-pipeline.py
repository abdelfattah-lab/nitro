from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import openvino.properties.device
import time

model_id = "models/llama3_optimum/"

model:OVModelForCausalLM = OVModelForCausalLM.from_pretrained(model_id, compile=False)
model.generation_config.cache_implementation = "static"
model.to("GPU")
model.compile()
# inference
prompt = "What is the meaning of"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))