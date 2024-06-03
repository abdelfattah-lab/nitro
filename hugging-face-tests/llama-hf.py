from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os

model_id = "meta-llama/Meta-Llama-3-8B"
device = "GPU"
model_ir = "ov_model"

# check if the folder exists
if not os.path.exists("models"):
    os.mkdir("models")

model_internal = os.path.join("models", model_ir)
# Load model if it exists, or generate model and save it
if os.path.exists(model_internal):
    print("ov_model found.")
    model = OVModelForCausalLM.from_pretrained(model_internal, device=device)
else:
    print("ov_model not found.")
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=device,)
    model.save_pretrained(model_internal)

tokenizer=AutoTokenizer.from_pretrained(model_id)

# Generate a pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
)

k = pipe(
    "How is your day today?",
    max_new_tokens = 128,
    eos_token_id=tokenizer.eos_token_id
  )
print(k)
