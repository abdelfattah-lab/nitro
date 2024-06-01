from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
import os

model_id = "meta-llama/Meta-Llama-3-8B"
device = "CPU"
quantization_config = OVWeightQuantizationConfig(bits=4)

# Load model if it exists, or generate model and save it
if os.path.exists("ov_model"):
    print("ov_model found.")

    model = OVModelForCausalLM.from_pretrained(
        "ov_model",
        compile=False,
        device=device,
        ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR":""},
        quantization_config=quantization_config
    )
else:
    print("ov_model not found.")

    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        compile=False,
        device=device,
        ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR":""},
        quantization_config=quantization_config
    )
    model.save_pretrained("ov_model")

tokenizer=AutoTokenizer.from_pretrained(model_id)

# Generate a pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Test
k = pipe(
    "How is your day today?",
    max_new_tokens = 128,
    eos_token_id=tokenizer.eos_token_id
  )
print(k)
