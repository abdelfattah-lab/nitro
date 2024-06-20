from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model_id = "/home/abdelfattah/openvino-llama/models/llama3_optimum/"

model:OVModelForCausalLM = OVModelForCausalLM.from_pretrained(model_id,
                                           ov_config={
                                            #    "KV_CACHE_PRECISION": "u8",
                                            #    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                                               "PERFORMANCE_HINT": "LATENCY"},
                                            device="HETERO:GPU,CPU"
                                           )

# inference

prompt = "What is going on?"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))