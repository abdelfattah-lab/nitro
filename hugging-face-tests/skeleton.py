from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM
import time

device_list = ['CPU', 'GPU', 'NPU']
model_list = {
    "meta-llama/Meta-Llama-3-8B"    : "Llama3",
    "meta-llama/Llama-2-7b-hf"      : "Llama2"
}

for name in model_list:
    for device in device_list:
        print(f"Testing {model_list[name]} on device {device}:")

        # Instantiate and compile model
        start = time.time()
        model = OVModelForCausalLM.from_pretrained(name, export=True, device_map=device)
        end = time.time()
        print("\t ➜ Instantite/compile model time elapsed:", end - start, "seconds")
        del model
        time.sleep(1)
