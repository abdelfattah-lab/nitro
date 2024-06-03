from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM
import time
from utils import print_bold

device_list = ['CPU', 'GPU', 'HETERO:CPU,GPU', 'HETERO:GPU,CPU']
model_list = {
    "meta-llama/Meta-Llama-3-8B"    : "Llama3",
    # "meta-llama/Llama-2-7b-hf"      : "Llama2"
}
inference_prompts = [
    "Once upon a time,",
    "Hello, how are you?",
    "You are going to"
]

for model_id in model_list:
    for device in device_list:
        print_bold(f"➜➜➜ Testing {model_list[model_id]} on device {device}:")

        ##### Instantiate and compile OpenVINO Model #####
        start = time.time()
        
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=device)
        
        end = time.time()
        print_bold(f"\t ➜ Compile model time: {end - start} seconds")
        

        ##### Instantiate tokenizer #####
        start = time.time()

        tokenizer=AutoTokenizer.from_pretrained(model_id)

        end = time.time()
        print_bold(f"\t ➜ Tokenizer time: {end - start} seconds")

        ##### Pipeline setup #####
        start = time.time()
        
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        
        end = time.time()
        print_bold(f"\t ➜ Pipeline time: {end - start} seconds")

        ##### Testing inference #####
        for i, prompt in enumerate(inference_prompts):
            start = time.time()

            output = pipe(
                prompt,
                max_new_tokens = 128,
                eos_token_id=tokenizer.eos_token_id
            )
            print(output)

            end = time.time()
            print_bold(f"\t ➜ Inference {i+1} time: {end - start} seconds")

            del output

        time.sleep(1)

        # delete and restart
        del model
        del tokenizer
        del pipe
