from nitro import LlamaPipeline
from nitro import ModelConfig, VerboseConfig, GenerationConfig

import numpy as np

model_config = ModelConfig(
    pretrained_model="meta-llama/Llama-3.2-1B",
    model_dir="llama_81",
    max_seq_len=128,
    export=True,
    do_chunk=True,
    chunk_size=16,
    compress=None
)

generation_config = GenerationConfig(
    device="CPU",
    do_sample=False,
    temperature=1.0
)

verbose_config = VerboseConfig(
    verbose = False,
    runtime = False,
    staging = False
)

llama = LlamaPipeline.from_pretrained(model_config, generation_config, verbose_config)

print(llama.generate(prompt=["The weather outside is super cold, but"], max_new_tokens=30))


for states in llama.model.models[1].query_state():
    if states.name == 'cache_v_3cache_v_3_out':
        # print(llama.tokens)
        # print(states.name)
        cache = states.state.data[0,:,0,0]
        c = 128-np.count_nonzero(cache)

        for i in range(len(llama.tokens)):
            print(f"{int(llama.tokens[i]):<5} -- {repr(llama.tokenizer.decode(llama.tokens[i])):<30} --> {float(cache[i+c]):>10.4f}")
