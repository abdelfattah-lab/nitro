from nitro import LlamaPipeline
from nitro import ModelConfig, VerboseConfig, GenerationConfig
from nncf import CompressWeightsMode

import numpy as np

# Model configurations for exporting and converting
model_config = ModelConfig(
    pretrained_model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_dir="llama_8_instruct",
    max_seq_len=128,
    export=True,
    do_chunk=True,
    chunk_size=16,
    compress=CompressWeightsMode.INT8_ASYM
)

# Generation process after exporting model
generation_config = GenerationConfig(
    device="NPU",
    do_sample=False,
    temperature=1.0
)

# Print statement configurations
verbose_config = VerboseConfig(
    verbose = False
)

llama = LlamaPipeline.from_pretrained(model_config, generation_config, verbose_config)

# For chat generation
llama.chat_generate(max_new_tokens=30)
