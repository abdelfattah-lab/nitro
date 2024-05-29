# Provided sample code from Hugging Face.

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cpu",
)

k = pipeline("Hey how are you doing today?")
print(k)