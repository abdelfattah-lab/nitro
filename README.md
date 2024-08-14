# NPU Inference for LLMs
This package aims to serve LLMs with inference on Intel NPUs, using the
OpenVINO toolkit. Because NPUs currently do not support LLMs, some modifications
have been made.

Note: this framework supports OpenVINO 2024.2.0 at the moment.

![NITRO Workflow](assets/readme-diagram.png)

# Getting Started

Import the Llama model:
```python
from generation import Llama
```
Use the `from_pretrained` function, which will generate the OpenVINO IR model for the model:
```python
llama = Llama.from_pretrained(pretrained_model="meta-llama/Meta-LLama-3-8B",
                                model_dir="openvino_llama",
                                max_batch_size=1, max_seq_len=128, chunk_size=16,
                                export=True,
                                device="NPU",
                                compile=True,
                                compress=False,
                                verbose=True)
```
The following parameters are defined as follows:
* `pretrained_model` - the name of the model, as defined in Hugging Face.
* `model_dir` - the specified folder to save the loaded model information.
* `max_batch_size` and `max_seq_len` - the maximum batch size and sequence length. Note that these files
* `export` - whether to generate the model from scratch. If not, the OpenVINO IR models will be used directly. It will be checked that `pretrained_model` matches the specified model in `model_dir/config.json`.
* `device` - the device to compile the model on.
* `compile` - whether to compile the model.
* `compress` - (currently not a supported feature) whether to use NNCF weight compression.

Finally, for text generation, use the `generate()` function:
```python
output = llama.generate(prompt=["I was wondering what is going on"],
                        max_new_tokens=10)
```

# Developer Guide

TODO
