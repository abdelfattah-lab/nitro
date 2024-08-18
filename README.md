# NITRO: NPU Inference for Transformers Optimization
This package aims to serve LLMs with inference on Intel NPUs, using the
OpenVINO toolkit. Because NPUs currently do not support LLMs, some modifications
have been made.

![NITRO Workflow](assets/readme-diagram.png)

# Installation

First, clone the repository:
```
git clone https://github.com/abdelfattah-lab/nitro.git
cd nitro
```
Then, in the top-level directory, run:
```
python setup.py
pip install .
```
# Getting Started

Import the Llama model:
```python
from nitro import LlamaPipeline
```
Use the `from_pretrained` function, which will generate the OpenVINO IR model for the model:
```python
llama = LlamaPipeline.from_pretrained(pretrained_model="meta-llama/Meta-LLama-3-8B",
                                model_dir="openvino_llama",
                                max_batch_size=1, max_seq_len=128, chunk_size=16,
                                export=True,
                                device="NPU",
                                compile=True,
                                compress=False,
                                verbose=True)
```
The parameters are given as follows:

| Parameter          | Description                                                                                         |
|--------------------|-----------------------------------------------------------------------------------------------------|
| `pretrained_model` | The name of the model, as defined in Hugging Face.                                                   |
| `model_dir`        | The specified folder to save the loaded model information.                                           |
| `max_batch_size`   | The maximum batch size. **Currently only supports 1.**                                               |
| `max_seq_len`      | The maximum sequence length.                                                                         |
| `export`           | Whether to generate the model from scratch. If not, the OpenVINO IR models will be used directly. It checks if `pretrained_model` matches the specified model in `model_dir/config.json`. |
| `device`           | The device to compile the model on.                                                                  |
| `compile`          | Whether to compile the model.                                                                        |
| `compress`         | (Currently not a supported feature) Whether to use NNCF weight compression.                         |

This will create the folder `openvino_llama` in the working directory.

Finally, for text generation, use the `generate()` function:
```python
output = llama.generate(prompt=["I was wondering what is going on"],
                        max_new_tokens=10)
```
To generate with the same model without rebuilding the model, set the `export` parameter to False.

# Developer Guide

TODO