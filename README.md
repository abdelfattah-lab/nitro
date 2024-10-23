# NITRO: NPU Inference for Transformers Optimization
This package aims to serve LLMs with inference on Intel Neural Processing Units
(NPUs) on Intel Core Ultra Processors, utilizing the OpenVINO toolkit. Because
NPUs currently do not support LLMs, some modifications have been made.

![NITRO Workflow](assets/readme-diagram.png)

## Developer Notes
This package has been validated on the Meteor Lake processor, with the Linux NPU
Driver 1.6.0 and OpenVINO 2024.3.0. Currently, the frameworks supports Llama3,
with some initial support for Qwen2.

# Installation

First, clone the repository:
```
git clone https://github.com/abdelfattah-lab/nitro.git
cd nitro
```
Then, in the top-level directory, run:
```
python setup.py install
pip install .
```
Then, install PyTorch of your choice. For instance, to install PyTorch for only
CPU, run
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
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

# Developer Information

This section aims to document the motivation, architecture, and components of NITRO.

## Simplified PyTorch Models
NITRO is centered around `ov.convert_model` from PyTorch to OpenVINO IR form. Models provided by Hugging Face utilize booleans and NoneTypes frequently and, as a result, are not very friendly for OpenVINO conversion, and consequently model conversions with Optimum are prone to certain restrictions. As a result, we are re-engineering and simplifying popular model structures to support OpenVINO conversion.

The major simplifications include:
- Input/Outputs for torch modules are tensors only, or data structures that contain tensors only (e.g. lists, tuples, or dictionaries). In the latter section, we must be careful.
- Caches are represented by tensors.
- Input names are standardized: `x`, `position_ids`, `mask`, and `kv_caches` (which is a dictionary of tensors).
- Output names are standardized: there are `x` and `logits`.

## Model Conversion
Unfortunately, model conversion is not as simple as calling `ov.model_convert`. Empirically, an 8B parameter model cannot be compiled all at once on the NPU: even with RAM of 96 GB, memory runs out. To overcome this, we introduce the concept of *chunking* - breaking down the model into smaller pieces and compiling them individually. For instance, a chunking strategy that has been validated includes compiling the embedding and final linear layer separately, and compiling the 32 transformer layers on Llama-3-8B in groups of 16 (i.e. two chunks).

Even this wasn't enough: compiling them all at once also threw an error. However, our solution involves the OpenVINO model cache: we first "warm up" each chunk by compiling and deleting the object one at a time. Then, with a cached model, compilation uses less resources and allows us to compile the entire model. In the LlamaPipeline class, we then interface the different models together.

Note: We tested a full compilation of Llama3-8B with OpenVINO 2024.3, which still killed.
