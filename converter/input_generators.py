"""
In conversion of PyTorch models to OpenVINO, a critical component is providing
example inputs in the convert_model function. When the model graph is complex
(which is often the case LLMs), scripting may require an example_input.

This file provides helper functions to generate example_inputs.
"""
import torch
from pytorch_model.utils.model_utils import precompute_freqs_cis_rect
from typing import Any

def generate_x(args) -> torch.Tensor:
    return torch.randint(0, args.vocab_size, [args.max_batch_size, args.inference_size])

def generate_mask(args) -> torch.Tensor:
    return torch.full([args.max_batch_size, args.n_heads, args.inference_size, args.max_seq_len], float('-inf'))

def generate_freq_cis(args) -> torch.Tensor:
    return precompute_freqs_cis_rect(
                args.dim // args.n_heads,
                args.max_seq_len * 2,
                args.rope_theta
            )[0:args.inference_size]

def generate_kv_caches(args) -> dict[str, torch.Tensor]:
    params = {}
    for i in range(args.n_layers):
        params[f"cache_k_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
        params[f"cache_v_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
    return params

def generate_auto(args, *input_names:str) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """
    Generates an example_input for conversion.

    Params:
        args (Any): the model arguments, e.g. LlamaArgs
        input_names (str): The list of input names that are used.

    The input names are standardized as the following:
    - ["x"] - token indices.
    - ["mask"] - a causal mask for the attention mechanism.
    - ["freqs_cis"] - the 
    - ["kv_caches"] - the KV-caches to be used. This keyword is special
                       because it generates a dictionary of k and v caches,
                       due the the dynamic sizing that is generally unsupported by the conversion.
    """
    inputs = {}

    mapping = {
        "x" : generate_x,
        "mask" : generate_mask,
        "freqs_cis" : generate_freq_cis,
        "kv_caches" : generate_kv_caches,
    }

    for input_name in input_names:
        if input_name not in mapping:
            raise ValueError(f"{input_name} is not a valid input name!")
        func = mapping[input_name]
        inputs[input_name] = func(args)
    
    return inputs

def flatten_dict(d):
    def recurse(t):
        items = {}
        for k, v in t.items():
            new_key = k  # Use the current key only
            if isinstance(v, dict):
                nested_items = recurse(v)
                for nk, nv in nested_items.items():
                    if nk in items:
                        raise KeyError(f"Key conflict: {nk}")
                    items[nk] = nv
            else:
                if new_key in items:
                    raise KeyError(f"Key conflict: {new_key}")
                items[new_key] = v
        return items

    return recurse(d)

def generate_shape_aux(inputs) -> dict[str, Any]:
    if isinstance(inputs, dict):
        shape_dict = {}
        for key, value in inputs.items():
            shape_dict[key] = generate_shape_aux(value)
        return shape_dict
    elif isinstance(inputs, torch.Tensor):
        return inputs.shape
    else:
        raise TypeError("Unsupported type: {}".format(type(inputs)))
    
def generate_shape(inputs) -> dict[str, torch.Size]:
    nested_shapes = generate_shape_aux(inputs)
    return flatten_dict(nested_shapes) # flattens the nested-kv_caches dict.