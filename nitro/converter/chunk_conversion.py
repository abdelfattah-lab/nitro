import openvino as ov
import re
from openvino.runtime.passes import Manager, MakeStateful
import torch
import numpy as np
from typing import Any
from nncf import compress_weights, CompressWeightsMode

def get_param_names(model:ov.Model) -> dict[str, Any]:
    params = model.get_parameters()
    names = {}
    for p in params:
        names[p.get_friendly_name()] = p
    return names

def parse_and_rename_layers(model: ov.Model):
    """
    Parses and renames layers in the model by adding the offset to the numerical part of the friendly names.

    Parameters:
        model (Model): The model to modify.
        offset (int): The offset to add to the numerical part of the friendly names.
    """
    import re

    for node in model.get_parameters():
        name = node.get_friendly_name()
        
        # Extract the numerical part from the name using regex
        match = re.search(r'_(\d+)$', name)
        if match:
            num = int(match.group(1))
            # Create the new name
            new_name = f"{name[:-len(str(num))]}{num}"
            node.set_friendly_name(new_name)

    for node in model.get_results():
        output_name = node.output(0).names.pop()
        node.set_friendly_name(output_name)

    return model

def make_stateful(model: ov.Model):

    # Get the parameters and results of the model
    parameters = model.get_parameters()
    results = model.get_results()

    # Define regex patterns for matching parameter and result names
    cache_k_pattern = re.compile(r"cache_k_(\d+)")
    cache_v_pattern = re.compile(r"cache_v_(\d+)")
    cache_k_out_pattern = re.compile(r"cache_k_(\d+)_out")
    cache_v_out_pattern = re.compile(r"cache_v_(\d+)_out")

    # Maps to store parameters and results by their index
    cache_k_params = {}
    cache_v_params = {}
    cache_k_results = {}
    cache_v_results = {}

    # Populate the maps with parameters
    for param in parameters:
        name = param.get_friendly_name()
        match = cache_k_pattern.match(name)
        if match:
            index = int(match.group(1))
            cache_k_params[index] = param
        match = cache_v_pattern.match(name)
        if match:
            index = int(match.group(1))
            cache_v_params[index] = param

    # Populate the maps with results
    for result in results:
        name = result.get_friendly_name()
        match = cache_k_out_pattern.match(name)
        if match:
            index = int(match.group(1))
            cache_k_results[index] = result
        match = cache_v_out_pattern.match(name)
        if match:
            index = int(match.group(1))
            cache_v_results[index] = result

    # Create pairs of matching parameters and results
    pairs = []
    for index, param in cache_k_params.items():
        if index in cache_k_results:
            pairs.append((param, cache_k_results[index]))
    for index, param in cache_v_params.items():
        if index in cache_v_results:
            pairs.append((param, cache_v_results[index]))

    manager = Manager()
    manager.register_pass(MakeStateful(pairs))
    manager.run_passes(model)

    # MakeStateful does not provide the (optional) input parameter
    constant = None
    for node in model.get_ops():
        if node.get_type_name() == "ReadValue":

            output_shape = node.get_output_shape(0)
            # output_type = node.get_output_element_type(0)
            
            # Create a constant node with zero values
            if constant is None:
                constant = ov.runtime.opset1.constant(np.zeros(output_shape, dtype=np.float32))

            node.set_arguments([constant])
    
    return model

def conversion_wrapper(model,
                       count,
                       llm_dir,
                       example_input,
                       input_shapes,
                       compress=None
                       ):
    """
    Model conversion and saving.
    """
    ov_model = ov.convert_model(model, example_input=example_input)
    names = get_param_names(ov_model)
    if 'kv_caches' in names:
        ov_model.remove_parameter(names['kv_caches'])
        names.pop('kv_caches')

    filtered_input_shapes = {key: input_shapes[key] for key in names}
    ov_model.reshape(filtered_input_shapes)
    print(f"Saving model to [{count}.xml]...")

    ov_model = parse_and_rename_layers(ov_model) # for transformer blocks
    ov_model = make_stateful(ov_model)

    if compress:
        ov_model = compress_weights(ov_model, mode=compress)
    ov.save_model(ov_model, llm_dir / f"{count}.xml")

    # Updating inputs
    for output in ov_model.outputs:
        names = output.get_names()
        shape = eval(output.get_shape().to_string()) # this is pretty jank - assumes no partial shapes
        for name in names:
            if name in example_input:
                example_input[name] = torch.randn(shape)
                input_shapes[name] = example_input[name].shape