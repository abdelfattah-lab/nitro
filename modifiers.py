import xml.etree.ElementTree as ET
import shutil
import os
import numpy as np
from typing import Any

import re
import openvino as ov
from openvino.runtime import Model
from openvino.runtime.passes import Manager, MakeStateful
import torch

def get_shape_dict(d):
    if isinstance(d, dict):
        shape_dict = {}
        for key, value in d.items():
            shape_dict[key] = get_shape_dict(value)
        return shape_dict
    elif isinstance(d, torch.Tensor):
        return d.shape
    else:
        raise TypeError("Unsupported type: {}".format(type(d)))
                
def get_param_names(model:ov.Model) -> dict[str, Any]:
    params = model.get_parameters()
    names = {}
    for p in params:
        names[p.get_friendly_name()] = p
    return names

def conversion_wrapper(model, count, llm_dir, example_input, input_shapes):
    """
    Model conversion and saving.
    """
    ov_model = ov.convert_model(model, example_input=example_input)
    names = get_param_names(ov_model)
    if 'params' in names:
        ov_model.remove_parameter(names['params'])
        names.pop('params')

    filtered_input_shapes = {key: input_shapes[key] for key in names}
    # print(filtered_input_shapes)
    ov_model.reshape(filtered_input_shapes)
    print(f"Saving model to [{count}.xml]...")

    ov_model = parse_and_rename_layers(ov_model) # for transformer blocks
    ov_model = make_stateful(ov_model)

    print(ov_model)

    ov.save_model(ov_model, llm_dir / f"{count}.xml")

    # Updating inputs
    for output in ov_model.outputs:
        names = output.get_names()
        shape = eval(output.get_shape().to_string()) # this is pretty jank - assumes no partial shapes
        for name in names:
            if name in example_input:
                example_input[name] = torch.randn(shape)
                input_shapes[name] = example_input[name].shape

def parse_and_rename_layers(model: Model):
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

def parse_and_rename_layers_old(file, output_file=None):
    if output_file is None:
        output_file = file
    tree = ET.parse(file)
    root = tree.getroot()

    # Find all layers
    layers = root.findall('.//layer')
    
    # Find all edges
    edges = root.findall('.//edge')
    
    for layer in layers:
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        
        # Check if the layer type is "Result"
        if layer_type == "Result":
            input_ports = layer.findall('.//input/port')
            for port in input_ports:
                dims = port.findall('.//dim')
                dim_values = [int(dim.text) for dim in dims]
                
                # Check for cache_v or cache_k
                if len(dim_values) == 4:
                    for edge in edges:
                        if edge.get('to-layer') == layer_id:
                            from_layer_id = edge.get('from-layer')
                            from_layer = root.find(f".//layer[@id='{from_layer_id}']")
                            if from_layer is not None:
                                from_layer_name = from_layer.get('name')
                                module_index = int(from_layer_name.split('.')[2].split('/')[0])
                                output_ports = from_layer.findall('.//output/port')
                                for port in output_ports:
                                    port_name = port.get('names')
                                    if port_name is not None:
                                        if port_name.startswith('cache_v'):
                                            new_name = f'cache_v_{int(module_index)}_out'
                                        elif port_name.startswith('cache_k'):
                                            new_name = f'cache_k_{int(module_index)}_out'
                                        else:
                                            continue
                                        layer.set('name', new_name)
                                        port.set('names', new_name)
                                        break
                            else:
                                layer.set('name', 'logit')
                
                # Check for logit
                elif len(dim_values) == 3:
                    layer.set('name', 'logit')
                    # Update output port names in the "from-layer"
                    for edge in edges:
                        if edge.get('to-layer') == layer_id:
                            from_layer_id = edge.get('from-layer')
                            from_layer = root.find(f".//layer[@id='{from_layer_id}']")
                            if from_layer is not None:
                                output_ports = from_layer.findall('.//output/port')
                                for port in output_ports:
                                    port_name = port.get('names')
                                    if port_name is not None:
                                        # Rename the output port to "logits"
                                        port.set('names', 'logits')
                                        break
    
    # Save the modified XML
    tree.write(output_file)

    # Copy the BIN, if different
    if output_file != file:
        input_bin_file = os.path.splitext(file)[0] + '.bin'
        output_bin_file = os.path.splitext(output_file)[0] + '.bin'
        shutil.copyfile(input_bin_file, output_bin_file)

from openvino.runtime import opset6
import numpy as np

def make_stateful(model:  Model):

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
