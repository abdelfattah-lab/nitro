# Random helper functions
import openvino as ov
import openvino.runtime.passes as passes
import torch
import torch.nn as nn
import os

def probe(model:ov.Model, probe_enabled:bool=True, *args):
    """
    Probe the attributes of an OV model, such as inputs and outputs.
    *args are the list of attributes you want to view.
    """
    if not probe_enabled: return
    for a in args:
        print(f"{a}:", model.__getattribute__(a))

def visualize_example(m:ov.Model, model_name:str='image'):
    """
    Produces a visualization in .dot format. Should save as an svg, or the
    output is a .dot file that is viewable from a dot loader, e.g.
    https://dreampuf.github.io/GraphvizOnline/
    """
    # file = _ensure_path_prefix(model_name)
    file = model_name + ".svg"
    pass_manager = passes.Manager()
    pass_manager.register_pass(passes.VisualizeTree(file_name=file))
    pass_manager.run_passes(m)

def _ensure_path_prefix(path:str, prefix:str="models") -> str:
    """
    Helper function to check if models exists.
    """
    normalized_path = os.path.normpath(path)
    parts = normalized_path.split(os.sep)
    
    if parts[0] != prefix: # Check if the first folder is prefix
        new_path = os.path.join(prefix, normalized_path)
    else:
        new_path = normalized_path
    return new_path

def _get_last_path_name(path:str) -> str:
    """
    Returns the name of the last path/directory.
    """
    normalized_path = os.path.normpath(path)
    last_path_name = os.path.basename(normalized_path)
    return last_path_name

def load_or_convert(model_name:str, model:nn.Module=None, force_update:bool=False, core:ov.Core=None, *args, **kwargs) -> ov.Model:
    """
    Checks if a model exists in a specified path. The relative path, given
    [model_name], will be models/[model_name]/[model_name].xml.
     
    If it exists, then invoke the core.read_model function. Otherwise, convert
    the PyTorch module and then save it to the path.
    """
    if core is None:
        core = ov.Core()

    # Path string standardization
    if model_name.endswith(".xml"):
        model_name = model_name[:-4] # trim the .xml
    path = _ensure_path_prefix(model_name)
    path = os.path.join(path, _get_last_path_name(path) + ".xml")
    
    # OpenVINO IR already exists: read the model
    if not force_update and os.path.exists(path):
        print(f"Path {path} found.")
        return core.read_model(path)
    
    if model is None:
        raise Exception("Path not found, and model is None. Provide a path that exists or a PyTorch model.")
    
    # Convert the specified module, and then save it
    print(f"Path {path} not found. Converting PyTorch model and then saving to path.")
    ov_model = ov.convert_model(model, *args, **kwargs)
    ov.save_model(ov_model, path) # saves the model
    
    return load_or_convert(model_name, model, force_update=False, *args, **kwargs) # because some compression may occur
