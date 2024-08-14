from pytorch_model import *
import pytorch_model
import torch.nn as nn
from typing import Any

classes_hierarchy = {
    "Llama" : {
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B",
    },
    "Qwen2" : {
        "Qwen/Qwen2-7B"
    }
}

class_to_model = {
    "Llama" : LlamaModel,
    "Qwen2" : Qwen2Model,
}

class_to_args = {
    "Llama" : get_llama_args
}

bindings = {}
for key, values in classes_hierarchy.items():
    for value in values:
        bindings[value] = key
    

def _obtain_family(model_name:str) -> str:
    if model_name not in bindings:
        return ValueError("Model is not supported!")
    return bindings[model_name]

def _obtain_model_class(model_family:str) -> Any:
    """
    Obtains the class associated with a specific model family.
    
    Args:
        model_class (str): model class, such as "Llama".
    """
    return class_to_model[model_family]

def _obtain_args(model_family):
    """
    Returns the argument function associated with the model name.
    """
    return class_to_args[model_family]

def get_args(model_name:str) -> Any:
    """
    Returns a data class containing all the relevant parameters.
    """
    family = _obtain_family(model_name)
    return _obtain_args(family)(model_name)

def get_model(model_name:str, args=None) -> nn.Module:
    """
    Returns the model, with the correct arguments initialized. Parameters have not
    yet been loaded.
    """
    family = _obtain_family(model_name)
    model = _obtain_model_class(family)
    if not args:
        args = _obtain_args(family)(model_name)

    return model(args)