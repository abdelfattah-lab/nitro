## Model loading

import json, os
from typing import Any

def get_args_aux(model:str, args_class:Any, json_file_path:str):
    """
    Obtains the configuration file for the specified model.

    Params
        model (str): the model file, as defined by Hugging Face (e.g. meta-llama/Meta-Llama-3-8B).
        args_class (Any): the corresponding class for the model family.
        json_file (str): the location of the json file.
    """
    with open(json_file_path, 'r') as file:
    # Parse the JSON data into a Python object (dictionary)
        data = json.load(file)
        if model not in data:
            raise ValueError(f"{model} is not a support model.")
        
        attributes = data[model] # dictionary w/ all the attributes.
        
        args = args_class()
        for key, value in attributes.items():
            setattr(args, key, value)
        
        return args