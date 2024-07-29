import torch
import torch.nn as nn
import openvino as ov
from typing import Optional, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from pathlib import Path

from converter.bindings import get_args, get_model
import os

class Converter():
    """
    Class to convert provided PyTorch models into OpenVINO IR formats.
    """

    def __init__(self, model:str, directory:Optional[str] = None):
        """
        Initializes a generic Converter.

        Params:
            model (str): Name of the model as defined in Hugging Face (e.g. meta-llama/Meta-Llama-3).
            dir (str): The directory to save the OpenVINO IR formats. If not specified, saves to 'ir_model' at working directory.
        """
        self.model_name = model

        if directory == None: directory = "ir_model"
        self.directory = Path(directory)
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # Saves and loads the weights
        self._save_weights()

        # Loading model and directory
        self.pytorch_model = get_model(self.model_name)
        self.model_args = get_args(self.model_name)

        self.pytorch_model.load_state_dict(torch.load(self.directory / "model_weights.pth"))

        # Generates example inputs for conversion
        self._generate_example_inputs_for_conversion()

        # Conversion to IR Format / Chunking
        self.convert()


    def convert(self) -> ov.Model:
        """
        Converts the PyTorch LLM model into chunks.
        """
        
        
    def _generate_example_inputs_for_conversion(self):
        pass


    def _save_weights(self) -> None:
        if not os.path.isfile(os.path.join(self.directory, "model_weights.pth")):
            print("Weights not found, loading model from Hugging Face:")
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model_weights = model.state_dict()

            # REMOVE model. PREFIX IF NEEDED.
            prefix_to_remove = 'model.'

            # Create a new dictionary with the updated keys
            new_checkpoint = {}
            for key, value in model_weights.items():
                # Remove the prefix from each key
                new_key = key[len(prefix_to_remove):] if key.startswith(prefix_to_remove) else key
                new_checkpoint[new_key] = value
            
            torch.save(new_checkpoint, os.path.join(self.directory, "model_weights.pth"))
            del model_weights
            del model


    def _stateful_model(self, suffix_match:str="_out") -> ov.Model:
        """
        Converts the models into stateful models. Stateful models are created by
        matching parameter names [x] with result names [x{suffix_match}]. The
        shapes must be the same.
        """
        pass



    def _rename(self):
        pass

    def generate_tokenizers(self):
        """
        Generates.
        """
        pass

Converter("meta-llama/Meta-Llama-3-8B", "npu_model")