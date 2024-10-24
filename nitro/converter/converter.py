import torch
import torch.nn as nn
import openvino as ov
from typing import Optional, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from pathlib import Path

from nitro.converter.bindings import get_args, get_model
from nitro.converter.input_generators import generate_auto, generate_shape
from nitro.converter.chunk_conversion import conversion_wrapper

import openvino_tokenizers as ot
from transformers import AutoTokenizer, AutoConfig

import json
import os

@dataclass
class ConversionConfig:
    chunk_size:int = 16
    inference_size:int = 1

def from_dict(cls, data: dict):
    # Extract only the keys that are in the dataclass fields
    valid_keys = {key: data[key] for key in data if key in cls.__annotations__}
    return cls(**valid_keys)

class Converter:
    """
    Class to convert provided PyTorch models into OpenVINO IR formats.
    """

    def __init__(self,
                 model:str,
                 directory:Optional[str] = None,
                 model_args:Optional[Any] = None,
                 conversion_args:Optional[ConversionConfig] = None):
        """
        Initializes a generic Converter.

        Params:
            model: Name of the model as defined in Hugging Face (e.g. meta-llama/Meta-Llama-3).
            dir: The directory to save the OpenVINO IR formats. If not specified, saves to 'ir_model' at working directory.
            model_args (Any): The model arguments to be passed.
        """
        self.model_name = model
        self.model_args = model_args
        self.conversion_args = conversion_args
        if conversion_args is None:
            self.conversion_args = ConversionConfig()

        # Configuring directories
        if directory == None: directory = "ir_model"
        
        self.directory = Path(directory)
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        
        self.llm_directory = self.directory / "model"
        if not os.path.isdir(self.llm_directory):
            os.makedirs(self.llm_directory)

        self.tokenizer_directory = self.directory / "tokenizer"
        if not os.path.isdir(self.tokenizer_directory):
            os.makedirs(self.tokenizer_directory)

    @classmethod
    def convert(cls,
                model:str,
                directory:Optional[str] = None,
                args:Optional[Any] = None):
        """
        Converts a model.
        """

        converter = cls(model, directory, args)
        print("Initializing model")
        converter.initialize_model()
        print("Converting chunks")
        converter.convert_chunks()

    def initialize_model(self):
        # Saves and loads the weights
        self._save_weights()

        # Loading model and directory
        if not self.model_args:
            self.model_args = get_args(self.model_name)
        
        self.pytorch_model = get_model(self.model_name, self.model_args)
        
        # print(self.model_args)

        self.pytorch_model.load_state_dict(torch.load(self.directory / "model_weights.pth"))
    
    def convert_chunks(self, chunk:bool=True):
        # Generates example inputs for conversion
        example_inputs = generate_auto(self.model_args, self.conversion_args, "x", "mask", "freqs_cis", "kv_caches")
        shapes = generate_shape(example_inputs)

        # Conversion to IR Format / Chunking
        if chunk:
            self._convert_chunks(example_inputs, shapes)
        else:
            self._convert_entire(example_inputs, shapes)

    def _convert_entire(self, example_inputs, shapes) -> None:
        """
        Converts the entire PyTorch LLM model as one OpenVINO.
        """
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = True, True, True
        conversion_wrapper(self.pytorch_model, 0, self.llm_directory, example_inputs, shapes)

    def _convert_chunks(self, example_inputs, shapes) -> None:
        """
        Converts the PyTorch LLM model into chunks.
        """
        count = 0
        print("Converting embedding layer...")
        
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = True, False, False
        kv_caches = None
        if "kv_caches" in example_inputs:
            kv_caches = True
            global_kv_caches = example_inputs["kv_caches"]
            example_inputs["kv_caches"] = torch.randn((1, 1))
        conversion_wrapper(self.pytorch_model, count, self.llm_directory, example_inputs, shapes)
        count += 1

        ############ Chunking transformer layers ############
        print("Converting transformer layers...")
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = False, True, False

        
        self.pytorch_model.set_chunk_size(self.conversion_args.chunk_size) # TODO: this is jank.
        for offset in range(0, self.model_args.num_hidden_layers, self.conversion_args.chunk_size):
            # Update kv caches
            if kv_caches:
                local_kv_caches = {}
                for i in range(offset, offset + self.conversion_args.chunk_size):
                    local_kv_caches[f"cache_k_{i}"] = global_kv_caches[f"cache_k_{i}"]
                    local_kv_caches[f"cache_v_{i}"] = global_kv_caches[f"cache_v_{i}"]
                example_inputs["kv_caches"] = local_kv_caches
            print(f" > Block: {offset}-{offset + self.conversion_args.chunk_size-1}")
            self.pytorch_model.offset = offset

            conversion_wrapper(self.pytorch_model, count, self.llm_directory, example_inputs, shapes)
            count += 1

        ############ Chunking output layer ############
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = False, False, True
        print("Converting output layer...")

        if kv_caches:
            example_inputs["kv_caches"] = torch.randn((1, 1))

        conversion_wrapper(self.pytorch_model, count, self.llm_directory, example_inputs, shapes)
        count += 1

    def _save_weights(self) -> None:
        """
        Obtains the weights from Hugging Face.
        """
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

    def generate_tokenizers(self):
        """
        Generates tokenizers.
        """

        ## NOT CURRENTLY USED.
        print("Generating tokenizers...")
        hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)

        ov.save_model(ov_tokenizer, self.tokenizer_directory / "openvino_tokenizer.xml")
        ov.save_model(ov_detokenizer, self.tokenizer_directory / "openvino_detokenizer.xml")