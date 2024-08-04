import torch
import torch.nn as nn
import openvino as ov
from typing import Optional, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from pathlib import Path

from converter.bindings import get_args, get_model
from converter.input_generators import generate_auto, generate_shape
from converter.chunk_conversion import conversion_wrapper

import openvino_tokenizers as ot
from transformers import AutoTokenizer

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

    def initialize_model(self, args:Any=None):
        # Saves and loads the weights
        self._save_weights()

        # Loading model and directory
        self.pytorch_model = get_model(self.model_name)
        self.model_args = get_args(self.model_name)

        self.pytorch_model.load_state_dict(torch.load(self.directory / "model_weights.pth"))
    
    def convert(self):
        # Generates example inputs for conversion
        example_inputs = generate_auto(self.model_args, "x", "mask", "freqs_cis", "kv_caches")
        shapes = generate_shape(example_inputs)

        # Conversion to IR Format / Chunking
        self._convert_chunks(example_inputs, shapes)

        self.generate_tokenizers()


    def _convert_chunks(self, example_inputs, shapes) -> None:
        """
        Converts the PyTorch LLM model into chunks.
        """
        count = 0
        print("Converting embedding layer...")
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = True, False, False
        conversion_wrapper(self.pytorch_model, count, self.llm_directory, example_inputs, shapes)
        count += 1

        ############ Chunking transformer layers ############
        print("Converting transformer layers...")
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = False, True, False

        for offset in range(0, self.model_args.n_layers, self.model_args.chunk_size):
            print(f" > Block: {offset}-{offset + self.model_args.chunk_size-1}")
            self.pytorch_model.offset = offset

            conversion_wrapper(self.pytorch_model, count, self.llm_directory, example_inputs, shapes)
            count += 1

        ############ Chunking output layer ############
        self.pytorch_model.include_embedding, self.pytorch_model.include_transformer, self.pytorch_model.include_output = False, False, True
        print("Converting output layer...")

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
        print("Generating tokenizers...")
        hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
        ov.save_model(ov_tokenizer, self.tokenizer_directory / "openvino_tokenizer.xml")
        ov.save_model(ov_detokenizer, self.tokenizer_directory / "openvino_detokenizer.xml")