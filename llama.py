from base import LLMBase
from model.llama.rewritten_models import Transformer
from model.llama.config import ModelArgs
from model.llama.helpers import precompute_freqs_cis_rect

import openvino as ov
import torch
import os
import shutil
from pathlib import Path


# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from modifiers import parse_and_rename_layers, make_stateful, conversion_wrapper, get_shape_dict
import re

class Llama(LLMBase):
    def __init__(self, model_dir: Path | str,
                 args:ModelArgs,
                 count:int,
                 device:str,
                 compile:bool=True,
                 compress:bool=True,
                 verbose:bool=False):
        
        super().__init__(model_dir, args, count, device, compile, compress, verbose)
        # Custom inputs
        self._freqs_cis = self._precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta)
        self.freqs_cis = self._freqs_cis[0:1]
        self.mask = torch.full([args.max_batch_size, args.n_heads, 1, args.max_seq_len], float('-inf'))
    
    @classmethod
    def from_pretrained(cls,
                        model_dir: Path | str,
                        max_batch_size: int,
                        max_seq_len: int,
                        chunk_size: int,
                        export:bool = False,
                        *args, **kwargs
                        ) -> "Llama":
        """
        Generates the Llama model from source code.
        """

        args = ModelArgs()
        args.chunk_size = chunk_size
        args.max_seq_len = max_seq_len
        args.max_batch_size = max_batch_size

        model_dir = Path(model_dir)
        llm_dir = model_dir / "model"

        if export:
            ############################################
            ###     GENERATION OF THE MAIN MODEL     ###
            ############################################

            print("Clearing directory", llm_dir)
            if os.path.exists(llm_dir) and os.path.isdir(llm_dir):
                # Iterate over all files and subdirectories in the directory
                for filename in os.listdir(llm_dir):
                    file_path = os.path.join(llm_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Remove the file or link
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove the subdirectory
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

            checkpoint = torch.load(model_dir / "consolidated.00.pth")
            print("Creating PyTorch model...")
            model = Transformer(args, chunk_size=chunk_size)
            print("Loading checkpoint...")
            model.load_state_dict(checkpoint, strict=True)

            count = 0 # counting the number of chunks.

            freqs_cis = precompute_freqs_cis_rect(
                args.dim // args.n_heads,
                args.max_seq_len * 2,
                args.rope_theta
            )

            # Preparing inputs
            L = 1
            example_input = {
                "x"         : torch.randint(0, args.vocab_size, [args.max_batch_size, L]),
                "mask"      : torch.full([args.max_batch_size, args.n_heads, L, args.max_seq_len], float('-inf')),
                "freqs_cis" : freqs_cis[0:L],
                "params"    : {}
            }

            for i in range(args.n_layers):
                example_input["params"][f"cache_k_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
                example_input["params"][f"cache_v_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
                
            input_shapes = get_shape_dict(example_input)
            for key in input_shapes["params"]:
                input_shapes[key] = input_shapes["params"][key]
            input_shapes.pop("params")

            ############ Chunking embedding layer ############
            print("Converting embedding layer...")
            model.include_embedding, model.include_transformer, model.include_output = True, False, False
            conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
            count += 1

            ############ Chunking transformer layers ############
            print("Converting transformer layers...")
            model.include_embedding, model.include_transformer, model.include_output = False, True, False

            for offset in range(0, args.n_layers, args.chunk_size):
                print(f" > Block: {offset}-{offset + args.chunk_size-1}")
                model.offset = offset

                conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
                count += 1

            ############ Chunking output layer ############
            model.include_embedding, model.include_transformer, model.include_output = False, False, True
            print("Converting output layer...")

            conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
            count += 1

            #######################################
            ###            TOKENIZER            ###
            #######################################

            import openvino_tokenizers as ot

            print("Generating tokenizers...")
            hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
            ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
            ov.save_model(ov_tokenizer, model_dir / "tokenizer" / "openvino_tokenizer.xml")
            ov.save_model(ov_detokenizer, model_dir / "tokenizer" / "openvino_detokenizer.xml")
        
            del model

        else:
            count = 0
            pattern = re.compile(r'^(\d+)\.xml$')
            for filename in os.listdir(llm_dir):
                match = pattern.match(filename)
                if match:
                    number = int(match.group(1))
                    if number > count:
                        count = number
            count += 1
            print(count)

        return Llama(model_dir, args, count, **kwargs)

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        precompute_freqs_cis, outputting as real numbers
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float16)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_cis = torch.view_as_real(freqs_cis)
        return freqs_cis

    def _iterate(self, token:int, step:int) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, returns the logits.
        """
        self._update_freqs_cis(step)
        self._update_mask(step)
        
        self.parallel_inputs["freqs_cis"] = self.freqs_cis
        self.parallel_inputs["mask"] = self.mask
        self.series_inputs["x"] = token

        output = self.model(self.parallel_inputs, self.series_inputs)
        return output
    

    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step:] = 0.0

    def _update_freqs_cis(self, step:int) -> None:
        self.freqs_cis = self._freqs_cis[step:step+1]

if __name__ == "__main__":

    llama = Llama.from_pretrained(model_dir="npu_model", max_batch_size=1,
                                  max_seq_len=128, chunk_size=32, export=False,
                                  device="NPU",
                                  compile=True,
                                  compress=False,
                                  verbose=True)

    output = llama.generate(prompt=["I was wondering why you"],
                            max_new_tokens=30)

    print(output)