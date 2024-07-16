from model.rewritten_models import Transformer
from model.config import ModelArgs
import torch
from model.helpers import precompute_freqs_cis_rect
from pathlib import Path
import re

from rename import parse_and_rename_layers

args = ModelArgs(n_layers=32, max_batch_size=1, max_seq_len=128,
                 outside_embedding=True, outside_end_layer=True, 
                 n_sub_layers=16)

# Abbreviations for simplification
DIM = args.dim
B = args.max_batch_size
L = 1
ML = args.max_seq_len
KVH = args.n_kv_heads
NH = args.n_heads
HD = args.dim // args.n_heads
TYPE = torch.float32
RT = args.rope_theta
LAYERS = args.n_layers

freqs_cis = precompute_freqs_cis_rect(
    args.dim // args.n_heads,
    args.max_seq_len * 2,
    args.rope_theta)

pos = 5
freqs_cis = freqs_cis[5:5+L]

import openvino as ov

# example_input = {
#     "x"         : torch.randint(0, 128256, [B, L]),
#     # "x"         : torch.randn([B, L, DIM]),
#     "mask"      : torch.full([B, NH, L, ML], float('-inf'),  dtype=TYPE),
#     "freqs_cis" : freqs_cis,
#     "cache_k"   : torch.zeros([LAYERS, B, ML, KVH, HD], dtype=TYPE),
#     "cache_v"   : torch.zeros([LAYERS, B, ML, KVH, HD], dtype=TYPE),
# }

example_input = {
    "x"         : torch.randint(0, 128256, [B, L]),
    "mask"      : torch.full([B, NH, L, ML], float('-inf'),  dtype=TYPE),
    "freqs_cis" : freqs_cis,
    "params"    : {}
}

if args.outside_embedding:
    example_input["x"] = torch.randn([B, L, DIM], dtype=TYPE)

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

# PyTorch conversion to OpenVINO
core = ov.Core()
print("Loading checkpoint...")
CHECKPOINT_DIR = Path("Meta-Llama-3-8B")
checkpoint = torch.load(CHECKPOINT_DIR / "consolidated.00.pth")
count = 1
for offset in range(0, args.n_layers, args.n_sub_layers):

    # Configuring input shapes
    example_input["params"] = {}
    
    for i in range(args.n_sub_layers):
        example_input["params"][f"cache_k_{i}"] = torch.zeros([B, ML, KVH, HD], dtype=TYPE)
    for i in range(args.n_sub_layers):
        example_input["params"][f"cache_v_{i}"] = torch.zeros([B, ML, KVH, HD], dtype=TYPE)
    
    input_shapes = get_shape_dict(example_input)
    
    for key in input_shapes["params"]:
        input_shapes[key] = input_shapes["params"][key]
    input_shapes.pop("params")

    

    # Loading PyTorch model
    print("Creating model...")
    model = Transformer(args)
    missing = model.load_state_dict(checkpoint, strict=False)

    print(model)
    # print(missing)
    print()
    print(model.layers[offset + args.n_sub_layers:])
    print(model.layers[:offset])
    print()
    del model.layers[offset + args.n_sub_layers:] # Removing layers
    del model.layers[:offset]
    model.n_layers = args.n_sub_layers

    ov_model = ov.convert_model(model, example_input=example_input)
    ov_model.reshape(input_shapes)
    
    # Rename layers if offset
    if offset != 0:
        for node in ov_model.get_ops():
            name = node.get_friendly_name()
            operation = node.get_type_name()
            if operation == 'Parameter' and name.startswith("cache"):
                new_name = re.sub(r'_(\d+)$', lambda x: f"_{int(x.group(1)) + offset}", name)
                node.set_friendly_name(new_name)
            else:
                new_name = re.sub(r'layers\.(\d+)', lambda x: f"layers.{int(x.group(1)) + offset}", name)
                node.set_friendly_name(new_name)
       
    ov.save_model(ov_model, f"npu_model/llama_{count}.xml")

    # Rewriting inputs and output names for the cache to be more friendly
    parse_and_rename_layers(f'npu_model/llama_{count}.xml')

    del model
    del ov_model

    count += 1


