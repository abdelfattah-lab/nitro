import torch.nn as nn
import torch.nn.functional as F
import torch
import openvino as ov
import time

from models import SmallConv
from utils import load_or_convert, visualize_example

size =128
in_channels=100

# Import model
model = SmallConv(in_channels=in_channels, out_channels=40, kernel=2, stride=1, padding=3)
core = ov.Core()

ov_model = load_or_convert("conv_module", model, force_update=False, core=core, example_input=torch.randn(1,in_channels, size,size), input=[1,in_channels, size,size])
visualize_example(ov_model, "conv_module")

start = time.time()

compiled_model = ov.compile_model(model=ov_model, device_name="CPU")
print(f"Compiling time: {time.time() - start}")
start = time.time()
for _ in range(1000):
    input = torch.randn(1,in_channels, size,size)    
    output = compiled_model(input)
print(f"Inference time: {time.time() - start}")
# print(output)
