import torch.nn as nn
import torch.nn.functional as F
import torch
import openvino as ov
import time

from models import SmallConv
from utils import load_or_convert, visualize_example

size = 64
in_channels=256
out_channels=256

# Import model
model = SmallConv(in_channels=in_channels, out_channels=256, kernel=3, stride=1, padding=1)
core = ov.Core()

ov_model = load_or_convert("conv_module", model, force_update=False, example_input=torch.randn(1,in_channels, size,size), input=[1,in_channels, size,size])
visualize_example(ov_model, "conv_module")

for device in ["GPU", "CPU", "NPU"]:
    start = time.time()
    compiled_model = ov.compile_model(model=ov_model, device_name=device)
    print(f"{device} Compiling time: {time.time() - start}")
    start = time.time()
    for _ in range(1000):
        input = torch.randn(1,in_channels, size,size)    
        output = compiled_model(input)
    print(f"{device} Inference time: {time.time() - start}")
    # print(output)
