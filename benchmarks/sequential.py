import torch.nn as nn
import torch.nn.functional as F
import torch
import openvino as ov
import time

from models import Iterative
from utils import load_or_convert, visualize_example

features = 64
reps = 500

# Import model
model = Iterative(features, reps=reps)
core = ov.Core()

ov_model = load_or_convert("seq_module", model, force_update=True, core=core, example_input=torch.randn(1,features), input=[1,features])
# visualize_example(ov_model, "seq_module")

for device in ["CPU", "GPU", "NPU"]:
    start = time.time()
    compiled_model = ov.compile_model(model=ov_model, device_name=device)
    print(f"{device} Compiling time: {time.time() - start}")
    start = time.time()
    for _ in range(1000):
        input = torch.randn(1,features)    
        output = compiled_model(input)
    print(f"{device} Inference time: {time.time() - start}")
    print()
