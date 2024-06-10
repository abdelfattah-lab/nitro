import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov
import pdb
import time

import openvino.runtime
import numpy as np

from models import LinearModule, SoftmaxModule, Function
from utils import probe, visualize_example
import pdb

model = Function(4, 4, dim=1)
input = torch.randn(1, 4)

device = "HETERO:CPU,GPU,NPU"
core = ov.Core()

# --------- Using OpenVINO conversion ---------

converted_model = ov.convert_model(model, example_input=input)
converted_model.reshape([1, 4])

# ----------- Heterogenous Computing -----------

# under CPU -> GPU -> NPU, by default they are all set to CPU
# supported_ops = core.query_model(converted_model, device)

# # Manually taking different processing and specifying GPU to CPU
# # note that [supported_ops] is just a returned dictionary.
# npu_ops = [
#     "__module.linear.linear/aten::linear/Add",
#        "__module.linear.linear/aten::linear/MatMul",
#        ]
# for op in npu_ops:
#     supported_ops[op] = "NPU"

# gpu_ops = [
#        "__module.softmax/aten::softmax/Softmax",
#        "Result_15"
#        ]
# for op in gpu_ops:
#     supported_ops[op] = "GPU"

# # Iterate through each node, and update the deviceaffinity
# for node in converted_model.get_ops():
#     affinity = supported_ops[node.get_friendly_name()]
#     print(node.get_type_name())
#     node.get_rt_info()["affinity"] = affinity

visualize_example(converted_model, file=f"image.svg")

for dev in ["GPU", "CPU", "HETERO:GPU,CPU", "HETERO:CPU,GPU"]:
    compiled_model = core.compile_model(model=converted_model,
                                        device_name=dev,
                                        # config = {"PERFORMANCE_HINT" : "THROUGHPUT"}
                                        )
    
    start = time.time()
    for _ in range(1000): output = compiled_model(input)[0]
    print(f" {dev} time elapsed:", time.time() - start)
    print(dev, output, "\n")