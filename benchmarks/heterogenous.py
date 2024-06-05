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

model = Function(100, 100, dim=1)
input = torch.randn(1, 100)

# --------- Using OpenVINO conversion ---------
converted_model = ov.convert_model(model, example_input=input)
converted_model.reshape([1, 100])

# ----------- Heterogenous Computing -----------
device = "HETERO:GPU,CPU"
core = ov.Core()
supported_ops = core.query_model(converted_model, device)

ops = [
    "__module.linear.linear/aten::linear/Add",
       "__module.linear.linear/aten::linear/MatMul",
       "__module.softmax/aten::softmax/Softmax",
       "Result_15"
       ]
for op in ops:
    supported_ops[op] = "NPU"
print(supported_ops)


for node in converted_model.get_ops():

    affinity = supported_ops[node.get_friendly_name()]
    print(node.get_type_name())
    node.get_rt_info()["affinity"] = affinity

visualize_example(converted_model, file=f"image.svg")

for dev in ["GPU", "CPU", "HETERO:GPU,CPU"]:
    compiled_model = core.compile_model(model=converted_model,
                                        device_name=dev,
                                        # config = {"PERFORMANCE_HINT" : "THROUGHPUT"}
                                        )
    
    start = time.time()
    output = compiled_model(input)[0]
    print(f" {dev} time elapsed:", time.time() - start)
    # print(dev, output, "\n")
pdb.set_trace()