#!/home/abdelfattah/openvino-env/bin/activate
import torch.nn as nn
import torch.nn.functional as F
import torch
from models import DualConvConcat
from utils import load_or_convert, visualize_example, probe
import openvino as ov
import time

IN_CHANNELS=30
SIZE=128
UPDATE=True

# Import model
model = DualConvConcat(in_channels=IN_CHANNELS, out_channels1=32, out_channels2=32,
                      kernel_size1=5, kernel_size2=5,
                       padding1=2, padding2=2)


ov_model = load_or_convert("parallel_model", model, force_update=False, input=[("x", [1,IN_CHANNELS,SIZE,SIZE]), ("y", [1,IN_CHANNELS,SIZE,SIZE])])
visualize_example(ov_model, "parallel_model")

# Device mapping configuration - needs to be manually configured which
# is SUPER tedious
convolution_1 = {
    "Constant_6",
    "Parameter_2",
    "Convert_8",
    "Constant_12",
    "Convolution_10",
    "Convert_14",
    "Add_16",
}

convolution_2 = {
    "Constant_18",
    "Parameter_4",
    "Convert_20",
    "Constant_24",
    "Convolution_22",
    "Convert_26",
    "Add_28",
}

concat = {
    "Concat_30",
    "Result_32"
}

all_names = set.union(convolution_1, convolution_2, concat) # verification step

setup = [
    "GPU",
    "GPU",
    "GPU"
         ]
print(f"  {setup[0]}       {setup[1]}")
print(f"     \     / ")
print(f"       {setup[2]}")

for node in ov_model.get_ops():
    id = node.get_name()
    assert id in all_names
    if id in convolution_1:
        node.get_rt_info()["affinity"] = setup[0]
    elif id in convolution_2:
        node.get_rt_info()["affinity"] = setup[1]
    elif id in concat:
        node.get_rt_info()["affinity"] = setup[2]

start = time.time()
core = ov.Core()
# print(core.get_property("OPENVINO_HETERO_VISUALIZE"))
compiled_model = core.compile_model(model=ov_model, device_name="HETERO")
print(f"Compiling time: {time.time() - start}")

infer_request = compiled_model.create_infer_request()
start = time.time()
for _ in range(1000):
    input1 = torch.randn(1, IN_CHANNELS, SIZE, SIZE)
    input2 = torch.randn(1, IN_CHANNELS, SIZE, SIZE)
    
    infer_request.infer((input1, input2))

print(f"Inference time: {time.time() - start}")
# print(output)

